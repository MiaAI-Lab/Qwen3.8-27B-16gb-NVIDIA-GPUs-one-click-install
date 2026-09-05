#!/usr/bin/env python3
"""Framework-agnostic core of the built-in web UI.

`ChatUI` owns everything that is not HTTP plumbing: static files, sessions on
disk, the approval hand-shake, and the event stream for one turn. Two thin
adapters mount it (tools/chatui.py): aiohttp inside the model server, and the
stdlib dev server for working on the UI without loading a model.

Handlers return either a `Response` (status, content type, bytes) or a
`Stream` (a generator of JSON-serialisable events, sent as SSE).

Sessions live in <root>/sessions/*.json and hold the OpenAI-shaped message
list, so what the UI renders and what the model is sent never drift apart.
"""
from __future__ import annotations

import json
import mimetypes
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import webui_models
import webui_picker
import webui_providers
import webui_tools
from webui_agent import ModelClient, run_turn, system_prompt

UI_DIR = Path(__file__).resolve().parent / "webui"
MAX_TITLE = 60
TURN_TTL = 900          # keep a finished turn replayable for a quarter hour
SEARCH_BLOB_CHARS = 200_000   # per conversation, enough for any real transcript
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}


@dataclass
class Response:
    status: int = 200
    content_type: str = "application/json"
    body: bytes = b""

    @staticmethod
    def json(obj, status=200):
        return Response(status, "application/json",
                        json.dumps(obj).encode("utf-8"))

    @staticmethod
    def error(message, status=400):
        return Response.json({"error": {"message": message}}, status)


@dataclass
class Stream:
    events: object          # generator of dicts
    cancel: object = None   # threading.Event the adapter sets on disconnect


class Turn:
    """A turn that belongs to the conversation, not to the browser window.

    The generator runs in its own thread and appends every event here; HTTP
    responses are only *views* over this log. Closing the tab drops a viewer,
    which is not a reason to stop working - the tools keep running, the answer
    is still saved, and reopening replays the whole turn from the start."""

    def __init__(self, sid, cancel):
        self.sid = sid
        self.cancel = cancel
        self.events: list = []
        self.done = False
        self.started = time.time()
        self.finished = None
        self._cond = threading.Condition()

    def append(self, event):
        with self._cond:
            self.events.append(event)
            self._cond.notify_all()

    def finish(self):
        with self._cond:
            self.done = True
            self.finished = time.time()
            self._cond.notify_all()

    def follow(self, start=0):
        """Yield everything from `start` onward, waiting for more until the
        turn ends. Several viewers can follow the same turn at once."""
        index = max(0, int(start or 0))
        while True:
            with self._cond:
                while index >= len(self.events) and not self.done:
                    self._cond.wait(timeout=1.0)
                if index >= len(self.events):
                    return                     # caught up and the turn is over
                batch = self.events[index:]
                index = len(self.events)
            for event in batch:
                yield event

    def summary(self):
        return {"running": not self.done, "events": len(self.events),
                "seconds": round((self.finished or time.time()) - self.started, 1)}


@dataclass
class _Pending:
    """One thing the model is waiting on the browser for: an approval, or the
    answer to a question."""

    event: threading.Event = field(default_factory=threading.Event)
    decision: str = "deny"
    answer: str = ""


PWA_FILES = frozenset({
    "/manifest.webmanifest", "/sw.js",
    "/icon-192.png", "/icon-512.png", "/icon-maskable-512.png", "/apple-touch-icon.png",
})


class ChatUI:
    def __init__(self, root: Path, cfg: dict, model_base: str, model_id: str,
                 context_length=None, vision=False):
        self.root = Path(root)
        self.cfg = cfg
        self.model_id = model_id
        self.context_length = context_length
        self.vision = vision
        self.model_base = model_base
        self.client = ModelClient(model_base, model_id,
                                  api_key=cfg.get("API_KEY", "local"))
        self._clients: dict[tuple, ModelClient] = {}
        self.registry = webui_tools.build_registry(cfg)
        self.sessions_dir = self.root / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._pending: dict[str, _Pending] = {}
        self._cancel: dict[str, threading.Event] = {}
        self._always: dict[str, set] = {}
        # tool state that has to survive between turns of one chat: which files
        # have been read, the current plan, background jobs
        self._contexts: dict[str, webui_tools.ToolContext] = {}
        self._save_locks: dict[str, threading.Lock] = {}
        self._turns: dict[str, Turn] = {}     # work outlives the browser window
        self._search_cache: dict[str, tuple] = {}   # file name -> (mtime, blob)
        self._active = 0                 # turns in flight, not a single flag
        self._lock = threading.Lock()
        self.default_workspace = str(
            Path(cfg.get("AGENT_WORKSPACE") or (self.root / "workspace")).expanduser())
        self.max_steps = int(cfg.get("AGENT_MAX_STEPS") or 8)
        # The server usually binds 0.0.0.0 so other devices can reach /v1. The
        # UI is a different matter: agent mode writes files and runs commands
        # on this machine, and there is no login. So it answers only the
        # machine it runs on until UI_LAN says otherwise.
        self.allow_lan = str(cfg.get("UI_LAN", "0")).strip().lower() in (
            "1", "true", "yes", "on")
        # Host names that may reach the UI even though it is not open to the
        # LAN. This is what makes `tailscale serve` work without exposing the
        # port: the proxy connects from loopback, so the peer check still
        # holds, and only the name has to be trusted.
        self.extra_hosts = {h.strip().lower()
                            for h in str(cfg.get("UI_HOSTS", "")).split(",")
                            if h.strip()}
        # native|browser: which folder picker the Agent's workspace chip opens
        self.picker = (cfg.get("FOLDER_PICKER") or "native").strip().lower()
        # the dev server has no model behind it, so it answers /health itself
        self.fake_health = False
        self._counters = {"prompt": 0, "completion": 0, "busy": False}
        # set by the launcher: only a supervised server can restart itself into
        # another model, because something has to start it again
        self.supervised = str(os.environ.get("SIMPLEX_SUPERVISED", "")).strip() in (
            "1", "true", "yes")
        self.on_restart = None      # callable installed by the server adapter

    def _origin_ok(self, method, headers):
        """A browser on another site can reach a loopback server two ways: DNS
        rebinding (the peer IP really is 127.0.0.1) and plain cross-site POSTs.
        Both are stopped by looking at what the browser *thinks* it is talking
        to, so the peer address is not the only gate."""
        if headers is None:
            return True, ""      # embedded/in-process call, not a browser
        # header names arrive in whatever case the client chose (urllib sends
        # "X-simplex-ui"), so normalise once
        items = (headers or {}).items() if hasattr(headers, "items") else (headers or [])
        lower = {str(k).lower(): str(v) for k, v in items}
        get = lower.get
        host = (get("host") or "").strip()
        hostname = host.rsplit(":", 1)[0].strip("[]").lower() if host else ""
        if (hostname and not self.allow_lan
                and hostname not in LOCAL_HOSTS
                and hostname not in self.extra_hosts):
            return False, (f"'{hostname}' is not an address this UI answers on - "
                           f"open it at http://127.0.0.1, add the name to "
                           f"UI_HOSTS in .env (for a Tailscale or reverse-proxy "
                           f"hostname), or set UI_LAN=1 to allow your network")
        site = (get("sec-fetch-site") or "").lower()
        if site and site not in ("same-origin", "none"):
            return False, "cross-site requests are refused"
        origin = (get("origin") or "").strip()
        if origin and host:
            if origin.split("//")[-1].lower() != host.lower():
                return False, "cross-origin requests are refused"
        elif not origin and method in ("POST", "PUT", "DELETE") and not site:
            # an old browser with neither header: allow only same-host GETs to
            # have gotten this far, and require the fetch marker we send
            if not get("x-simplex-ui"):
                return False, "this request did not come from the chat UI"
        return True, ""

    @staticmethod
    def _is_local(remote):
        if not remote:
            return True                     # no peer information: dev server
        host = str(remote).strip().strip("[]").split("%")[0]
        return host in ("127.0.0.1", "::1", "localhost", "::ffff:127.0.0.1") \
            or host.startswith("127.")

    # ------------------------------------------------------------ static ----

    def _static(self, name):
        path = (UI_DIR / name).resolve()
        if UI_DIR.resolve() not in path.parents or not path.is_file():
            return Response(404, "text/plain", b"not found")
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript",):
            ctype += "; charset=utf-8"
        return Response(200, ctype, path.read_bytes())

    # ---------------------------------------------------------- sessions ----

    def _path(self, sid):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", sid or ""):
            return None
        return self.sessions_dir / f"{sid}.json"

    def _load(self, sid):
        p = self._path(sid)
        if not p or not p.is_file():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            return None

    def _save(self, session, touch=True):
        """Atomic, and serialised per session: a rename, a rewind and the end of
        a turn can all land at once, and a shared temp file name would let one
        writer publish another's half-written JSON.

        touch=False is for changes that are not activity - reordering pinned
        chats should not make them look freshly used once unpinned."""
        if touch:
            session["updated"] = time.time()
        p = self._path(session["id"])
        if not p:
            return
        with self._save_locks.setdefault(session["id"], threading.Lock()):
            tmp = p.with_suffix(f".{uuid.uuid4().hex[:8]}.tmp")
            try:
                tmp.write_text(json.dumps(session, indent=1), encoding="utf-8")
                tmp.replace(p)
            finally:
                if tmp.exists():
                    tmp.unlink(missing_ok=True)

    def _append_turn(self, sid, new_messages, plan=None, extra=None):
        """Re-read, append, write. The turn started from a snapshot; anything
        that happened to the session meanwhile (a rename, another edit) has to
        survive, and earlier turns keep their reasoning because we never write
        the model-facing copy back."""
        session = self._load(sid) or {"id": sid, "messages": []}
        session["messages"] = (session.get("messages") or []) + list(new_messages)
        if plan is not None:
            session["plan"] = plan
        for key, value in (extra or {}).items():
            session[key] = value
        self._save(session)
        return session

    def _local(self):
        return {"name": "This computer", "base_url": self.model_base,
                "api_key": self.cfg.get("API_KEY", "local"), "model": self.model_id}

    def _client_for(self, target):
        """One client per endpoint+model, kept so a chat does not rebuild it on
        every turn. The local one is whatever this instance was built with -
        which the dev server replaces with a mock."""
        if not target.get("remote"):
            return self.client
        key = (target["base_url"], target["model"], target["api_key"])
        client = self._clients.get(key)
        if client is None:
            client = ModelClient(target["base_url"], target["model"],
                                 api_key=target["api_key"] or "none")
            self._clients[key] = client
        return client

    def _gc_turns(self):
        """Finished turns stay replayable for a while - a browser that comes
        back a minute later should still see how it ended."""
        cutoff = time.time() - TURN_TTL
        for sid, turn in list(self._turns.items()):
            if turn.done and (turn.finished or 0) < cutoff:
                self._turns.pop(sid, None)

    def attach(self, body):
        """Watch a turn that is already running (or just finished)."""
        sid = str(body.get("session_id") or "")
        with self._lock:
            turn = self._turns.get(sid)
        if not turn:
            return Response.error("nothing is running in this conversation", 404)
        return Stream(turn.follow(int(body.get("from") or 0)))

    def _live(self):
        with self._lock:
            self._gc_turns()
            return {sid: turn.summary() for sid, turn in self._turns.items()}

    def _drop_session(self, sid):
        """Session state is not just the file: a turn may still be running and
        background jobs are real processes - both have to be stopped, not
        forgotten."""
        cancel = self._cancel.get(sid)
        if cancel:
            cancel.set()
        self._turns.pop(sid, None)
        ctx = self._contexts.pop(sid, None)
        if ctx:
            for job in list(ctx.jobs.values()):
                try:
                    if job.running:
                        job.proc.terminate()
                except Exception:                      # noqa: BLE001
                    pass
        self._cancel.pop(sid, None)
        self._always.pop(sid, None)
        self._save_locks.pop(sid, None)

    @staticmethod
    def _searchable(session):
        """Everything in a conversation worth matching on, as one lowercase
        blob. Pasted images are data URLs megabytes long and match nothing, so
        they are left out."""
        parts = [session.get("title") or ""]
        for m in session.get("messages") or []:
            content = m.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                parts.extend(p.get("text", "") for p in content
                             if isinstance(p, dict) and p.get("type") == "text")
            for call in m.get("tool_calls") or []:
                parts.append(str((call.get("function") or {}).get("name", "")))
        return "\n".join(parts)[:SEARCH_BLOB_CHARS].lower()

    def _index(self, path: Path, session: dict):
        """Cache the blob against the file's mtime: searching should not re-read
        and re-parse every conversation on every keystroke."""
        try:
            stamp = path.stat().st_mtime
        except OSError:
            return ""
        cached = self._search_cache.get(path.name)
        if cached and cached[0] == stamp:
            return cached[1]
        blob = self._searchable(session)
        self._search_cache[path.name] = (stamp, blob)
        return blob

    @staticmethod
    def _snippet(blob, query, width=70):
        at = blob.find(query)
        if at < 0:
            return ""
        start = max(0, at - width // 2)
        text = blob[start:at + len(query) + width // 2].replace("\n", " ").strip()
        return ("..." if start else "") + text + "..."

    def _next_pin_order(self):
        """One past the highest order among currently-pinned chats, so a
        freshly pinned one lands at the bottom of that list, not the top."""
        best = -1
        for p in self.sessions_dir.glob("*.json"):
            try:
                s = json.loads(p.read_text(encoding="utf-8"))
            except ValueError:
                continue
            if s.get("pinned"):
                best = max(best, s.get("order", 0))
        return best + 1

    def _list_sessions(self, query=""):
        query = (query or "").strip().lower()
        rows = []
        for p in self.sessions_dir.glob("*.json"):
            try:
                s = json.loads(p.read_text(encoding="utf-8"))
            except ValueError:
                continue
            snippet = ""
            if query:
                blob = self._index(p, s)
                if query not in blob:
                    continue
                if query not in (s.get("title") or "").lower():
                    snippet = self._snippet(blob, query)
            turn = self._turns.get(s.get("id"))
            rows.append({"id": s.get("id"), "title": s.get("title") or "New chat",
                         "mode": s.get("mode", "chat"),
                         "updated": s.get("updated", 0),
                         "pinned": bool(s.get("pinned")),
                         "order": s.get("order", 0),
                         "snippet": snippet,
                         "running": bool(turn and not turn.done),
                         "messages": sum(1 for m in s.get("messages", [])
                                         if m.get("role") == "user")})
        # pinned first, in the order the sidebar's drag-and-drop set (ties -
        # never-reordered pins - fall back to newest); everything else newest
        # first, exactly as before drag-and-drop existed.
        pinned = sorted((r for r in rows if r["pinned"]),
                        key=lambda r: (r["order"], -r["updated"]))
        rest = sorted((r for r in rows if not r["pinned"]), key=lambda r: -r["updated"])
        rows = pinned + rest
        return rows

    @staticmethod
    def _title_from(text):
        one = " ".join((text or "").split())
        return (one[:MAX_TITLE] + ("..." if len(one) > MAX_TITLE else "")) or "New chat"

    # ------------------------------------------------------------ config ----

    def _config(self):
        def names(mode):
            return [{"name": t.name, "risk": t.risk}
                    for t in webui_tools.tools_for(mode, self.cfg, self.registry)]
        return {
            "model": self.model_id,
            "context_length": self.context_length,
            # This server can turn thinking off exactly (the chat template emits
            # an empty <think></think>) but has no way to enforce a level once
            # generation starts, so it advertises none. A provider's levels come
            # from what its own /models said - see webui_providers.probe_efforts.
            "efforts": [],
            "vision": self.vision,
            "title": self.cfg.get("UI_TITLE") or "Simplex",
            "default_workspace": self.default_workspace,
            "max_steps": self.max_steps,
            "native_picker": self.picker == "native" and webui_picker.available(),
            "can_switch_model": bool(self.supervised and self.on_restart),
            "providers": [webui_providers.public(p)
                          for p in webui_providers.load(self.root)],
            "tools": {"chat": names("chat"), "agent": names("agent")},
            "defaults": {
                "temperature": float(self.cfg.get("TEMPERATURE") or 0.6),
                "top_p": float(self.cfg.get("TOP_P") or 0.95),
                "top_k": int(self.cfg.get("TOP_K") or 20),
                "max_tokens": int(self.cfg.get("MAX_TOKENS") or 4096),
            },
        }

    # ---------------------------------------------------- folder browsing ----

    def _browse(self, raw):
        p = Path(raw).expanduser() if raw else Path.home()
        try:
            p = p.resolve()
            if not p.is_dir():
                p = p.parent
            dirs = []
            for entry in sorted(p.iterdir(), key=lambda e: e.name.lower()):
                try:
                    if entry.is_dir() and not entry.name.startswith("."):
                        dirs.append({"name": entry.name, "path": str(entry)})
                except OSError:
                    continue
        except OSError as e:
            return Response.error(f"cannot list {raw}: {e}", 400)
        return Response.json({
            "path": str(p),
            "parent": str(p.parent) if p.parent != p else None,
            "dirs": dirs[:500],
        })

    # ------------------------------------------------------------- turns ----

    @staticmethod
    def _for_model(messages):
        """History as the model should see it: no stored reasoning replayed."""
        out = []
        for m in messages:
            if m.get("role") == "assistant" and not m.get("content") \
                    and not m.get("tool_calls"):
                continue
            out.append({k: v for k, v in m.items()
                        if k not in ("reasoning_content", "ui")})
        return out

    def _wait(self, call_id, timeout, cancelled):
        """Wait for the browser, but wake up if the user pressed Stop - a turn
        blocked on an approval nobody will answer used to sit for 15 minutes
        holding the session busy."""
        with self._lock:
            pend = self._pending.get(call_id) or _Pending()
            self._pending[call_id] = pend
        deadline = time.time() + timeout
        while time.time() < deadline:
            if pend.event.wait(timeout=0.25):
                break
            if cancelled and cancelled():
                break
        with self._lock:
            self._pending.pop(call_id, None)
        return pend

    def _asker(self, cancelled=None):
        """ask_user: same hand-shake as an approval, but the browser sends text
        back instead of a verdict."""
        def ask(call_id, payload):
            pend = self._wait(call_id,
                              float(self.cfg.get("QUESTION_TIMEOUT") or 1800),
                              cancelled)
            return pend.answer if pend.event.is_set() else ""
        return ask

    def _forget(self, key):
        with self._lock:
            self._pending.pop(key, None)

    def _register_approval(self, key):
        """Called while the 'approval' event is still on its way out, so the
        browser can never answer a request we have not recorded yet."""
        with self._lock:
            self._pending[key] = _Pending()

    def _approver(self, sid, cancelled=None):
        allowed = self._always.setdefault(sid, set())

        def approve(call_id, tool, args):
            pend = self._wait(call_id,
                              # a turn now outlives the window it was started
                              # from, so an approval should wait long enough to
                              # still be there when someone comes back
                              float(self.cfg.get("APPROVAL_TIMEOUT") or 1800),
                              cancelled)
            decision = pend.decision if pend.event.is_set() else "deny"
            if decision == "always":
                allowed.add(tool.name)
            return decision in ("allow", "always")

        # "always allow" is answered before the card is ever drawn (run_turn
        # asks this first), so an auto-approved call shows no prompt at all
        return approve, allowed.__contains__

    def chat(self, body):
        sid = body.get("session_id") or uuid.uuid4().hex[:16]
        mode = "agent" if body.get("mode") == "agent" else "chat"
        settings = body.get("settings") or {}
        user_content = body.get("content")
        if not user_content:
            return Response.error("`content` is required")

        session = self._load(sid) or {
            "id": sid, "title": None, "mode": mode,
            "workspace": body.get("workspace") or self.default_workspace,
            "created": time.time(), "messages": [],
        }
        session["mode"] = mode
        if body.get("workspace") and body["workspace"] != session.get("workspace"):
            # "always allow run_command" was consent for the old folder
            session["workspace"] = body["workspace"]
            self._always.pop(sid, None)

        try:
            target = webui_providers.resolve(
                self.root, str(body.get("provider") or session.get("provider") or ""),
                str(body.get("model") or session.get("model") or ""), self._local())
        except webui_providers.ProviderError as e:
            return Response.error(str(e), 400)
        session["provider"] = target["id"]
        session["model"] = target["model"]
        client = self._client_for(target)

        ctx = self._contexts.get(sid)
        if ctx is None:
            ctx = webui_tools.ToolContext(cfg=self.cfg)
            ctx.plan = list(session.get("plan") or [])
            self._contexts[sid] = ctx

        # workspace only matters in agent mode, and must exist before we start
        workspace = None
        if mode == "agent":
            root = Path(session.get("workspace") or self.default_workspace).expanduser()
            try:
                root.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                return Response.error(f"cannot use workspace {root}: {e}")
            workspace = webui_tools.Workspace(root)
        ctx.workspace = workspace

        text_only = user_content if isinstance(user_content, str) else " ".join(
            part.get("text", "") for part in user_content
            if isinstance(part, dict) and part.get("type") == "text")
        session["messages"].append({"role": "user", "content": user_content})
        if not session.get("title"):
            session["title"] = self._title_from(text_only)
        self._save(session)

        tool_list = webui_tools.tools_for(mode, self.cfg, self.registry)
        sys_msg = {"role": "system",
                   "content": settings.get("system") or system_prompt(
                       mode, tool_list, session.get("workspace"))}
        convo = [sys_msg] + self._for_model(session["messages"])

        sampling = {
            "temperature": float(settings.get("temperature", 0.6)),
            "top_p": float(settings.get("top_p", 0.95)),
            "top_k": int(settings.get("top_k", 20)),
            "max_tokens": int(settings.get("max_tokens", 4096)),
        }
        # How hard the model should think. "off" is exact - the chat template
        # emits an empty <think></think> and there is nowhere to reason. The
        # effort levels are guidance: this engine cannot cut thinking short
        # mid-generation, so they are sent in both spellings a server might
        # understand (chat_template_kwargs is what vLLM/SGLang and this kit's
        # own server read; reasoning_effort is what hosted APIs read) and the
        # model obliges or does not.
        thinking = str(settings.get("thinking", "")).strip().lower()
        if thinking == "off":
            sampling["chat_template_kwargs"] = {"enable_thinking": False}
            sampling["reasoning_effort"] = "none"
        elif thinking in ("low", "medium", "high"):
            sampling["reasoning_effort"] = thinking
        with self._lock:
            self._gc_turns()
            live = self._turns.get(sid)
            if live and not live.done:
                return Response.error(
                    "this conversation is still working - open it to watch, or "
                    "press Stop first", 409)
        cancel = self._cancel.setdefault(sid, threading.Event())
        cancel.clear()
        approve, pre_approved = self._approver(sid, cancel.is_set)
        ask = self._asker(cancel.is_set)

        def events():
            yield {"type": "session", "id": sid, "title": session["title"],
                   "mode": mode, "workspace": session.get("workspace"),
                   "provider": target["id"], "provider_name": target["name"],
                   "model": target["model"]}
            final, failure, plan = None, None, None
            sent = len(convo)          # anything past this is new this turn
            with self._lock:
                self._active += 1
                self._counters["busy"] = True
            try:
                for ev in run_turn(client, convo, tool_list, ctx,
                                   mode=mode, sampling=sampling,
                                   max_steps=self.max_steps, approve=approve,
                                   pre_approved=pre_approved,
                                   ask=ask, cancelled=cancel.is_set):
                    if ev["type"] in ("approval", "question"):
                        self._register_approval(ev["id"])
                    if ev["type"] == "tool_result":
                        # whatever happened - answered, auto-approved, or the
                        # tool refused the arguments - nothing is waiting now
                        self._forget(ev["id"])
                    if ev["type"] == "plan":
                        plan = ev["items"]
                    if ev["type"] == "error":
                        failure = ev.get("message")
                    if ev["type"] == "done":
                        final = ev
                        continue
                    if self.fake_health and ev["type"] in ("content", "reasoning"):
                        self._counters["completion"] += 1      # dev-server only
                    yield ev
            except Exception as e:                     # noqa: BLE001
                failure = f"{type(e).__name__}: {e}"
                yield {"type": "error", "message": failure}
            finally:
                with self._lock:
                    self._active = max(0, self._active - 1)
                    self._counters["busy"] = self._active > 0
            fresh = (final or {}).get("messages") or convo
            self._append_turn(sid, fresh[sent:], plan,
                              {"provider": target["id"], "model": target["model"],
                               "mode": mode, "workspace": session.get("workspace"),
                               "title": session["title"]})
            yield {"type": "done",
                   "usage": (final or {}).get("usage") or {},
                   "tok_s": (final or {}).get("tok_s"),
                   "seconds": (final or {}).get("seconds"),
                   "failed": bool(failure), "error": failure,
                   "cancelled": bool(cancel.is_set() and not failure),
                   "session": {"id": sid, "title": session["title"]}}
        # The turn is owned by the session from here on. The response below is
        # just the first viewer of it.
        turn = Turn(sid, cancel)
        with self._lock:
            self._turns[sid] = turn

        def pump():
            try:
                for event in events():
                    turn.append(event)
            except Exception as e:                     # noqa: BLE001
                turn.append({"type": "error",
                             "message": f"{type(e).__name__}: {e}"})
            finally:
                turn.finish()

        threading.Thread(target=pump, daemon=True, name=f"turn-{sid}").start()
        return Stream(turn.follow(0))

    # ----------------------------------------------------------- routing ----

    def handle(self, method, path, query, body_bytes, remote=None, headers=None):
        ok, why = self._origin_ok(method, headers)
        if not ok:
            return Response.error(why, 403)
        if not self.allow_lan and not self._is_local(remote):
            return Response.error(
                "The chat UI answers only the computer it runs on. Agent mode "
                "can write files and run commands here and there is no login, "
                "so other devices are refused. Set UI_LAN=1 in .env if you "
                "trust everyone on this network. The /v1 API is unaffected.",
                403)
        try:
            body = json.loads(body_bytes) if body_bytes else {}
        except ValueError:
            return Response.error("invalid JSON")

        if method == "GET":
            if path in ("/", "/index.html", "/ui", "/ui/"):
                return self._static("index.html")
            if path.startswith("/ui/static/"):
                return self._static(path[len("/ui/static/"):])
            # Installable-app files. These live at the root on purpose: a
            # service worker may only control the paths below its own URL, and
            # the manifest's icons are referenced from the root scope too.
            if path in PWA_FILES:
                return self._static(path[1:])
            if path == "/health" and self.fake_health:
                return Response.json({
                    "ok": True, "backend": "mock",
                    "busy": self._counters["busy"],
                    "prompt_tokens_total": self._counters["prompt"],
                    "completion_tokens_total": self._counters["completion"],
                    "context_length": self.context_length, "vision": self.vision})
            if path == "/ui/config":
                return Response.json(self._config())
            if path == "/ui/sessions":
                return Response.json({
                    "sessions": self._list_sessions(query.get("q", "")),
                    "live": self._live()})
            if path.startswith("/ui/sessions/"):
                got = self._load(path.rsplit("/", 1)[-1])
                return Response.json(got) if got else Response.error("no such session", 404)
            if path == "/ui/models":
                remote = []
                for provider in webui_providers.load(self.root):
                    for name in (provider.get("models")
                                 or [provider.get("default_model")]):
                        if not name:
                            continue
                        remote.append({"provider": provider["id"],
                                       "provider_name": provider["name"],
                                       "model": name,
                                       "base_url": provider["base_url"]})
                return Response.json({
                    "models": webui_models.installed(
                        self.root, self.cfg.get("MODEL_DIR", "")),
                    "remote": remote,
                    "current": self.model_id,
                    "can_switch": bool(self.supervised and self.on_restart),
                    "note": ("" if self.supervised else
                             "start the server with start.bat / start.sh to "
                             "switch models from here"),
                })
            if path == "/ui/providers":
                return Response.json({
                    "providers": [webui_providers.public(p)
                                  for p in webui_providers.load(self.root)],
                    "local": {"id": webui_providers.LOCAL_ID,
                              "name": "This computer",
                              "model": self.model_id,
                              "base_url": self.model_base},
                })
            if path == "/ui/browse":
                return self._browse(query.get("path"))
            return Response(404, "text/plain", b"not found")

        if method == "DELETE" and path.startswith("/ui/providers/"):
            webui_providers.delete(self.root, path.rsplit("/", 1)[-1])
            return Response.json({"ok": True})

        if method == "DELETE" and path.startswith("/ui/sessions/"):
            sid = path.rsplit("/", 1)[-1]
            p = self._path(sid)
            if p and p.is_file():
                p.unlink()
            self._drop_session(sid)
            return Response.json({"ok": True})

        if method == "POST":
            if path == "/ui/chat":
                return self.chat(body)
            if path == "/ui/approve":
                key = body.get("id") or body.get("key")
                decision = body.get("decision", "deny")
                with self._lock:
                    pend = self._pending.get(key)
                if not pend:
                    return Response.error("no such approval request", 404)
                pend.decision = decision
                pend.event.set()
                return Response.json({"ok": True})
            if path == "/ui/pick_folder":
                if not self._is_local(remote):
                    return Response.error(
                        "the system folder dialog would open on the computer "
                        "running the server, not on this device", 409)
                if self.picker != "native":
                    return Response.error("FOLDER_PICKER is not 'native'", 409)
                if not webui_picker.available():
                    return Response.error(
                        "this machine has no system folder dialog", 503)
                try:
                    chosen = webui_picker.pick(body.get("start")
                                               or self.default_workspace)
                except webui_picker.PickerUnavailable as e:
                    return Response.error(str(e), 503)
                return Response.json({"path": chosen} if chosen
                                     else {"cancelled": True})
            if path == "/ui/switch_model":
                if not self._is_local(remote):
                    return Response.error("only this computer may change the "
                                          "loaded model", 409)
                if not (self.supervised and self.on_restart):
                    return Response.error(
                        "this server was not started by start.bat / start.sh, "
                        "so it cannot restart itself", 409)
                if self._counters["busy"]:
                    return Response.error(
                        "a reply is still being generated - stop it first", 409)
                try:
                    updates = webui_models.apply(self.root,
                                                 str(body.get("dir") or ""), self.cfg)
                except webui_models.Incomplete as e:
                    return Response.error(str(e), 400)
                except (ValueError, OSError) as e:
                    return Response.error(str(e), 400)
                self.on_restart()          # the process exits; the launcher relaunches
                return Response.json({"ok": True, "restarting": True,
                                      "settings": updates})
            if path == "/ui/providers":
                # keys are only ever written, never read back out
                try:
                    saved = webui_providers.upsert(self.root, body)
                except webui_providers.ProviderError as e:
                    return Response.error(str(e), 400)
                return Response.json({"provider": webui_providers.public(saved)})
            if path == "/ui/providers/test":
                try:
                    return Response.json(webui_providers.test(self.root, body))
                except webui_providers.ProviderError as e:
                    return Response.error(str(e), 400)
            if path == "/ui/attach":
                return self.attach(body)
            if path == "/ui/rewind":
                # Drop the last exchange so the browser can resend it (retry)
                # or put it back in the composer (edit). The plan and the
                # read-before-edit record stay: they describe the workspace,
                # not the transcript.
                session = self._load(body.get("session_id"))
                if not session:
                    return Response.error("no such session", 404)
                messages = session.get("messages") or []
                cut = max((i for i, m in enumerate(messages)
                           if m.get("role") == "user"), default=None)
                if cut is None:
                    return Response.error("nothing to rewind", 400)
                dropped = messages[cut]
                session["messages"] = messages[:cut]
                self._save(session)
                return Response.json({"ok": True, "content": dropped.get("content"),
                                      "kept": len(session["messages"])})
            if path == "/ui/answer":
                with self._lock:
                    pend = self._pending.get(body.get("id"))
                if not pend:
                    return Response.error("no such question", 404)
                pend.answer = str(body.get("text") or "")[:2000]
                pend.decision = "answer"
                pend.event.set()
                return Response.json({"ok": True})
            if path == "/ui/cancel":
                # Explicit Stop is the only thing that ends a turn early;
                # closing the window is not.
                ev = self._cancel.get(body.get("session_id"))
                if ev:
                    ev.set()
                return Response.json({"ok": True})
            if path == "/ui/sessions/reorder":
                # The sidebar drags pinned chats into a new sequence and posts
                # the whole resulting id list; we just number them 0..n and
                # save. Simpler and more robust than fractional insert-between
                # math, and one bad id in a stale payload cannot corrupt the
                # rest - it is just skipped.
                ids = body.get("ids")
                if not isinstance(ids, list) or not ids:
                    return Response.error("ids must be a non-empty list", 400)
                changed = []
                for i, sid in enumerate(ids[:500]):
                    session = self._load(str(sid))
                    if not session or not session.get("pinned"):
                        continue        # stale client state - nothing to move
                    session["order"] = i
                    self._save(session, touch=False)
                    changed.append(session["id"])
                return Response.json({"ok": True, "reordered": changed})
            if path.startswith("/ui/sessions/"):
                session = self._load(path.rsplit("/", 1)[-1])
                if not session:
                    return Response.error("no such session", 404)
                if body.get("title"):
                    session["title"] = str(body["title"])[:MAX_TITLE]
                if "pinned" in body:
                    was_pinned = bool(session.get("pinned"))
                    session["pinned"] = bool(body["pinned"])
                    if session["pinned"] and not was_pinned:
                        # Every unpinned -> pinned transition joins the end of
                        # the pinned list, not the top - dragging is how you
                        # promote one, not pinning. A stale "order" from a
                        # previous pinning must not resurrect its old spot.
                        session["order"] = self._next_pin_order()
                self._save(session)
                return Response.json({"ok": True, "pinned": bool(session.get("pinned")),
                                      "title": session.get("title")})
        return Response(404, "text/plain", b"not found")
