#!/usr/bin/env python3
"""DeepSeek Harness (`dsh`) - this kit's chat and agent front end.

The kit serves the model; the harness is what you talk to. `dsh` is DeepSeek's
open-source agent harness (MIT, https://github.com/deepseek-ai/deepseek-harness),
a Node application published to npm, so nothing of it is vendored here: the
launcher runs it with `npx` and it caches itself after the first run.

What this module owns is the join between the two:

  describe_model()  asks the running server what it is - id, context window,
                    modalities, and which reasoning levels its chat template
                    actually acts on - over the same /v1 any client would use.
  write_settings()  turns that answer into `.dsh/settings.yaml`, the document
                    dsh reads for provider routes, so the model is configured
                    before the browser opens and nobody has to type a base URL
                    into a form.
  start()           runs `npx @deepseek-ai/dsh@<version> web`, with DSH_HOME
                    pointed inside this folder so a harness installed here
                    cannot disturb one you already use elsewhere.
  watch_output()    reads back the one line dsh prints, because that line is
                    the only way in. dsh authenticates the browser with a token
                    minted fresh every launch: it prints
                    `dsh web: http://127.0.0.1:3080/?token=...`, and opening
                    that once sets a cookie good for thirty days and redirects
                    to a clean `/`. Open the address without the token and dsh
                    answers "authentication required; reopen the URL printed by
                    dsh web" - which is what happens to anyone who composes the
                    URL from the port instead of reading it.

The launcher owns two keys in that file - the provider route and the default
model - and nothing else in it. A copy of what it last wrote is kept beside it,
and generation stops as soon as those keys stop matching it: a hand-tuned route
survives every restart, and deleting the file is the only way back to a
generated one. Every other key is read past and written back untouched, which
is what lets dsh keep state of its own in the same document - it saves the
notices it has shown you there - without that freezing the join.
"""
from __future__ import annotations

import json
import os
import queue
import signal
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

# The version the kit is known to work with. `latest` follows the newest
# release instead, which is the right setting while dsh is in developer
# preview and moving fast - and the wrong one when a run has to reproduce.
DEFAULT_VERSION = "0.1.3-alpha.2"
DEFAULT_PORT = 3080
PACKAGE = "@deepseek-ai/dsh"

# The provider route dsh serves this kit's model on. The id is permanent as far
# as dsh is concerned - sessions and defaults reference it - so it is a plain
# name here rather than anything derived from the model of the day.
PROVIDER_ID = "qwen-local"
KEY_ENV = "QWEN_LOCAL_API_KEY"

GENERATED_MARKER = ".settings.generated"
# The top-level keys the launcher writes, in the order it writes them: the
# route to this kit's server, and what the harness opens on. Everything else in
# settings.yaml belongs to whoever put it there and is spliced back around
# these on every rewrite.
OWNED_KEYS = ("llm-pi-ai", "agent-default-model")
# dsh takes a key with or without its package prefix, and a document it has
# round-tripped can come back carrying the long form. Both name one key here.
KEY_PREFIX = "@deepseek-ai/dsh-"
# Where the launcher leaves the URL dsh printed, so the model server's landing
# page can hand a browser over to an authenticated address rather than to a 401.
# It lives inside DSH_HOME, beside the credential store it is no more secret
# than, and it is rewritten on every launch because the token is.
URL_FILE = "url"
# `dsh web: http://127.0.0.1:3080/?token=<base64url>`  (a LAN address may follow)
URL_RE = re.compile(r"https?://[^\s()<>\"']+[?&]token=[A-Za-z0-9_-]+")


# --------------------------------------------------------------- paths ------
def home(root: Path) -> Path:
    """DSH_HOME for this kit: settings, credentials and profiles live here."""
    return Path(root) / ".dsh"


def settings_path(root: Path) -> Path:
    return home(root) / "settings.yaml"


def url(port: int | str) -> str:
    """The address, without the token that gets a browser past the front door."""
    return f"http://127.0.0.1:{port}/"


def url_path(root: Path) -> Path:
    return home(root) / URL_FILE


def read_url(root: Path) -> str | None:
    """The authenticated URL of the harness this kit last started, or None."""
    try:
        text = url_path(root).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return text or None


def write_url(root: Path, authenticated: str) -> None:
    path = url_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(authenticated + "\n", encoding="utf-8")
    try:                                   # best effort; Windows ignores it
        os.chmod(path, 0o600)
    except OSError:
        pass


def clear_url(root: Path) -> None:
    """Forget it on the way out. The token dies with the process that minted
    it, and a stale one sends a browser to the same 401 as no token at all."""
    try:
        url_path(root).unlink()
    except OSError:
        pass


def token_accepted(authenticated: str, timeout: float = 5.0) -> bool:
    """Whether that URL still gets past dsh's front door.

    A launch token dies with the process that minted it. A harness killed
    without clearing the file leaves a URL behind that looks perfectly fine and
    answers 401 - the same error as having no token at all, which is the whole
    confusion this exists to prevent. The token is not single-use: dsh compares
    it and mints a cookie, so asking costs nothing."""
    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):     # noqa: D102
            return None

    opener = urllib.request.build_opener(_NoRedirect)
    try:
        with opener.open(authenticated, timeout=timeout) as r:
            return r.status < 400
    except urllib.error.HTTPError as e:
        return e.code in (301, 302, 303, 307, 308)       # the cookie redirect
    except (urllib.error.URLError, OSError):
        return False


def watch_output(proc: subprocess.Popen, echo=None) -> "queue.Queue[str]":
    """Read dsh's output in a thread; return a queue the URL arrives on.

    One thread owns the stream: it echoes every line (so npm's download and any
    failure land in the launcher's log like the server's own output does) and
    puts the authenticated URL on the queue the once, when dsh prints it."""
    found: "queue.Queue[str]" = queue.Queue(maxsize=1)

    def run() -> None:
        stream = proc.stdout
        if stream is None:
            return
        seen = False
        for raw in iter(stream.readline, b""):
            text = raw.decode("utf-8", "replace")
            if echo is not None:
                try:
                    echo(text)
                except Exception:          # noqa: BLE001 - never kill the pump
                    pass
            if not seen:
                match = URL_RE.search(text)
                if match:
                    seen = True
                    try:
                        found.put_nowait(match.group(0))
                    except queue.Full:     # noqa: WPS329 - cannot happen; harmless
                        pass

    threading.Thread(target=run, daemon=True).start()
    return found


# ---------------------------------------------------------------- node ------
def node_missing() -> str | None:
    """None when Node and npx are usable, otherwise what to tell the user."""
    npx = shutil.which("npx")
    node = shutil.which("node")
    if npx and node:
        return None
    return (
        "The DeepSeek Harness is a Node application and Node is not installed.\n"
        "  1. Open https://nodejs.org/ and install the LTS build (22 or newer)\n"
        "  2. Start this again - nothing else has to be set up by hand\n"
        "  The model server itself does not need Node: UI=no in .env skips this."
    )


# ------------------------------------------------------- ask the server -----
def describe_model(base_url: str, timeout: float = 10.0) -> dict:
    """The server's own /v1/models row: id, context, modalities, efforts.

    Everything the harness needs to know about the model is already published
    there - the same answer any other OpenAI client would get - so none of it
    is read out of .env, where it can disagree with what actually loaded."""
    req = urllib.request.Request(base_url.rstrip("/") + "/models",
                                 headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8"))
    rows = data.get("data") or []
    if not rows:
        raise ValueError(f"{base_url}/models listed no models")
    return rows[0]


def _levels(row: dict) -> list[str]:
    """Reasoning levels the template acts on, in escalation order.

    dsh names the same seven levels pi-ai does, and this server reports the
    subset its chat template both accepts and renders differently - so the two
    vocabularies meet without a translation table."""
    order = ("minimal", "low", "medium", "high", "xhigh", "max")
    got = [str(x).strip().lower() for x in (row.get("supported_reasoning_efforts") or [])]
    return [lvl for lvl in order if lvl in got]


def _vision(row: dict) -> bool:
    arch = row.get("architecture") or {}
    return "image" in [str(m).lower() for m in (arch.get("input_modalities") or [])]


def _context(row: dict, fallback: int = 262144) -> int:
    for key in ("max_model_len", "context_length", "context_window"):
        try:
            n = int(row.get(key) or 0)
        except (TypeError, ValueError):
            continue
        if n > 0:
            return n
    return fallback


# ------------------------------------------------------------ settings ------
def settings_text(base_url: str, row: dict, max_tokens: int = 65536) -> str:
    """`.dsh/settings.yaml` for one route pointing at this kit's server.

    Written by hand rather than through a YAML library so the kit keeps its
    "standard library only" promise, and so the file that lands in front of a
    person carries the reasons for what is in it."""
    model_id = str(row.get("id") or "local-model")
    ctx = _context(row)
    levels = _levels(row)
    vision = _vision(row)

    lines = [
        "# Written by tools/dsh.py from what the model server reported on /v1/models.",
        "#",
        "# The launcher owns the keys below and rewrites them when the model changes.",
        "# Edit them and they are yours: it compares them with the copy it last wrote",
        f"# ({GENERATED_MARKER} beside this file) and stops generating as soon as the two",
        "# differ. Delete this file to get a fresh one at the next start.",
        "#",
        "# Anything else you add here - another provider, a tool policy, the state dsh",
        "# saves in this same file - is read past and written back untouched.",
        "#",
        "# Every field dsh accepts is listed in its configuration catalog:",
        "# https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/config-catalog.md",
        "",
        "llm-pi-ai:",
        "  providers:",
        f"    {PROVIDER_ID}:",
        "      displayName: Qwen3.8-27B (this PC)",
        "      api: openai-completions",
        f"      baseURL: {base_url.rstrip('/')}",
        "      # The server ignores the key, but a route without a credential",
        "      # reference fails every request with MISSING_CREDENTIAL. The",
        "      # launcher sets this variable before starting dsh.",
        f"      apiKeyEnv: {KEY_ENV}",
        "      compat:",
        "        # This server reads the system prompt from role \"system\" and caps",
        "        # output with max_tokens; a reasoning model would otherwise be sent",
        "        # role \"developer\" and max_completion_tokens, which it does not read.",
        "        supportsDeveloperRole: false",
        "        maxTokensField: max_tokens",
        "      models:",
        f"        - id: {model_id}",
        f"          name: {model_id}",
        f"          contextWindow: {ctx}",
        f"          maxTokens: {max_tokens}",
    ]
    if vision:
        lines.append("          # The checkpoint's vision tower is loaded, so images are")
        lines.append("          # accepted. Start the server with VISION=off and regenerate")
        lines.append("          # this file to take them away again.")
        lines.append("          input: [text, image]")
    else:
        lines.append("          # Text only: this server started without a vision tower.")
        lines.append("          input: [text]")
    if levels:
        lines.append("          reasoningEfforts:")
        lines.append("            # `off` carries a value on purpose: an empty one sends no")
        lines.append("            # reasoning field at all, and this template thinks unless")
        lines.append("            # it is told not to. Quoted because a YAML 1.1 reader -")
        lines.append("            # anything you might open this in - would take a bare off")
        lines.append("            # for the boolean false.")
        lines.append("            'off': 'off'")
        for lvl in levels:
            lines.append(f"            {lvl}: {lvl}")
    else:
        lines.append("          # The chat template ignores reasoning_effort, so offering a")
        lines.append("          # level that changes nothing would be worse than offering none.")
        lines.append("          reasoningEfforts: false")
    lines.append("")
    lines.append("agent-default-model:")
    lines.append("  # What a new chat opens on. dsh's own default is DeepSeek's hosted")
    lines.append("  # V4-Flash, and the choice is remembered per session, so without this")
    lines.append("  # the model this PC just loaded - the whole point of the kit - is one")
    lines.append("  # menu away on every new chat, and the wrong answer sticks to every")
    lines.append("  # session that was started before anyone noticed. This only moves the")
    lines.append("  # default: every other route stays in the picker.")
    lines.append(f"  provider: {PROVIDER_ID}")
    lines.append(f"  model: {model_id}")
    lines.append("")
    return "\n".join(lines)


def _canon(key: str | None) -> str | None:
    """One name for a key dsh accepts under two."""
    if key is None:
        return None
    k = key.strip().strip('"').strip("'")
    return k[len(KEY_PREFIX):] if k.startswith(KEY_PREFIX) else k


def _cut(text: str) -> list[tuple[str | None, str]]:
    """A settings document cut into top-level blocks, in order.

    Each entry is `(key, text)`. A block starts at the comments and blank lines
    that come before a top-level key - an explanation belongs to the key it
    explains, so replacing a block replaces its reasons with it - and runs to
    the line before the next one; anything trailing with no key of its own
    comes back under `None`.

    Cut as text rather than parsed as YAML because the generated file carries
    the reasons for what is in it as comments, and no writer this kit is
    allowed to depend on keeps them.
    """
    out: list[tuple[str | None, str]] = []
    key: str | None = None
    body: list[str] = []
    held: list[str] = []                  # comments and blanks awaiting a key

    for line in text.splitlines(keepends=True):
        bare = line.strip()
        top = (line[:1] not in ("", " ", "\t", "#", "\n", "\r")
               and ":" in line.split("#")[0])
        if top:
            if key is not None or body:
                out.append((key, "".join(body)))
            key = _canon(line.split(":", 1)[0])
            body = held + [line]
            held = []
        elif key is None:
            held.append(line)             # the file's own header
        elif bare == "" or bare.startswith("#"):
            held.append(line)             # might belong to the next key
        else:
            body.extend(held)             # it did not - it was ours
            held = []
            body.append(line)
    if key is not None or body:
        out.append((key, "".join(body)))
    if held:
        out.append((None, "".join(held)))
    return out


def _splice(have: str, want: str) -> str:
    """`want`'s blocks put back into `have`, every other block left alone."""
    fresh = {k: t for k, t in _cut(want) if k in OWNED_KEYS}
    out: list[str] = []
    seen: set[str] = set()
    for k, t in _cut(have):
        if k in fresh and k not in seen:
            out.append(fresh[k])
            seen.add(k)
        elif k in fresh:
            continue                      # a duplicate of a key we own
        else:
            out.append(t)
    for k in OWNED_KEYS:                  # generated, but not in the file yet
        if k in fresh and k not in seen:
            if out and not out[-1].endswith("\n"):
                out.append("\n")
            out.append(fresh[k])
    return "".join(out)


def _owned(text: str) -> dict | None:
    """The launcher-owned keys as data, or None if the file will not parse.

    Reading them as data is what lets dsh reformat the document - it
    round-trips settings.yaml when it saves state of its own, respacing flow
    sequences on the way through - without the launcher mistaking its own
    words, said differently, for somebody's edit. PyYAML is optional here as
    everywhere in the kit; without it the same question is asked of the text.
    """
    try:
        import yaml                                    # noqa: WPS433
    except ImportError:
        return None
    try:
        doc = yaml.safe_load(text)
    except Exception:      # noqa: BLE001 - a file we cannot read is not ours to judge
        return None
    if not isinstance(doc, dict):
        return None
    got = {_canon(k): v for k, v in doc.items()}
    return {k: got[k] for k in OWNED_KEYS if k in got}


def _owned_text(text: str) -> dict:
    """The same question without PyYAML: our blocks as they are written."""
    return {k: t for k, t in _cut(text) if k in OWNED_KEYS}


def _mine(*texts: str) -> tuple:
    """Each document's owned keys, compared like with like."""
    read = [_owned(t) for t in texts]
    if any(r is None for r in read):
        return tuple(_owned_text(t) for t in texts)
    return tuple(read)


def write_settings(root: Path, base_url: str, row: dict,
                   max_tokens: int = 65536) -> tuple[Path, str]:
    """Generate `.dsh/settings.yaml`. Returns (path, outcome).

    Outcome is "created", "updated", "unchanged" or "kept-yours".

    Only the keys this module writes are compared, and only they are replaced.
    A second provider, a tool policy, the notices dsh records in the same file
    are written back where they were - so dsh saving its own state no longer
    counts as the edit that stops generation, which is what used to freeze the
    route on the first launch anyone dismissed a notice on."""
    root = Path(root)
    h = home(root)
    h.mkdir(parents=True, exist_ok=True)
    path = settings_path(root)
    stamp = h / GENERATED_MARKER
    want = settings_text(base_url, row, max_tokens)

    if not path.is_file():
        path.write_text(want, encoding="utf-8")
        stamp.write_text(want, encoding="utf-8")
        return path, "created"

    have = path.read_text(encoding="utf-8")
    if have == want:
        stamp.write_text(want, encoding="utf-8")
        return path, "unchanged"
    if not stamp.is_file():
        return path, "kept-yours"         # a file the launcher never wrote
    last = stamp.read_text(encoding="utf-8")

    now, then, fresh = _mine(have, last, want)
    if now != then:
        return path, "kept-yours"
    if now == fresh:
        # Ours, and already saying what this run would say. Rewriting it would
        # only undo dsh's spacing for the pleasure of doing it again next
        # launch.
        stamp.write_text(want, encoding="utf-8")
        return path, "unchanged"

    path.write_text(_splice(have, want), encoding="utf-8")
    stamp.write_text(want, encoding="utf-8")
    return path, "updated"


# --------------------------------------------------------------- launch -----
def command(version: str, port: int) -> list[str]:
    npx = shutil.which("npx") or "npx"
    spec = PACKAGE if version in ("", "latest") else f"{PACKAGE}@{version}"
    return [npx, "-y", spec, "web", "--port", str(port), "--no-open"]


def start(root: Path, port: int = DEFAULT_PORT, version: str = DEFAULT_VERSION,
          api_key: str = "local", extra_env: dict | None = None) -> subprocess.Popen:
    """Run `dsh web`, its output on this process's pipes for the log to catch."""
    root = Path(root)
    env = dict(os.environ)
    env["DSH_HOME"] = str(home(root))
    env[KEY_ENV] = api_key or "local"
    env.setdefault("NO_UPDATE_NOTIFIER", "1")
    if extra_env:
        env.update(extra_env)
    home(root).mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        command(version, port), cwd=str(root), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0,
        shell=False)


def listening(port: int) -> bool:
    """True when something already holds the port - a harness from an earlier
    run, most likely. A model switch re-execs the launcher while the harness
    it started is still up, and starting a second one would only lose a race
    for the port."""
    try:
        with socket.create_connection(("127.0.0.1", int(port)), timeout=1):
            return True
    except OSError:
        return False


def wait_for_server(base_url: str, seconds: float) -> bool:
    """Poll the model server's /health until it answers. A first load compiles
    kernels, so this is minutes, not seconds."""
    health = base_url.rstrip("/").rsplit("/v1", 1)[0] + "/health"
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health, timeout=2) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(2)
    return False


def await_url(proc: subprocess.Popen, found: "queue.Queue[str]",
              timeout: float = 420.0) -> str | None:
    """The URL dsh printed, or None - but never waiting past the death of the
    process that would have printed it.

    A first launch downloads dsh from npm, so the ceiling is minutes. A launch
    that cannot bind the port, or that npm cannot resolve at all, is over in
    seconds - and waiting out the ceiling for a process that has already exited
    is how a clear error turns into a hang with nothing on screen."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            return found.get(timeout=0.5)
        except queue.Empty:
            pass
        if proc.poll() is not None:
            # It is gone. The line may still be in flight between the pipe and
            # the reader thread, so give that a moment before giving up.
            try:
                return found.get(timeout=2.0)
            except queue.Empty:
                return None
    return None


def wait_ready(proc: subprocess.Popen | None, port: int,
               timeout: float = 240.0) -> bool:
    """True once something answers on `port`. A first run downloads dsh from
    npm, which is why the wait is minutes rather than seconds."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            return False
        try:
            with socket.create_connection(("127.0.0.1", int(port)), timeout=1):
                return True
        except OSError:
            time.sleep(0.5)
    return False


# ------------------------------------------------------------------ cli -----
def main() -> int:
    """`python tools/dsh.py [--port N] [--base URL]` - configure and run it.

    The same thing windows\\start.bat / linux/start.sh do after the model is loaded, for when
    the server is already running and you only want the harness."""
    import argparse
    ap = argparse.ArgumentParser(description="configure and run the DeepSeek Harness")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--base", default=None, help="the model server's OpenAI base URL")
    ap.add_argument("--version", default=None)
    ap.add_argument("--open", action="store_true")
    ap.add_argument("--settings-only", action="store_true",
                    help="write .dsh/settings.yaml and stop")
    ap.add_argument("--wait", type=float, default=0.0, metavar="SECONDS",
                    help="wait this long for the model server to answer /health "
                         "before asking it what it is")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    cfg = {}
    envfile = root / ".env"
    if envfile.is_file():
        for raw in envfile.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                cfg[k.strip()] = v.split("#")[0].strip().strip('"').strip("'")

    port = args.port or int(cfg.get("DSH_PORT") or DEFAULT_PORT)
    version = args.version or cfg.get("DSH_VERSION") or DEFAULT_VERSION
    base = args.base or f"http://127.0.0.1:{cfg.get('PORT', '8888')}/v1"

    if args.wait > 0 and not wait_for_server(base, args.wait):
        print(f"  The model server at {base} did not answer within "
              f"{int(args.wait)}s.", file=sys.stderr)
        return 1
    try:
        row = describe_model(base)
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError) as e:
        print(f"  Cannot reach the model server at {base}: {e}", file=sys.stderr)
        print("  Start it first (windows\\start.bat / ./linux/start.sh), then run this again.",
              file=sys.stderr)
        return 1

    path, outcome = write_settings(root, base, row,
                                   int(cfg.get("MAX_TOKENS") or 65536))
    print(f"  settings  {path}  ({outcome})")
    if args.settings_only:
        return 0

    if listening(port):
        # Already up. Its settings were just rewritten and it re-reads them per
        # request, so the new model is reachable without touching the process.
        # Its token was minted by that process and can only be read back from
        # the file it was written to; without it a browser gets dsh's 401.
        opening = read_url(root)
        if opening is not None and token_accepted(opening):
            print(f"  harness   {opening}  (already running)")
            if args.open:
                import webbrowser
                webbrowser.open(opening)
            return 0
        clear_url(root)
        print(f"  Port {port} is held by something this kit cannot get into: "
              f"either a harness left over from a run that did not shut down, "
              f"or another program.", file=sys.stderr)
        print(f"  windows\\stop.bat / ./linux/stop.sh clears a leftover harness; DSH_PORT in "
              f".env moves this one out of the way.", file=sys.stderr)
        return 1

    missing = node_missing()
    if missing:
        print("\n  " + missing.replace("\n", "\n  "), file=sys.stderr)
        return 1

    proc = start(root, port, version, cfg.get("API_KEY") or "local")

    # linux/start.sh backgrounds this and stops it with a signal on the way out.
    # Python does not run `finally` for SIGTERM, so the harness would outlive
    # the launcher and the dead token would outlive the harness. Turn the
    # signal into an ordinary exit and both are cleaned up.
    def _bye(signum, _frame):
        if proc.poll() is None:
            try:
                proc.terminate()
            except OSError:
                pass
        clear_url(root)
        raise SystemExit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _bye)
        except (ValueError, OSError):     # not the main thread, or no such signal
            pass

    found = watch_output(proc, echo=lambda text: (sys.stdout.write(text),
                                                  sys.stdout.flush()))
    opening = await_url(proc, found)
    if opening is None:
        if proc.poll() is not None:
            print(f"\n  The harness stopped before it could serve (exit "
                  f"{proc.returncode}). The reason is in the lines above.",
                  file=sys.stderr)
            clear_url(root)
            return proc.returncode or 1
        print(f"  The harness did not print its address. It is meant to be at "
              f"{url(port)}, but dsh mints a token every launch and refuses a "
              f"browser that arrives without it.", file=sys.stderr)
    else:
        write_url(root, opening)
        print(f"  harness   {opening}")
    print("  Ctrl+C to stop", flush=True)
    if args.open and opening is not None and wait_ready(proc, port):
        # A server with no desktop session has nothing to open, and saying so
        # beats a stack trace or - worse - silence next to an address the
        # person then has to guess carries a token.
        opened = False
        try:
            import webbrowser
            opened = webbrowser.open(opening)
        except Exception as e:            # noqa: BLE001
            print(f"  Could not open a browser ({e}).", file=sys.stderr)
        if not opened:
            print("  No browser here - open the address above yourself. It "
                  "carries a one-time token; the plain address works after "
                  "that.", flush=True)
    try:
        return proc.wait()
    except (KeyboardInterrupt, SystemExit):
        return 0
    finally:
        if proc.poll() is None:
            proc.terminate()
        clear_url(root)


if __name__ == "__main__":
    raise SystemExit(main())
