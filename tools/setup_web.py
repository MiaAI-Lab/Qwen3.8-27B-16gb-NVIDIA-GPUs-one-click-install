"""The first run, in the browser.

Instead of a console that asks questions nobody reads, the launcher opens a
page: it shows what it found on the machine, offers the quants that fit the
card, then installs and downloads with a real progress bar and a live log.
The console keeps printing the same lines, so nothing is hidden - it is just
no longer the thing the user has to interact with.

The server is a standard-library HTTP server bound to the loopback address on
the same port the chat UI will use, so the address in the browser never
changes. It closes as soon as setup finishes, and the page polls until the
model server answers on the same port.
"""

from __future__ import annotations

import json
import os
import re
import socket
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
STATIC = TOOLS / "webui"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import setup_core as core                    # noqa: E402

LOCAL_HOSTS = {"localhost", "127.0.0.1", "[::1]", "::1"}
MAX_LOG = 4000
RETRY_GRACE = 600      # seconds the setup page stays up after a failure
MIME = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
        ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml",
        ".png": "image/png", ".webmanifest": "application/manifest+json",
        ".json": "application/json"}


class Setup:
    """State machine + append-only event log. One instance per launch."""

    PHASES = ("probe", "choose", "install", "done", "error", "cancelled")

    def __init__(self, cfg: dict, console=None, force: bool = False):
        self.cfg = dict(cfg)
        self._console = console
        self.force = bool(force)   # `start.bat profile`: show the menu even
                                   # when nothing actually needs installing
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self.events: list[dict] = []
        # Events are numbered from the beginning of time, but only the last
        # MAX_LOG are kept. `_dropped` is how many fell off the front, which is
        # what turns a client's absolute cursor back into a list position. The
        # chat UI's Turn never trims, so it could index the list directly; this
        # one does trim, and indexing it directly froze the page at 4000 events.
        self._dropped = 0
        self.log: list[dict] = []
        self.phase = "probe"
        self.error: dict | None = None
        self.probe_data: dict = {}
        self.menu: dict = {"options": []}
        self.steps = core.Steps(core.install_steps(), on_change=self._steps_changed)
        self.download: dict = {"state": "idle", "percent": 0.0}
        self._cancel = threading.Event()
        self._worker: threading.Thread | None = None
        self._dl = None
        self.finished = threading.Event()

    # ------------------------------------------------------------- events --
    @property
    def terminal(self) -> bool:
        return self.phase in self.TERMINAL

    def _emit(self, type_: str, /, **data) -> None:
        # positional-only: log entries carry their own `kind` field, which must
        # land in the event body rather than colliding with this parameter
        with self._cond:
            self.events.append({"i": self._dropped + len(self.events),
                                "type": type_, **data})
            excess = len(self.events) - MAX_LOG
            if excess > 0:
                del self.events[:excess]
                self._dropped += excess
            self._cond.notify_all()

    def _steps_changed(self, items: list[dict]) -> None:
        self._emit("steps", steps=items)

    def add_log(self, line: str, kind: str = "out") -> None:
        entry = {"line": line, "kind": kind, "t": round(time.time(), 3)}
        with self._lock:
            self.log.append(entry)
            del self.log[:-MAX_LOG]
        self._emit("log", **entry)
        if self._console:
            try:
                self._console(line, kind)
            except Exception:                # noqa: BLE001
                pass

    TERMINAL = ("done", "error", "cancelled")

    def set_phase(self, phase: str) -> None:
        with self._lock:
            self.phase = phase
        self._emit("phase", phase=phase, state=self.state())
        # Not a latch: "Try again" on the error screen puts setup back into
        # `install`, and the waiters - run() and every open SSE stream - have to
        # go back to waiting, or the page ends up talking to a server that has
        # already decided it is finished.
        if phase in self.TERMINAL:
            self.finished.set()
        else:
            self.finished.clear()
        with self._cond:
            self._cond.notify_all()

    def follow(self, start: int = 0):
        """Yield events from absolute number `start` on, blocking while setup is
        still moving. A cursor pointing at events that have already been trimmed
        resumes at the oldest one still held, rather than waiting for a number
        that will never come round again."""
        cursor = max(0, int(start or 0))
        while True:
            with self._cond:
                while cursor >= self._dropped + len(self.events) and not self.finished.is_set():
                    self._cond.wait(timeout=1.0)
                index = max(0, cursor - self._dropped)
                if index >= len(self.events):
                    return
                batch = self.events[index:]
                cursor = self._dropped + len(self.events)
            for e in batch:
                yield e

    # -------------------------------------------------------------- state --
    def state(self) -> dict:
        with self._lock:
            return {
                "phase": self.phase,
                "error": self.error,
                "probe": self.probe_data,
                "menu": self.menu,
                "cards": core.cards(),
                "steps": self.steps.as_list(),
                "download": dict(self.download),
                "events": self._dropped + len(self.events),
                "profile": self.cfg.get("PROFILE") or "",
                "model_id": self.cfg.get("MODEL_ID") or "",
                "cancelled": self._cancel.is_set(),
            }

    # --------------------------------------------------------------- flow --
    _KEEP = object()          # "leave the images answer as it is"

    def set_vision(self, want_vision) -> None:
        """Re-plan the menu for a new images answer, and nothing else.

        Deliberately does not call probe(): nothing about the machine changed.
        probe() shells out to nvidia-smi, spawns a Python to read wheel tags and
        checks the disk - seconds of work, which is what made these three
        buttons feel broken. None here means "decide for me", and it has to be
        able to *clear* a previous yes/no, which is why this is not just
        refresh(None): there, None means "keep whatever was chosen"."""
        self.want_vision = want_vision
        self.menu = core.options_for(self.cfg, 0.0, want_vision)
        self._emit("vision", vision=want_vision, state=self.state())

    def refresh(self, want_vision=_KEEP) -> None:
        self.probe_data = core.probe(self.cfg)
        if want_vision is not Setup._KEEP:
            self.want_vision = want_vision
        self.menu = core.options_for(self.cfg, 0.0, getattr(self, "want_vision", None))
        needed, reasons = core.needs_setup(self.cfg)
        self.probe_data["reasons"] = reasons
        if not needed and not self.force:
            self.set_phase("done")
            return
        self.set_phase("choose")

    def choose(self, quant_id: str, vram_gib: float = 0.0) -> None:
        want = getattr(self, "want_vision", None)
        self.cfg = core.apply_choice(self.cfg, quant_id, vram_gib, want)
        self.menu = core.options_for(self.cfg, vram_gib, want)
        self.probe_data = core.probe(self.cfg)
        self._emit("chose", profile=self.cfg.get("PROFILE"), state=self.state())

    TOKEN_RE = re.compile(r"^[A-Za-z0-9_\-.]{0,200}$")

    def set_token(self, token: str) -> None:
        import profiles
        token = (token or "").strip()
        # .strip() does not remove interior newlines, and .env is parsed line by
        # line - so "hf_x\nWHEEL_INDEX=http://attacker/" would add a second
        # setting that makes pip install wheels from wherever it says. Tokens
        # are opaque ASCII; anything else is not a token.
        if not self.TOKEN_RE.match(token):
            raise core.SetupError(
                "That does not look like a Hugging Face access token.",
                "Tokens are a single line of letters, digits, dashes and "
                "underscores - copy it again from huggingface.co/settings/tokens.")
        profiles.write_env(core.ENV_FILE, {"HF_TOKEN": token})
        self.cfg["HF_TOKEN"] = token
        if token:
            os.environ["HF_TOKEN"] = token

    def cancel(self) -> None:
        self._cancel.set()
        if self._dl is not None:
            self._dl.cancel()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def start(self) -> None:
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return
            self._cancel.clear()
            self._dl = None          # never let a Stop cancel the previous run
            self.error = None
            self.steps = core.Steps(core.install_steps(), on_change=self._steps_changed)
            self._worker = threading.Thread(target=self._run, daemon=True)
            w = self._worker
        self.set_phase("install")
        w.start()

    def _fail(self, err: core.SetupError | Exception) -> None:
        if isinstance(err, core.SetupError):
            self.error = err.as_dict()
        else:
            self.error = {"message": str(err) or err.__class__.__name__, "hint": ""}
        self.add_log(self.error["message"], "error")
        if self.error.get("hint"):
            self.add_log(self.error["hint"], "hint")
        self.set_phase("error")

    def _run(self) -> None:
        """Weights and Python packages come down at the same time: one is
        network-bound and the other is mostly CPU, so serialising them would
        add ten minutes for nothing."""
        weights_error: list[Exception] = []

        def weights() -> None:
            try:
                if self.cancelled:
                    return
                have, _ = core.weights_ready(self.cfg)
                if have:
                    self.download = {"state": "done", "percent": 100.0,
                                     "message": "Weights were already on disk"}
                    self._emit("download", download=self.download)
                    return
                dl = core.download_weights(
                    self.cfg, on_progress=self._on_download,
                    cancelled=lambda: self.cancelled)
                self._dl = dl
                if self.cancelled:
                    # Stop arrived while the Download was being built, when
                    # cancel() had nothing to call: honour it here instead.
                    dl.cancel()
                dl.run()
            except Exception as e:            # noqa: BLE001 - reported on the page
                weights_error.append(e)
                self.download = {"state": "error", "percent": self.download.get("percent", 0),
                                 "message": str(e)}
                self._emit("download", download=self.download)

        t = threading.Thread(target=weights, daemon=True)
        t.start()
        try:
            core.install_environment(self.cfg, self.add_log, self.steps,
                                     cancelled=lambda: self.cancelled)
        except Exception as e:                # noqa: BLE001
            self.cancel()
            t.join(timeout=20)
            self._fail(e)
            return
        t.join()
        if self.cancelled:
            self.set_phase("cancelled")
            return
        if weights_error:
            self._fail(weights_error[0])
            return
        self.set_phase("done")

    def _on_download(self, p: dict) -> None:
        self.download = p
        self._emit("download", download=p)


# ------------------------------------------------------------------ http -----

def _peer_ok(client_address) -> bool:
    """The chat UI refuses anything that is not the machine it runs on
    (webui_app._is_local); setup writes .env and runs pip, so it does at least
    as much. A Host header and a custom header stop a browser, but not curl -
    only the socket's peer address does that."""
    if not client_address:
        return True                     # in-process test client
    host = str(client_address[0]).strip().strip("[]").split("%")[0]
    return host in ("127.0.0.1", "::1", "localhost", "::ffff:127.0.0.1") \
        or host.startswith("127.")


def _host_ok(headers) -> bool:
    host = (headers.get("Host") or "").strip()
    name = host.rsplit(":", 1)[0] if ":" in host and not host.endswith("]") else host
    return name.strip("[]").lower() in {h.strip("[]") for h in LOCAL_HOSTS}


def make_handler(setup: Setup):
    class Handler(BaseHTTPRequestHandler):
        server_version = "Simplex-setup"
        protocol_version = "HTTP/1.1"

        # ------------------------------------------------------- plumbing --
        def log_message(self, fmt, *args):    # quiet: the page is the log
            pass

        def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj).encode("utf-8"), "application/json; charset=utf-8")

        def _guard(self, mutating: bool) -> bool:
            if not _peer_ok(getattr(self, "client_address", None)):
                self._json({"error": "setup answers only this computer"}, 403)
                return False
            if not _host_ok(self.headers):
                self._json({"error": "bad host"}, 403)
                return False
            if mutating and (self.headers.get("X-Simplex-UI") or "") != "1":
                self._json({"error": "this request did not come from the setup page"}, 403)
                return False
            return True

        def _body(self) -> dict:
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                return {}
            if n <= 0 or n > 1_000_000:
                return {}
            try:
                return json.loads(self.rfile.read(n).decode("utf-8")) or {}
            except Exception:                 # noqa: BLE001
                return {}

        # ------------------------------------------------------------ GET --
        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            if not self._guard(False):
                return
            path = urllib.parse.urlparse(self.path).path
            query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)

            if path in ("/", "/index.html", "/setup", "/setup/"):
                return self._static("setup.html")
            if path.startswith("/ui/static/"):
                return self._static(path[len("/ui/static/"):])
            if path == "/setup/state":
                return self._json(setup.state())
            if path == "/setup/cards":
                # Its own endpoint as well as a field on /setup/state: the page's
                # static files are read from disk per request, so a browser can be
                # running new JS against a server process started before the code
                # was updated. Then state carries no cards and the picker sits
                # empty with nothing to explain why - this gives it a second
                # source, and a 404 here is what tells the page to say "restart".
                return self._json({"cards": core.cards()})
            if path == "/setup/log":
                return self._json({"log": setup.log[-500:]})
            if path == "/setup/events":
                return self._events(int((query.get("from") or ["0"])[0] or 0))
            if path == "/health":
                # the model server is not up yet; the page uses this to know
                # when to hand over
                return self._json({"status": "setup"}, 503)
            self._json({"error": "not found"}, 404)

        def _static(self, rel: str):
            target = (STATIC / rel).resolve()
            try:
                target.relative_to(STATIC.resolve())
            except ValueError:
                return self._json({"error": "not found"}, 404)
            if not target.is_file():
                return self._json({"error": "not found"}, 404)
            ctype = MIME.get(target.suffix.lower(), "application/octet-stream")
            self._send(200, target.read_bytes(), ctype)

        def _events(self, start: int):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                self.wfile.write(b": open\n\n")
                self.wfile.flush()
                for event in setup.follow(start):
                    self.wfile.write(b"data: " + json.dumps(event).encode("utf-8") + b"\n\n")
                    self.wfile.flush()
                tail = {"i": len(setup.events), "type": "phase",
                        "phase": setup.phase, "state": setup.state()}
                self.wfile.write(b"data: " + json.dumps(tail).encode("utf-8") + b"\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                return
            finally:
                self.close_connection = True

        # ----------------------------------------------------------- POST --
        def do_POST(self):
            if not self._guard(True):
                return
            path = urllib.parse.urlparse(self.path).path
            body = self._body()
            try:
                if path == "/setup/choose":
                    setup.choose(str(body.get("quant") or ""), float(body.get("vram") or 0))
                    return self._json(setup.state())
                if path == "/setup/token":
                    setup.set_token(str(body.get("token") or ""))
                    return self._json({"ok": True})
                if path == "/setup/start":
                    if body.get("quant"):
                        setup.choose(str(body["quant"]), float(body.get("vram") or 0))
                    setup.start()
                    return self._json(setup.state())
                if path == "/setup/cancel":
                    setup.cancel()
                    return self._json(setup.state())
                if path == "/setup/simulate":
                    # Read-only by construction: it calls the planner directly and
                    # never goes near self.cfg or .env, so exploring other cards
                    # mid-install cannot change this install.
                    want = body.get("vision", None)
                    return self._json(core.simulate(
                        float(body.get("vram") or 0),
                        float(body.get("cc") or 0),
                        str(body.get("driver") or ""),
                        str(body.get("name") or ""),
                        None if want is None else bool(want)))
                if path == "/setup/vision":
                    # Its own endpoint, not a flavour of refresh: this is only
                    # ever about images, so an absent / null value unambiguously
                    # means "decide for me" rather than "leave it alone", and it
                    # skips the machine probe that refresh has to do.
                    want = body.get("vision", None)
                    setup.set_vision(None if want is None else bool(want))
                    return self._json(setup.state())
                if path == "/setup/refresh":
                    setup.refresh()
                    return self._json(setup.state())
            except core.SetupError as e:
                return self._json({"error": str(e), "hint": e.hint}, 400)
            except Exception as e:            # noqa: BLE001
                return self._json({"error": str(e) or e.__class__.__name__}, 500)
            self._json({"error": "not found"}, 404)

    return Handler


def _port_free(host: str, port: int) -> bool:
    """Windows lets a second socket bind a port another one is already
    listening on when SO_REUSEADDR is set, so a bind test alone always says
    "free" and the friendly message below is never reached. Ask whether anyone
    answers first; only then try to bind."""
    probe = socket.socket()
    probe.settimeout(0.4)
    try:
        if probe.connect_ex(("127.0.0.1" if host in ("0.0.0.0", "") else host, port)) == 0:
            return False               # something is listening there right now
    except OSError:
        pass
    finally:
        probe.close()
    s = socket.socket()
    try:
        if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            s.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        else:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def serve(cfg: dict, port: int, host: str = "127.0.0.1", console=None, force: bool = False):
    """Start the setup server. Returns (setup, httpd, url)."""
    setup = Setup(cfg, console=console, force=force)
    handler = make_handler(setup)
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.2},
                     daemon=True).start()
    setup.refresh()
    return setup, httpd, f"http://{'127.0.0.1' if host in ('0.0.0.0', '') else host}:{port}/"


def _wait_for_retry(setup: Setup, seconds: float) -> bool:
    """True if setup left the error state again within the window - which is
    what pressing Try again or Choose a different model does."""
    deadline = time.time() + seconds
    while time.time() < deadline:
        if not setup.terminal:
            return True
        if setup.phase != "error":
            return True
        time.sleep(0.25)
    return False


def run(cfg: dict, port: int, host: str = "127.0.0.1", open_browser: bool = True,
        console=None, banner=None, force: bool = False,
        retry_message=None) -> tuple[str, dict | None]:
    """Blocking: put setup in the browser and come back when it is finished.

    Returns (outcome, config). The outcome is "done", "cancelled" or "error";
    the caller needs the difference, because a cancelled setup means the person
    made a decision and a failed one means they were told why on the page. In
    neither case should the console start asking the same questions again."""
    if not _port_free(host, port):
        raise core.SetupError(
            f"Port {port} is already in use, so the setup page cannot open there.",
            "Another copy of Simplex is probably running. Close it, or set a "
            "different PORT in .env.")
    setup, httpd, url = serve(cfg, port, host, console, force=force)
    if banner:
        banner(url)
    if open_browser:
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception:                     # noqa: BLE001
            pass
    try:
        while True:
            while not setup.finished.wait(timeout=0.5):
                pass
            if setup.phase == "done":
                break
            if setup.phase == "cancelled":
                time.sleep(1.5)          # let the page draw it, then stop
                break
            # An error is the one case where the user has something to do: fix
            # what the page told them about and press Try again. Tearing the
            # server down immediately made both buttons on that screen dead.
            if retry_message:
                retry_message(RETRY_GRACE)
            if not _wait_for_retry(setup, RETRY_GRACE):
                break
    except KeyboardInterrupt:
        setup.cancel()
        raise
    finally:
        httpd.shutdown()
        httpd.server_close()
    if setup.phase != "done":
        return setup.phase if setup.phase in ("cancelled", "error") else "error", None
    import win_start
    return "done", win_start.load_dotenv(core.ENV_FILE)


def main(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Simplex first-run setup, in a browser")
    ap.add_argument("--port", type=int, default=8888)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args(argv)
    import win_start
    cfg = win_start.load_dotenv(core.ENV_FILE)

    def console(line: str, kind: str) -> None:
        prefix = {"cmd": "  > ", "error": "  ! ", "hint": "    ", "out": "    "}.get(kind, "    ")
        print(prefix + line, flush=True)

    outcome, _cfg = run(cfg, a.port, a.host, not a.no_browser, console=console,
                        banner=lambda u: print(f"\n  Setup is open at {u}\n"),
                        retry_message=lambda s: print(
                            f"\n  Setup failed. The page is still open for "
                            f"{int(s / 60)} minutes if you want to try again.\n"))
    print(f"\n  setup: {outcome}\n")
    return 0 if outcome == "done" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
