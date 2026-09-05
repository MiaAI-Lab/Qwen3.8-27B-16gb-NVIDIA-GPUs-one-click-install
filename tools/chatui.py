#!/usr/bin/env python3
"""Mounts the built-in web UI, and runs it standalone for development.

Two adapters over the same core (tools/webui_app.py):

  mount(app, ui)   adds the UI routes to the running aiohttp model server, so
                   http://127.0.0.1:<PORT>/ is the chat app and /v1 stays a
                   plain OpenAI endpoint for any other client.

  main()           a stdlib http.server that serves the UI on its own port and
                   talks to a model server elsewhere - or to --mock, which
                   scripts replies and tool calls so the UI can be worked on
                   without loading 10 GB of weights.

Standalone use:
    python tools/chatui.py --mock                  # UI + fake model on :8890
    python tools/chatui.py --model-base http://127.0.0.1:8888/v1
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))

from webui_app import ChatUI, Response, Stream       # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SSE_HEADERS = {"Content-Type": "text/event-stream", "Cache-Control": "no-cache",
               "Connection": "keep-alive", "X-Accel-Buffering": "no"}


def sse(event) -> bytes:
    return f"data: {json.dumps(event)}\n\n".encode("utf-8")


# ------------------------------------------------------- aiohttp adapter ----

def mount(app, ui: ChatUI) -> None:
    """Add the UI routes to an aiohttp application (the model server)."""
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    from aiohttp import web
    import webui_models

    # The UI's own pool. asyncio's default executor is also where the model
    # server runs generation; a UI turn holds a thread for its whole length and
    # calls back into /v1, so sharing one pool can deadlock both.
    pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="chatui")
    app.on_cleanup.append(lambda _app: asyncio.to_thread(pool.shutdown, False))

    def restart():
        """Switching models means loading different weights into the same VRAM,
        so the process leaves and the launcher starts it again (exit 87). The
        delay lets the HTTP response reach the browser first."""
        threading.Timer(0.7, lambda: os._exit(webui_models.RESTART_CODE)).start()

    ui.on_restart = restart

    async def handler(request):
        raw = await request.read() if request.method in ("POST", "PUT") else b""
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            pool, ui.handle, request.method, request.path,
            {k: v for k, v in request.query.items()}, raw, request.remote,
            dict(request.headers))

        if isinstance(result, Response):
            return web.Response(status=result.status, body=result.body,
                                content_type=result.content_type.split(";")[0],
                                charset="utf-8" if "charset" in result.content_type
                                else None,
                                headers=result.headers or None)

        resp = web.StreamResponse(headers=SSE_HEADERS)
        await resp.prepare(request)
        queue: asyncio.Queue = asyncio.Queue()

        def pump():
            try:
                for event in result.events:
                    loop.call_soon_threadsafe(queue.put_nowait, event)
            except Exception as e:                      # noqa: BLE001
                loop.call_soon_threadsafe(queue.put_nowait, {
                    "type": "error", "message": f"{type(e).__name__}: {e}"})
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        loop.run_in_executor(pool, pump)
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                await resp.write(sse(event))
        except (ConnectionError, asyncio.CancelledError):
            # The tab was closed. That ends this *view* of the turn, not the
            # turn: it keeps running in the session, finishes its tools, saves
            # its answer, and is replayed when the browser comes back. Only
            # Stop cancels.
            raise
        finally:
            try:
                await resp.write_eof()
            except Exception:                           # noqa: BLE001
                pass
        return resp

    for pattern in ("/", "/index.html", "/ui", "/ui/{tail:.*}"):
        app.router.add_route("*", pattern, handler)


# --------------------------------------------------- stdlib dev adapter ----

class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    ui: ChatUI = None                                   # set by serve()

    def log_message(self, fmt, *args):                  # quieter console
        if os.environ.get("CHATUI_VERBOSE"):
            super().log_message(fmt, *args)

    def _dispatch(self, method):
        """One request, with the client allowed to leave at any point.

        A browser hangs up constantly and legitimately - a refresh, a closed
        tab, Stop aborting the fetch part-way through a streamed turn - and the
        write that was in flight fails. Only two of the three shapes that takes
        were caught, and Windows raises the third: WinError 10053 arrives as
        ConnectionAbortedError, so an ordinary refresh printed a nine-frame
        traceback into the console the person is using as their log.

        All three are ConnectionError. None of them is a fault, and none of
        them touches the turn: it keeps running in its session, finishes its
        tools, saves its answer, and is replayed when the browser comes back.
        """
        try:
            self._respond(method)
        except ConnectionError as e:
            self.close_connection = True     # the socket is gone; do not read on
            if os.environ.get("CHATUI_VERBOSE"):
                self.log_message("client went away: %s", e)

    def _respond(self, method):
        parsed = urlparse(self.path)
        query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        result = self.ui.handle(method, parsed.path, query, raw,
                                self.client_address[0], dict(self.headers))
        if isinstance(result, Response):
            self.send_response(result.status)
            self.send_header("Content-Type", result.content_type)
            self.send_header("Content-Length", str(len(result.body)))
            for key, value in (result.headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(result.body)
            return
        self.send_response(200)
        for k, v in SSE_HEADERS.items():
            self.send_header(k, v)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for event in result.events:
            chunk = sse(event)
            self.wfile.write(f"{len(chunk):X}\r\n".encode())
            self.wfile.write(chunk + b"\r\n")
            self.wfile.flush()
        self.wfile.write(b"0\r\n\r\n")

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def do_DELETE(self):
        self._dispatch("DELETE")


class _Server(ThreadingHTTPServer):
    """The same forgiveness one level up.

    socketserver prints "Exception occurred during processing of request from"
    and a full traceback for anything that escapes a handler - including the
    writes it does itself, on teardown, after the client has gone. A browser
    that closed its tab is not an error worth a traceback in a console someone
    is reading as a log."""

    daemon_threads = True

    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1], ConnectionError):
            return
        super().handle_error(request, client_address)


def load_env(path: Path) -> dict:
    cfg = {}
    if not path.is_file():
        return cfg
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        cfg[key.strip()] = value.split("#")[0].strip().strip('"').strip("'")
    return cfg


# ------------------------------------------------------------ restarting ----
# Starting the UI should mean "give me the UI", not "fail because the copy I
# opened an hour ago still holds the port". The old instance is also running the
# old code, which is exactly what you were trying to get rid of. So take the
# port back - but only from something that is recognisably this same UI, never
# from whatever else happened to be listening.

def _listening_pid(port: int) -> int | None:
    """PID of the process listening on `port`, or None. stdlib only."""
    if os.name == "nt":
        try:
            out = subprocess.run(["netstat", "-ano", "-p", "tcp"],
                                 capture_output=True, text=True, timeout=15).stdout
        except (OSError, subprocess.SubprocessError):
            return None
        for line in out.splitlines():
            parts = line.split()
            # Proto  Local            Foreign          State      PID
            if len(parts) >= 5 and parts[3].upper() == "LISTENING" \
                    and parts[1].rsplit(":", 1)[-1] == str(port):
                try:
                    return int(parts[4])
                except ValueError:
                    continue
        return None
    for cmd in (["lsof", "-nP", "-tiTCP:%d" % port, "-sTCP:LISTEN"],
                ["ss", "-ltnpH", "sport = :%d" % port]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=15).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        if cmd[0] == "lsof":
            for line in out.split():
                if line.strip().isdigit():
                    return int(line.strip())
        else:
            m = re.search(r"pid=(\d+)", out)
            if m:
                return int(m.group(1))
    return None


def _cmdline(pid: int) -> str:
    """The process's command line, lower-cased. Empty when it cannot be read."""
    if os.name == "nt":
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 f"(Get-CimInstance Win32_Process -Filter 'ProcessId={pid}').CommandLine"],
                capture_output=True, text=True, timeout=20).stdout
        except (OSError, subprocess.SubprocessError):
            return ""
        return out.strip().lower()
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\0", b" ").decode("utf-8", "replace").strip().lower()


def _is_this_ui(cmdline: str) -> bool:
    """True only for another copy of this program - never anything else."""
    return "chatui" in cmdline and "python" in cmdline


def stop_existing(port: int, timeout: float = 10.0, log=print) -> str:
    """Free `port` if this same UI is holding it. Returns what happened."""
    pid = _listening_pid(port)
    if not pid:
        return "nothing was listening"
    if pid == os.getpid():
        return "that is this process"
    cmd = _cmdline(pid)
    if cmd and not _is_this_ui(cmd):
        return (f"port {port} is held by something else (pid {pid}) - "
                "leaving it alone; use --port to pick another")
    log(f"stopping the copy already on port {port} (pid {pid}) ...")
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid)], capture_output=True, timeout=15)
        else:
            os.kill(pid, signal.SIGTERM)
    except (OSError, subprocess.SubprocessError) as e:
        return f"could not stop pid {pid}: {e}"

    deadline = time.time() + timeout
    while time.time() < deadline:
        if _listening_pid(port) is None:
            return f"stopped pid {pid}"
        time.sleep(0.25)
    # it ignored the polite request
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/F"],
                           capture_output=True, timeout=15)
        else:
            os.kill(pid, signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        pass
    deadline = time.time() + 5
    while time.time() < deadline:
        if _listening_pid(port) is None:
            return f"force-stopped pid {pid}"
        time.sleep(0.25)
    return f"pid {pid} would not let go of port {port}"


def build_ui(cfg, model_base, model_id, root=ROOT, context=None, vision=False):
    return ChatUI(root=root, cfg=cfg, model_base=model_base, model_id=model_id,
                  context_length=context, vision=vision)


def main() -> int:
    ap = argparse.ArgumentParser(description="run the chat UI standalone")
    ap.add_argument("--port", type=int, default=8890)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--model-base", default=None,
                    help="OpenAI base URL (default: PORT from .env on localhost)")
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--root", default=str(ROOT),
                    help="kit root (sessions/ and the default workspace live here)")
    ap.add_argument("--mock", action="store_true",
                    help="serve a scripted fake model - for UI work with no GPU")
    ap.add_argument("--open", action="store_true", help="open a browser")
    ap.add_argument("--no-restart", action="store_true",
                    help="fail if the port is busy instead of replacing the copy "
                         "already running there")
    args = ap.parse_args()

    cfg = load_env(Path(args.root) / ".env")
    base = args.model_base or f"http://127.0.0.1:{cfg.get('PORT', '8888')}/v1"
    model_id = args.model_id or cfg.get("MODEL_ID") or "local-model"
    ui = build_ui(cfg, base, model_id, root=Path(args.root),
                  context=int(cfg.get("CONTEXT_SIZE") or 0) or None)
    if args.mock:
        from webui_mock import MockClient                # noqa: WPS433
        ui.client = MockClient()
        ui.model_id = model_id = "mock-model"
        ui.fake_health = True       # so the UI's live tok/s has something to read
        # a dev server has no launcher to restart it; pretend, so the model
        # picker can be worked on without a GPU
        ui.supervised = True
        ui.on_restart = lambda: print("  (dev) a restart would happen here",
                                      flush=True)
        ui.context_length = ui.context_length or 199936   # so the meter renders
    _Handler.ui = ui

    if not args.no_restart:
        said = stop_existing(args.port, log=lambda m: print(f"  {m}", flush=True))
        if said not in ("nothing was listening", "that is this process"):
            print(f"  {said}", flush=True)

    try:
        server = _Server((args.host, args.port), _Handler)
    except OSError as e:
        print(f"\n  Cannot use port {args.port}: {e}")
        print("  Something else is holding it. Close it, or start with "
              "--port <other>.\n", flush=True)
        return 1
    url = f"http://{args.host}:{args.port}/"
    print(f"  chat UI  {url}")
    print(f"  model    {'MOCK (scripted replies)' if args.mock else base}")
    print("  Ctrl+C to stop", flush=True)
    if args.open:
        import webbrowser
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
