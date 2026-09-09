"""simplex - one command for the whole kit, on Windows and on Linux.

    simplex setup                 install the environment and fetch a model
    simplex start                 load a model and serve it
    simplex start --no-harness    ...without the DeepSeek Harness
    simplex start -b              ...in the background, log in logs/
    simplex stop                  stop the server and the harness
    simplex restart               stop, then start again
    simplex status                what is running, and where
    simplex logs -f               follow the launcher log
    simplex harness start|stop|status|open|settings
    simplex models                what is on the disk
    simplex doctor                check this machine and this install

Why this exists: windows\\START-HERE.bat/setup.sh, windows\\start.bat/start.sh and windows\\stop.bat/stop.sh
are six files, three of them written twice - once in cmd and once in bash -
which is how Windows and Linux drifted apart (a first run on Linux used to
stop and send you to the other script; a stop on Linux always ended in a
SIGKILL). Everything below is one implementation that runs on both.

The heavy lifting is still the code that was already here and already tested:
`setup` and `start` hand over to tools/win_start.py on Windows and to linux/start.sh
on Linux. What lives here is the part that was duplicated - finding out what is
running, stopping it politely, reading the log, and saying what is wrong.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
LOGS = ROOT / "logs"
ENV_FILE = ROOT / ".env"
ENV_EXAMPLE = ROOT / ".env.example"
SERVER_PID = LOGS / "server.pid"
HARNESS_PID = LOGS / "harness.pid"
WINDOWS = os.name == "nt"

if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))


# --------------------------------------------------------------- colour ----
def _colour_ok() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if not sys.stdout.isatty():
        return False
    if WINDOWS:
        try:
            import ctypes
            k = ctypes.windll.kernel32
            k.SetConsoleMode(k.GetStdHandle(-11), 7)
        except Exception:                                   # noqa: BLE001
            return False
    return True


_C = _colour_ok()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _C else text


def bold(t):  return _c("1", t)
def dim(t):   return _c("2", t)
def green(t): return _c("32", t)
def yellow(t): return _c("33", t)
def red(t):   return _c("31", t)
def cyan(t):  return _c("36", t)


def out(msg: str = "") -> None:
    print(msg, flush=True)


def fail(msg: str, code: int = 1):
    print(red("  error  ") + msg, file=sys.stderr, flush=True)
    return code


# ------------------------------------------------------------------ .env ----
def read_env(path: Path = ENV_FILE) -> dict:
    """.env as data, never as shell. Same rules as linux/start.sh and win_start.py:
    KEY=value, a trailing comment, quotes stripped, CRLF tolerated."""
    cfg: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return cfg
    for line in text.splitlines():
        line = line.strip().lstrip("﻿")
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #")[0].strip()
        cfg[key] = value
    return cfg


def env_int(cfg: dict, key: str, default: int) -> int:
    m = re.match(r"\s*(\d+)", cfg.get(key, "") or "")
    return int(m.group(1)) if m else default


def venv_python() -> Path:
    return ROOT / ".venv" / ("Scripts/python.exe" if WINDOWS else "bin/python")


def any_python() -> str:
    vp = venv_python()
    return str(vp) if vp.is_file() else sys.executable


# ------------------------------------------------------------ processes ----
def port_listening(port: int, host: str = "127.0.0.1") -> bool:
    s = socket.socket()
    s.settimeout(0.4)
    try:
        return s.connect_ex((host, int(port))) == 0
    except OSError:
        return False
    finally:
        s.close()


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if WINDOWS:
        try:
            r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                               capture_output=True, text=True, timeout=10)
            return str(pid) in r.stdout
        except Exception:                                   # noqa: BLE001
            return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        return True


def _cmdline(pid: int) -> str:
    """The process's command line, for the "is this really ours?" check."""
    if WINDOWS:
        try:
            r = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 f"(Get-CimInstance Win32_Process -Filter 'ProcessId={pid}')."
                 "CommandLine"],
                capture_output=True, text=True, timeout=20)
            return (r.stdout or "").strip()
        except Exception:                                   # noqa: BLE001
            return ""
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        return raw.replace(b"\0", b" ").decode("utf-8", "replace").strip()
    except OSError:
        return ""


def port_owner(port: int) -> tuple[int, str]:
    """(pid, command line) of whatever is listening on `port`, or (0, "").

    No iproute2 and no lsof: /proc on Linux, Get-NetTCPConnection on Windows.
    linux/stop.sh needed `ss`, which is not on every image."""
    port = int(port)
    if WINDOWS:
        try:
            r = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 f"(Get-NetTCPConnection -LocalPort {port} -State Listen "
                 "-ErrorAction SilentlyContinue | Select-Object -First 1)"
                 ".OwningProcess"],
                capture_output=True, text=True, timeout=20)
            pid = int((r.stdout or "0").strip() or 0)
        except Exception:                                   # noqa: BLE001
            pid = 0
        return (pid, _cmdline(pid)) if pid else (0, "")

    inodes = []
    for proc_net in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(proc_net) as fh:
                next(fh, None)
                for line in fh:
                    f = line.split()
                    if len(f) < 10 or f[3] != "0A":          # 0A = LISTEN
                        continue
                    if int(f[1].split(":")[1], 16) == port:
                        inodes.append(f[9])
        except OSError:
            continue
    if not inodes:
        return 0, ""
    wanted = {f"socket:[{i}]" for i in inodes}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        fddir = f"/proc/{entry}/fd"
        try:
            for fd in os.listdir(fddir):
                try:
                    if os.readlink(f"{fddir}/{fd}") in wanted:
                        return int(entry), _cmdline(int(entry))
                except OSError:
                    continue
        except OSError:
            continue
    return 0, ""                       # listening, but owned by another user


def read_pidfile(path: Path) -> int:
    try:
        pid = int(path.read_text().strip() or 0)
    except (OSError, ValueError):
        return 0
    return pid if _pid_alive(pid) else 0


def stop_pid(pid: int, timeout: float = 20.0, label: str = "process") -> bool:
    """Ask politely, then insist. Returns True once the process is gone.

    SIGTERM rather than SIGINT: a launcher started in the background runs as a
    job of a non-interactive shell, which sets SIGINT to SIG_IGN, and the child
    inherits that across exec. linux/stop.sh sent INT to the server and so always sat
    out its ten seconds and finished with a SIGKILL - the same bug that was
    already fixed for the harness. On Windows there is no SIGTERM to send, so
    taskkill without /F asks first and /F is the fallback."""
    if not _pid_alive(pid):
        return True
    try:
        if WINDOWS:
            subprocess.run(["taskkill", "/PID", str(pid), "/T"],
                           capture_output=True, timeout=20)
        else:
            os.kill(pid, signal.SIGTERM)
    except Exception:                                       # noqa: BLE001
        pass
    # A console process with no window often cannot be closed politely on
    # Windows at all ("can only be terminated forcefully"), so waiting out a
    # long timeout there buys nothing - windows\stop.bat has always forced after a
    # second. On POSIX the wait is worth it: SIGTERM is how the server gets to
    # release the card and its cache pages before it goes.
    polite = min(timeout, 3.0) if WINDOWS else timeout
    step = 0.5 if WINDOWS else 0.25
    deadline = time.time() + polite
    while time.time() < deadline:
        if not _pid_alive(pid):
            return True
        time.sleep(step)
    out(f"  {label} did not stop in {polite:.0f}s - forcing it.")
    try:
        if WINDOWS:
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                           capture_output=True, timeout=20)
        else:
            os.kill(pid, signal.SIGKILL)
    except Exception:                                       # noqa: BLE001
        pass
    deadline = time.time() + 5
    while time.time() < deadline:
        if not _pid_alive(pid):
            return True
        time.sleep(0.25)
    return not _pid_alive(pid)


# -------------------------------------------------------------- the kit ----
def http_json(url: str, timeout: float = 3.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read() or b"{}")
    except (urllib.error.URLError, OSError, ValueError):
        return None


def harness_url() -> str | None:
    try:
        return (ROOT / ".dsh" / "url").read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def newest_log() -> Path | None:
    logs = sorted(LOGS.glob("simplex-*.log"), key=lambda p: p.stat().st_mtime
                  if p.exists() else 0, reverse=True)
    return logs[0] if logs else None


def snapshot() -> dict:
    """Everything `status` reports, in one dict - also what --json prints."""
    cfg = read_env()
    port = env_int(cfg, "PORT", 8888)
    dsh_port = env_int(cfg, "DSH_PORT", 3080)
    state = {
        "root": str(ROOT),
        "server": {"port": port, "running": False, "pid": 0, "model": None,
                   "context": None, "vision": None, "efforts": [], "ours": None},
        "harness": {"port": dsh_port, "running": False, "pid": 0, "url": None},
        "log": str(newest_log()) if newest_log() else None,
        "profile": cfg.get("PROFILE") or None,
        "ui": (cfg.get("UI") or "browser"),
    }

    if port_listening(port):
        state["server"]["running"] = True
        pid, cmd = port_owner(port)
        state["server"]["pid"] = pid
        state["server"]["ours"] = ("serve_openai" in cmd) if cmd else None
        health = http_json(f"http://127.0.0.1:{port}/health")
        if health:
            state["server"]["context"] = health.get("context_length")
            state["server"]["vision"] = health.get("vision")
            state["server"]["ours"] = True
        row = (http_json(f"http://127.0.0.1:{port}/v1/models") or {}).get("data")
        if row:
            state["server"]["model"] = row[0].get("id")
            state["server"]["efforts"] = row[0].get(
                "supported_reasoning_efforts") or []
    if not state["server"]["pid"]:
        state["server"]["pid"] = read_pidfile(SERVER_PID)

    if port_listening(dsh_port):
        state["harness"]["running"] = True
        pid, _cmd = port_owner(dsh_port)
        state["harness"]["pid"] = pid or read_pidfile(HARNESS_PID)
        state["harness"]["url"] = harness_url()
    return state


# ------------------------------------------------------------- delegation ---
def launcher_cmd(mode: str, extra: list[str]) -> list[str]:
    """The command that actually installs or starts - the code that was here
    before this file and is what the test suite covers."""
    if WINDOWS:
        return [any_python(), str(TOOLS / "win_start.py")] + \
               (["setup"] if mode == "setup" else []) + extra
    bash = shutil.which("bash") or "/bin/bash"
    return [bash, str(ROOT / "linux/start.sh")] + \
           (["setup"] if mode == "setup" else []) + extra


def child_env(harness: str | None) -> dict:
    env = dict(os.environ)
    if harness is not None:
        # An override .env cannot win, on purpose: .env is read after the
        # environment in both launchers, so a flag needs a name of its own.
        env["SIMPLEX_UI"] = harness
    return env


def run_launcher(mode: str, harness: str | None, extra: list[str]) -> int:
    cmd = launcher_cmd(mode, extra)
    try:
        return subprocess.call(cmd, cwd=str(ROOT), env=child_env(harness))
    except FileNotFoundError as e:
        return fail(f"cannot run the launcher ({e}). "
                    f"Is this the kit's own folder? {ROOT}")
    except KeyboardInterrupt:
        return 130


def start_detached(mode: str, harness: str | None, extra: list[str]) -> int:
    """Background start, on both systems. linux/start.sh had --background; Windows
    had nothing, so `windows\\start.bat` always owned a console window."""
    LOGS.mkdir(exist_ok=True)
    log = LOGS / time.strftime("simplex-%Y-%m-%d_%H%M%S.log")
    cmd = launcher_cmd(mode, extra)
    kwargs = {}
    if WINDOWS:
        kwargs["creationflags"] = (subprocess.CREATE_NEW_PROCESS_GROUP |
                                   0x00000008 |          # DETACHED_PROCESS
                                   0x08000000)           # CREATE_NO_WINDOW
    else:
        kwargs["start_new_session"] = True
    with open(log, "wb") as fh:
        proc = subprocess.Popen(cmd, cwd=str(ROOT), env=child_env(harness),
                                stdout=fh, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, **kwargs)
    SERVER_PID.write_text(str(proc.pid))
    cfg = read_env()
    port = env_int(cfg, "PORT", 8888)
    out(f"  Starting in the background (pid {proc.pid}).")
    out(f"    log:    {dim(str(log))}")
    out(f"    stop:   simplex stop")
    out(f"    check:  simplex status")
    time.sleep(3)
    if proc.poll() is not None:
        out()
        out(red("  It stopped straight away. The end of the log:"))
        try:
            for line in log.read_text(errors="replace").splitlines()[-20:]:
                out("  | " + line)
        except OSError:
            pass
        SERVER_PID.unlink(missing_ok=True)
        return 1
    out()
    out(f"  It is starting. When the model is loaded it answers on "
        f"{cyan(f'http://127.0.0.1:{port}/')}")
    return 0


# ----------------------------------------------------------------- verbs ----
def cmd_setup(a) -> int:
    if not ENV_FILE.is_file():
        if not ENV_EXAMPLE.is_file():
            return fail(".env.example is missing; cannot create .env")
        shutil.copyfile(ENV_EXAMPLE, ENV_FILE)
        out(f"  Created .env from .env.example.")
    return run_launcher("setup", None, a.rest)


def cmd_start(a) -> int:
    if not ENV_FILE.is_file():
        if not ENV_EXAMPLE.is_file():
            return fail(".env.example is missing; cannot create .env")
        shutil.copyfile(ENV_EXAMPLE, ENV_FILE)
        out("  Created .env from .env.example.")

    cfg = read_env()
    port = a.port or env_int(cfg, "PORT", 8888)

    # Look before loading. A second start used to run the whole model load -
    # minutes, and the VRAM for it - before the bind failed on a port that was
    # already taken. The harness port has been checked like this all along.
    if port_listening(port):
        pid, cmd = port_owner(port)
        if not cmd or "serve_openai" in cmd:
            out(yellow("  A model server is already running on port "
                       f"{port}") + f" (pid {pid or '?'}).")
            out("  Stop it with " + bold("simplex stop") + ", or see "
                + bold("simplex status") + ".")
            return 0
        out(red(f"  Port {port} is held by something else") +
            f" (pid {pid}): {cmd[:70]}")
        out("  Close it, or set a different PORT in .env.")
        return 1

    harness = {"on": "browser", "off": "no", None: None}[a.harness]
    extra = list(a.rest)
    if a.no_pick:
        extra.append("--no-pick")
    if a.background:
        return start_detached("start", harness, extra)
    return run_launcher("start", harness, extra)


def cmd_stop(a) -> int:
    cfg = read_env()
    port = a.port or env_int(cfg, "PORT", 8888)
    dsh_port = env_int(cfg, "DSH_PORT", 3080)
    rc = 0

    if not a.server_only:
        pid, cmd = port_owner(dsh_port)
        if pid and (not cmd or re.search(r"dsh|deepseek-harness|node", cmd)):
            out(f"  Stopping the harness (pid {pid}) on port {dsh_port} ...")
            if stop_pid(pid, a.timeout, "the harness"):
                out(green("  Harness stopped."))
                HARNESS_PID.unlink(missing_ok=True)
                (ROOT / ".dsh" / "url").unlink(missing_ok=True)
            else:
                out(red(f"  Port {dsh_port} is still open."))
                rc = 1
        elif pid:
            out(f"  Port {dsh_port} is held by {cmd[:60]} - leaving it alone.")
        elif a.harness_only:
            out(f"  No harness listening on port {dsh_port}.")

    if a.harness_only:
        return rc

    pid, cmd = port_owner(port)
    if not pid and not port_listening(port):
        out(f"  No server listening on port {port}.")
        _clear_stale_pidfile()
        return rc
    if cmd and "serve_openai" not in cmd and "win_start" not in cmd:
        out(red(f"  Port {port} is held by another process") +
            f" (pid {pid}): {cmd[:70]}")
        out("  Refusing to kill it.")
        return 1
    if not pid:
        pid = read_pidfile(SERVER_PID)
    if not pid:
        out(f"  Something holds port {port} but this account cannot see which "
            f"process. Stop it from the window it runs in.")
        return 1
    out(f"  Stopping the server (pid {pid}) on port {port} ...")
    if stop_pid(pid, a.timeout, "the server"):
        out(green("  Stopped."))
        SERVER_PID.unlink(missing_ok=True)
    else:
        out(red(f"  ERROR: port {port} is still open."))
        rc = 1
    _clear_stale_pidfile()
    return rc


def _clear_stale_pidfile() -> None:
    """A pidfile whose process is gone is a lie `status` would repeat."""
    if SERVER_PID.exists() and not read_pidfile(SERVER_PID):
        SERVER_PID.unlink(missing_ok=True)


def cmd_restart(a) -> int:
    rc = cmd_stop(argparse.Namespace(port=a.port, timeout=a.timeout,
                                     server_only=False, harness_only=False))
    if rc:
        return rc
    time.sleep(1.0)
    return cmd_start(a)


def cmd_status(a) -> int:
    st = snapshot()
    if a.json:
        out(json.dumps(st, indent=2))
        return 0 if st["server"]["running"] else 1

    s, h = st["server"], st["harness"]
    out()
    if s["running"]:
        who = "" if s["ours"] is not False else yellow("  (not this kit)")
        out(f"  server    {green('running')}  "
            f"http://127.0.0.1:{s['port']}/v1"
            f"{'  pid ' + str(s['pid']) if s['pid'] else ''}{who}")
        if s["model"]:
            bits = [f"model {bold(s['model'])}"]
            if s["context"]:
                bits.append(f"context {s['context']:,}")
            bits.append("images " + ("on" if s["vision"] else "off"))
            if s["efforts"]:
                bits.append("effort " + "/".join(s["efforts"]))
            out("            " + dim("  |  ".join(bits)))
    else:
        out(f"  server    {dim('not running')}  (port {s['port']})")

    if h["running"]:
        out(f"  harness   {green('running')}  "
            f"http://127.0.0.1:{h['port']}/"
            f"{'  pid ' + str(h['pid']) if h['pid'] else ''}")
        if h["url"]:
            out("            " + dim(h["url"]))
    else:
        want = str(st["ui"]).lower() not in ("no", "off", "0", "none", "false")
        out(f"  harness   {dim('not running')}  (port {h['port']}"
            f"{'' if want else ', UI=no in .env'})")

    if st["profile"]:
        out(f"  profile   {st['profile']}")
    if st["log"]:
        out(f"  log       {dim(st['log'])}")
    out()
    return 0 if s["running"] else 1


def cmd_logs(a) -> int:
    log = Path(a.file) if a.file else newest_log()
    if not log or not log.exists():
        return fail("no log yet - start something first.")
    out(dim(f"  {log}"))
    if not a.follow:
        try:
            lines = log.read_text(errors="replace").splitlines()
        except OSError as e:
            return fail(str(e))
        for line in lines[-a.lines:]:
            out(line)
        return 0
    with open(log, "r", errors="replace") as fh:
        fh.seek(0, os.SEEK_END)
        try:
            while True:
                line = fh.readline()
                if line:
                    out(line.rstrip("\n"))
                else:
                    time.sleep(0.3)
        except KeyboardInterrupt:
            return 0


def cmd_harness(a) -> int:
    import dsh                                              # noqa: WPS433
    cfg = read_env()
    port = env_int(cfg, "DSH_PORT", dsh.DEFAULT_PORT)
    base = f"http://127.0.0.1:{env_int(cfg, 'PORT', 8888)}/v1"

    if a.action == "status":
        if dsh.listening(port):
            out(f"  harness   {green('running')}  http://127.0.0.1:{port}/")
            if harness_url():
                out("            " + dim(harness_url()))
            return 0
        out(f"  harness   {dim('not running')}  (port {port})")
        return 1

    if a.action == "stop":
        return cmd_stop(argparse.Namespace(port=None, timeout=a.timeout,
                                           server_only=False, harness_only=True))

    if a.action == "open":
        url = harness_url()
        if not url:
            return fail("the harness is not running - `simplex harness start`.")
        out(f"  {cyan(url)}")
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception:                                   # noqa: BLE001
            pass
        return 0

    # start / settings both need the server to be up: everything the harness is
    # told about the model is read off /v1/models rather than out of .env.
    if not port_listening(env_int(cfg, "PORT", 8888)):
        return fail("the model server is not running - "
                    "`simplex start` first (the harness is configured from it).")

    if a.action == "settings":
        return subprocess.call(
            [any_python(), str(TOOLS / "dsh.py"), "--base", base,
             "--port", str(port), "--settings-only"], cwd=str(ROOT))

    if dsh.listening(port):
        out(f"  A harness is already running on port {port}.")
        if harness_url():
            out("  " + cyan(harness_url()))
        return 0

    argv = [any_python(), str(TOOLS / "dsh.py"), "--base", base,
            "--port", str(port), "--wait", "600"]
    if a.foreground:
        if not a.no_open:
            argv.append("--open")
        return subprocess.call(argv, cwd=str(ROOT))

    # Detached by default. `simplex harness start` is "attach a harness to the
    # server that is already running", and a verb that then holds the terminal
    # until you Ctrl-C it is not that.
    LOGS.mkdir(exist_ok=True)
    log = LOGS / time.strftime("harness-%Y-%m-%d_%H%M%S.log")
    kwargs = {}
    if WINDOWS:
        kwargs["creationflags"] = (subprocess.CREATE_NEW_PROCESS_GROUP |
                                   0x00000008 | 0x08000000)
    else:
        kwargs["start_new_session"] = True
    with open(log, "wb") as fh:
        proc = subprocess.Popen(argv, cwd=str(ROOT), stdout=fh,
                                stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, **kwargs)
    HARNESS_PID.write_text(str(proc.pid))
    deadline = time.time() + 300           # a first run downloads it from npm
    while time.time() < deadline:
        if proc.poll() is not None:
            out(red("  The harness stopped straight away. The end of its log:"))
            for line in log.read_text(errors="replace").splitlines()[-15:]:
                out("  | " + line)
            HARNESS_PID.unlink(missing_ok=True)
            return 1
        url = harness_url()
        if url and dsh.listening(port):
            out(f"  harness   {green('running')}  pid {proc.pid}")
            out("  " + cyan(url))
            out(dim(f"  log: {log}"))
            if not a.no_open:
                try:
                    import webbrowser
                    webbrowser.open(url)
                except Exception:                           # noqa: BLE001
                    pass
            return 0
        time.sleep(0.5)
    out(yellow(f"  The harness has not answered on port {port} yet."))
    out(dim(f"  Watch it:  simplex logs --file {log} -f"))
    return 1


def cmd_models(a) -> int:
    import downloader                                       # noqa: WPS433
    models_dir = ROOT / "models"
    rows = []
    if models_dir.is_dir():
        for d in sorted(models_dir.iterdir()):
            if not d.is_dir():
                continue
            state = downloader.folder_state(d)
            rows.append({"name": d.name, "state": state["state"],
                         "reason": state["reason"],
                         "gb": round(state["bytes"] / 1024 ** 3, 1)})
    if a.json:
        out(json.dumps(rows, indent=2))
        return 0
    if not rows:
        out("  Nothing downloaded yet - run " + bold("simplex setup") + ".")
        return 1
    cfg = read_env()
    current = (cfg.get("MODEL_DIR") or "").replace("\\", "/").rstrip("/").split("/")[-1]
    out()
    for r in rows:
        mark = green(" <- current") if r["name"] == current else ""
        word = {"complete": green("ready"), "partial": yellow("partial"),
                "missing": dim("empty")}[r["state"]]
        out(f"  {word:<18}  {r['gb']:>6.1f} GB  {r['name']}{mark}")
        if r["state"] != "complete" and r["reason"]:
            out(f"  {'':<18}  {dim(r['reason'])}")
    out()
    return 0


def cmd_doctor(a) -> int:
    import downloader                                       # noqa: WPS433
    cfg = read_env()
    problems, notes = [], []

    def ok(label, detail=""):
        out(f"  {green('ok')}    {label}" + (dim("  " + detail) if detail else ""))

    def bad(label, detail=""):
        problems.append(label)
        out(f"  {red('FAIL')}  {label}" + (("  " + detail) if detail else ""))

    def warn(label, detail=""):
        notes.append(label)
        out(f"  {yellow('warn')}  {label}" + (("  " + detail) if detail else ""))

    out()
    out(bold(f"  {platform.system()} {platform.release()}   python "
             f"{platform.python_version()} ({platform.architecture()[0]})"))
    out(dim(f"  {ROOT}"))
    out()

    if sys.version_info < (3, 11):
        bad("python 3.11+", f"this is {platform.python_version()}")
    else:
        ok("python version", platform.python_version())

    vp = venv_python()
    if not vp.is_file():
        bad("the kit's .venv", "run: simplex setup")
    else:
        probe = subprocess.run(
            [str(vp), "-c",
             "import json,sys\n"
             "d={}\n"
             "try:\n"
             "  from exllamav3.version import __version__ as v; d['engine']=v\n"
             "except Exception as e: d['engine']=None\n"
             "for m in ('torch','aiohttp','huggingface_hub','PIL'):\n"
             "  try:\n"
             "    __import__(m); d[m]=True\n"
             "  except Exception: d[m]=False\n"
             "print(json.dumps(d))"],
            capture_output=True, text=True, timeout=120)
        try:
            info = json.loads(probe.stdout.strip().splitlines()[-1])
        except Exception:                                   # noqa: BLE001
            info = {}
        if info.get("engine") == "1.4.4":
            ok("exllamav3", "1.4.4")
        elif info.get("engine"):
            bad("exllamav3 version", f"{info['engine']} (this kit needs 1.4.4)")
        else:
            bad("exllamav3", "not importable in .venv - run: simplex setup")
        for mod in ("torch", "aiohttp", "huggingface_hub"):
            (ok if info.get(mod) else bad)(mod, "" if info.get(mod) else "missing")
        if not info.get("PIL"):
            warn("pillow", "missing - images will be off")

    if shutil.which("nvidia-smi"):
        try:
            import profiles                                 # noqa: WPS433
            g = profiles.detect_gpu()
            if g.total_gib:
                ok("GPU", f"{g.name}  {g.total_gib:.0f} GB  {g.arch}  "
                          f"driver {g.driver}")
            else:
                warn("GPU", "nvidia-smi answered nothing useful")
        except Exception as e:                              # noqa: BLE001
            warn("GPU probe", str(e)[:60])
    else:
        bad("nvidia-smi", "no NVIDIA driver on PATH")

    if shutil.which("node") and shutil.which("npx"):
        ok("node", shutil.which("node"))
    else:
        warn("node", "not installed - the harness cannot run (UI=no skips it)")

    port = env_int(cfg, "PORT", 8888)
    dsh_port = env_int(cfg, "DSH_PORT", 3080)
    for label, p in (("model port", port), ("harness port", dsh_port)):
        if port_listening(p):
            pid, cmdl = port_owner(p)
            mine = "serve_openai" in cmdl or re.search(r"dsh|node", cmdl or "")
            (ok if mine else warn)(f"{label} {p}",
                                   f"in use by pid {pid} {cmdl[:40]}")
        else:
            ok(f"{label} {p}", "free")

    ctx = env_int(cfg, "CONTEXT_SIZE", 0)
    if ctx and ctx % 256:
        bad("CONTEXT_SIZE", f"{ctx} is not a multiple of 256")
    elif ctx:
        ok("CONTEXT_SIZE", f"{ctx:,}")
    cq = (cfg.get("CACHE_QUANT") or "none").strip()
    if not re.fullmatch(r"none|[2-8](,[2-8])?", cq):
        bad("CACHE_QUANT", f"{cq!r} is not none/2-8/k,v")
    else:
        ok("CACHE_QUANT", cq)

    md = cfg.get("MODEL_DIR")
    if md:
        state = downloader.folder_state(ROOT / md)
        if state["state"] == "complete":
            ok("weights", md)
        elif state["state"] == "partial":
            bad("weights", f"{md}: {state['reason']} - run: simplex setup")
        else:
            bad("weights", f"{md} is empty - run: simplex setup")
    else:
        bad("MODEL_DIR", "not set in .env - run: simplex setup")

    try:
        free = shutil.disk_usage(ROOT).free / 1024 ** 3
        (ok if free > 25 else warn)("disk", f"{free:.0f} GB free")
    except OSError:
        pass

    out()
    if problems:
        out(red(f"  {len(problems)} problem(s): ") + ", ".join(problems))
        return 1
    if notes:
        out(yellow(f"  {len(notes)} note(s): ") + ", ".join(notes))
    else:
        out(green("  Everything checks out."))
    out()
    return 0


# ------------------------------------------------------------------ argv ----
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="simplex", description="Run the local Qwen3.8-27B kit.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Every verb takes --help.")
    sub = p.add_subparsers(dest="verb")

    def harness_flags(sp):
        g = sp.add_mutually_exclusive_group()
        g.add_argument("--harness", dest="harness", action="store_const",
                       const="on", help="start the DeepSeek Harness too "
                                        "(overrides UI= in .env)")
        g.add_argument("--no-harness", dest="harness", action="store_const",
                       const="off", help="serve /v1 only")
        sp.set_defaults(harness=None)

    s = sub.add_parser("setup", help="install the environment and fetch a model")
    s.add_argument("rest", nargs=argparse.REMAINDER)
    s.set_defaults(func=cmd_setup)

    s = sub.add_parser("start", help="load a model and serve it")
    harness_flags(s)
    s.add_argument("-b", "--background", action="store_true",
                   help="detach, with the output in logs/")
    s.add_argument("-p", "--port", type=int, help="override PORT for this run")
    s.add_argument("--no-pick", action="store_true",
                   help="do not ask which model; use MODEL_DIR from .env")
    s.add_argument("rest", nargs=argparse.REMAINDER)
    s.set_defaults(func=cmd_start)

    s = sub.add_parser("stop", help="stop the server and the harness")
    s.add_argument("-p", "--port", type=int)
    s.add_argument("--timeout", type=float, default=20.0)
    g = s.add_mutually_exclusive_group()
    g.add_argument("--server-only", action="store_true",
                   help="leave the harness running")
    g.add_argument("--harness-only", action="store_true",
                   help="leave the model loaded")
    s.set_defaults(func=cmd_stop)

    s = sub.add_parser("restart", help="stop, then start again")
    harness_flags(s)
    s.add_argument("-b", "--background", action="store_true")
    s.add_argument("-p", "--port", type=int)
    s.add_argument("--no-pick", action="store_true", default=True)
    s.add_argument("--timeout", type=float, default=20.0)
    s.add_argument("rest", nargs=argparse.REMAINDER)
    s.set_defaults(func=cmd_restart)

    s = sub.add_parser("status", help="what is running, and where")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("logs", help="show or follow the launcher log")
    s.add_argument("-f", "--follow", action="store_true")
    s.add_argument("-n", "--lines", type=int, default=40)
    s.add_argument("--file")
    s.set_defaults(func=cmd_logs)

    s = sub.add_parser("harness", help="the DeepSeek Harness on its own")
    s.add_argument("action", nargs="?", default="status",
                   choices=["start", "stop", "status", "open", "settings"])
    s.add_argument("--no-open", action="store_true",
                   help="do not open a browser at the token URL")
    s.add_argument("--foreground", action="store_true",
                   help="hold the terminal and show the harness's own output")
    s.add_argument("--timeout", type=float, default=20.0)
    s.set_defaults(func=cmd_harness)

    s = sub.add_parser("models", help="what is on the disk")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_models)

    s = sub.add_parser("doctor", help="check this machine and this install")
    s.set_defaults(func=cmd_doctor)
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if not argv:
        parser.print_help()
        out()
        return cmd_status(argparse.Namespace(json=False))
    a = parser.parse_args(argv)
    if not getattr(a, "func", None):
        parser.print_help()
        return 2
    # argparse.REMAINDER keeps a leading "--"; the launchers should not see it
    if getattr(a, "rest", None) and a.rest and a.rest[0] == "--":
        a.rest = a.rest[1:]
    try:
        return a.func(a)
    except KeyboardInterrupt:
        out()
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
