"""Start-menu, desktop and start-at-login entries for Windows.

A kit that is unzipped rather than installed has no way to be launched except
by finding start.bat in Explorer. This module puts "Simplex" where Windows
users look for programs, using nothing but PowerShell, which is always there.

Every function is a no-op that returns an empty result off Windows, so the
launcher can call them unconditionally.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ICON = Path(__file__).resolve().parent / "simplex.ico"
MARKER = ROOT / ".simplex" / "shortcuts.json"
APP_NAME = "Simplex"
TASK_NAME = "Simplex at sign-in"

IS_WINDOWS = sys.platform == "win32"
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _ps(script: str, timeout: int = 60) -> tuple[int, str]:
    """Run a PowerShell script from stdin. Nothing is interpolated into a
    command line, so a path with a quote or an ampersand cannot break out."""
    exe = "powershell.exe" if IS_WINDOWS else "pwsh"
    cmd = [exe, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", "-"]
    try:
        r = subprocess.run(cmd, input=script, capture_output=True, text=True,
                           timeout=timeout, creationflags=CREATE_NO_WINDOW if IS_WINDOWS else 0)
        return r.returncode, ((r.stdout or "") + (r.stderr or "")).strip()
    except FileNotFoundError:
        return 127, "PowerShell was not found"
    except subprocess.TimeoutExpired:
        return 124, "PowerShell did not answer"


def launcher() -> tuple[Path, str]:
    """(target, arguments) for a shortcut: the packaged exe when one was built,
    otherwise start.bat."""
    exe = ROOT / f"{APP_NAME}.exe"
    if exe.is_file():
        return exe, ""
    return ROOT / "start.bat", ""


def desktop_dir() -> Path | None:
    if not IS_WINDOWS:
        return None
    p = Path(os.environ.get("USERPROFILE", "")) / "Desktop"
    return p if p.is_dir() else None


def start_menu_dir() -> Path | None:
    if not IS_WINDOWS:
        return None
    appdata = os.environ.get("APPDATA")
    if not appdata:
        return None
    p = Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs"
    return p if p.is_dir() else None


def create(name: str = APP_NAME, *, desktop: bool = True, start_menu: bool = True,
           description: str = "Chat with a local model on your own GPU") -> list[Path]:
    """Write the .lnk files. Returns the ones that were created."""
    if not IS_WINDOWS:
        return []
    target, args = launcher()
    if not target.is_file():
        return []
    made: list[Path] = []
    places = []
    if start_menu and start_menu_dir():
        places.append(start_menu_dir() / f"{name}.lnk")
    if desktop and desktop_dir():
        places.append(desktop_dir() / f"{name}.lnk")
    places = [p for p in places if not p.exists()]     # never overwrite a user's own
    for link in places:
        script = f"""
$ws = New-Object -ComObject WScript.Shell
$sc = $ws.CreateShortcut({_q(str(link))})
$sc.TargetPath = {_q(str(target))}
$sc.Arguments = {_q(args)}
$sc.WorkingDirectory = {_q(str(ROOT))}
$sc.IconLocation = {_q(str(ICON) + ",0")}
$sc.Description = {_q(description)}
$sc.WindowStyle = 7
$sc.Save()
"""
        code, _out = _ps(script)
        if code == 0 and link.is_file():
            made.append(link)
    if made:
        _remember(made)
    return made


def remove(name: str = APP_NAME) -> list[Path]:
    gone = []
    for d in (start_menu_dir(), desktop_dir()):
        if not d:
            continue
        link = d / f"{name}.lnk"
        if link.is_file():
            try:
                link.unlink()
                gone.append(link)
            except OSError:
                pass
    return gone


def looks_installed() -> bool:
    """True when an installer put this copy here. Its wizard already asked
    about the desktop icon and the Start-menu group, so creating our own on
    top of that overrides a choice the user made and leaves two "Simplex"
    entries in Start-menu search."""
    return any(ROOT.glob("unins*.exe"))


def exists(name: str = APP_NAME) -> bool:
    return any((d / f"{name}.lnk").is_file() for d in (start_menu_dir(), desktop_dir()) if d)


# PowerShell's grammar accepts five code points as a single quote, not one:
# the ASCII apostrophe and four typographic variants. Escaping only U+0027
# leaves the others able to close the literal early - and these strings carry
# the user's account name and install path, so 'O\u2019Brien' is a real name
# that would have injected into a script run with -ExecutionPolicy Bypass.
_PS_QUOTES = "'\u2018\u2019\u201a\u201b"


def _q(s: str) -> str:
    """A PowerShell single-quoted literal, with every quote character doubled."""
    out = []
    for ch in str(s):
        out.append(ch * 2 if ch in _PS_QUOTES else ch)
    return "'" + "".join(out) + "'"


def _remember(links: list[Path]) -> None:
    import json
    MARKER.parent.mkdir(parents=True, exist_ok=True)
    try:
        MARKER.write_text(json.dumps({"links": [str(p) for p in links]}, indent=2),
                          encoding="utf-8")
    except OSError:
        pass


def offered() -> bool:
    """True once shortcuts have been created or the user has been asked."""
    return MARKER.is_file()


def mark_offered(created: bool = False) -> None:
    import json
    MARKER.parent.mkdir(parents=True, exist_ok=True)
    try:
        MARKER.write_text(json.dumps({"created": created}, indent=2), encoding="utf-8")
    except OSError:
        pass


# ------------------------------------------------------------ auto start -----

def autostart(enable: bool, minimized: bool = True) -> tuple[bool, str]:
    """Start Simplex when this user signs in, as a scheduled task.

    A scheduled task and not a service: the server needs a normal desktop
    session to reach the GPU the way a user's own programs do, and a task is
    something the user can see and delete in Task Scheduler without admin
    rights."""
    if not IS_WINDOWS:
        return False, "not Windows"
    if not enable:
        r = subprocess.run(["schtasks", "/Delete", "/TN", TASK_NAME, "/F"],
                           capture_output=True, text=True, creationflags=CREATE_NO_WINDOW)
        return r.returncode == 0, (r.stdout or r.stderr or "").strip()
    target, _args = launcher()
    if not target.is_file():
        return False, "nothing to launch"
    # /RL LIMITED: no elevation, so no UAC prompt at sign-in
    cmd = ["schtasks", "/Create", "/TN", TASK_NAME, "/SC", "ONLOGON", "/RL", "LIMITED",
           "/F", "/TR", f'"{target}"']
    r = subprocess.run(cmd, capture_output=True, text=True, creationflags=CREATE_NO_WINDOW)
    return r.returncode == 0, (r.stdout or r.stderr or "").strip()


def autostart_enabled() -> bool:
    if not IS_WINDOWS:
        return False
    r = subprocess.run(["schtasks", "/Query", "/TN", TASK_NAME],
                       capture_output=True, text=True, creationflags=CREATE_NO_WINDOW)
    return r.returncode == 0


def main(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Windows shortcuts for Simplex")
    ap.add_argument("action", choices=["create", "remove", "status", "autostart-on", "autostart-off"])
    a = ap.parse_args(argv)
    if a.action == "create":
        made = create()
        print("created:" if made else "nothing created (Windows only)")
        for p in made:
            print("  ", p)
    elif a.action == "remove":
        for p in remove():
            print("removed", p)
    elif a.action == "status":
        print("launcher      :", launcher()[0])
        print("start menu    :", start_menu_dir())
        print("desktop       :", desktop_dir())
        print("shortcut here :", exists())
        print("start at login:", autostart_enabled())
    elif a.action == "autostart-on":
        print(autostart(True))
    else:
        print(autostart(False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
