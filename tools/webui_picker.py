#!/usr/bin/env python3
"""The system folder picker, for choosing the agent's workspace.

The kit's own folder browser still exists (it is the only option when the UI
is open from another device, since a dialog would appear on the *server's*
screen), but when the browser and the server are the same machine the real
picker is what people expect.

Windows uses PowerShell + WinForms `FolderBrowserDialog`; everything else
uses tkinter's `askdirectory`. Both run in a short-lived child process, so a
dialog that is ignored or crashes can never take the model server with it,
and the start path travels in the environment rather than inside a quoted
command line.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading

DEFAULT_TIMEOUT = 300           # a dialog nobody answers is not a hung server
_lock = threading.Lock()        # one dialog at a time, whoever asks

# -STA is required for shell dialogs; a topmost owner form keeps the picker in
# front of the browser window instead of behind it.
POWERSHELL = r"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.FolderBrowserDialog
$dialog.Description = 'Choose the folder the agent may work in'
$dialog.ShowNewFolderButton = $true
$start = $env:WEBUI_PICK_START
if ($start -and (Test-Path -LiteralPath $start)) { $dialog.SelectedPath = $start }
$owner = New-Object System.Windows.Forms.Form
$owner.TopMost = $true
$owner.ShowInTaskbar = $false
$owner.Opacity = 0
$owner.Show()
$result = $dialog.ShowDialog($owner)
$owner.Close()
if ($result -eq [System.Windows.Forms.DialogResult]::OK) {
  [Console]::Out.Write($dialog.SelectedPath)
}
"""

TK = (
    "import os, tkinter as tk\n"
    "from tkinter import filedialog\n"
    "root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True)\n"
    "start = os.environ.get('WEBUI_PICK_START') or None\n"
    "path = filedialog.askdirectory(title='Choose the agent workspace',\n"
    "                               initialdir=start, mustexist=True)\n"
    "print(path or '', end='')\n"
)


class PickerUnavailable(Exception):
    """No system dialog on this machine - the caller falls back to the kit's."""


def _powershell() -> str | None:
    return shutil.which("powershell") or shutil.which("pwsh")


def available() -> bool:
    """True when a dialog can actually be shown from this process."""
    if sys.platform == "win32":
        return _powershell() is not None
    if not os.environ.get("DISPLAY") and sys.platform != "darwin":
        return False                                  # headless Linux
    try:
        import tkinter                                # noqa: F401,WPS433
    except ImportError:
        return False
    return True


def _run(cmd, start, timeout, stdin=None):
    env = dict(os.environ)
    if start:
        env["WEBUI_PICK_START"] = str(start)
    try:
        proc = subprocess.run(cmd, input=stdin, env=env, timeout=timeout,
                              capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        raise PickerUnavailable(
            f"the folder dialog was still open after {timeout}s") from None
    except OSError as e:
        raise PickerUnavailable(str(e)) from e
    if proc.returncode != 0 and not proc.stdout.strip():
        detail = (proc.stderr or "").strip().splitlines()
        raise PickerUnavailable(detail[-1] if detail else
                                f"the dialog exited with code {proc.returncode}")
    return proc.stdout.strip()


def pick(start=None, timeout=DEFAULT_TIMEOUT) -> str | None:
    """Show the dialog and return the chosen folder, or None if cancelled.

    Raises PickerUnavailable when no dialog could be shown at all.
    """
    if not _lock.acquire(blocking=False):
        raise PickerUnavailable("a folder dialog is already open")
    try:
        if sys.platform == "win32":
            shell = _powershell()
            if not shell:
                raise PickerUnavailable("PowerShell was not found")
            out = _run([shell, "-NoProfile", "-NonInteractive", "-STA",
                        "-WindowStyle", "Hidden", "-Command", "-"],
                       start, timeout, stdin=POWERSHELL)
        else:
            out = _run([sys.executable, "-c", TK], start, timeout)
    finally:
        _lock.release()
    return out or None


if __name__ == "__main__":
    print("available:", available())
    if available():
        print("picked:", pick(sys.argv[1] if len(sys.argv) > 1 else None))
