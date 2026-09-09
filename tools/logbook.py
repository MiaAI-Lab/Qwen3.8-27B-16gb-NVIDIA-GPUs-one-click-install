"""Everything the launcher and the server print, also written to a file.

Two reasons. A crash that scrolls off a console window, or a console window
that closes, is unrecoverable for the person it happened to; and the tray menu
needs somewhere to point when it says "View the log". Console output is
unchanged - this only adds a copy.

Old logs are pruned so the folder cannot grow forever.
"""

from __future__ import annotations

import io
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"
KEEP = 10
ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
# Everything a progress bar overwrote. On the console a CR redraws one line; in
# a file it would be hundreds of copies of that line, so only the last survives.
REDRAWN = re.compile(r"[^\n\r]*\r")


def for_file(text: str) -> str:
    """The console keeps colour and redraws; the file keeps neither."""
    return REDRAWN.sub("", ANSI.sub("", text.replace("\r\n", "\n")))


class Tee(io.TextIOBase):
    """Writes to the real stream and to the log file. Colour codes are kept on
    the console and stripped from the file, where they are noise."""

    def __init__(self, stream, sink, lock: threading.Lock):
        self._stream, self._sink, self._lock = stream, sink, lock

    def write(self, text: str) -> int:
        n = 0
        try:
            n = self._stream.write(text)
            self._stream.flush()
        except Exception:                     # noqa: BLE001 - a closed console must not stop logging
            pass
        clean = for_file(text)
        if clean:
            with self._lock:
                try:
                    self._sink.write(clean)
                    self._sink.flush()
                except Exception:             # noqa: BLE001
                    pass
        return n or len(text)

    def flush(self) -> None:
        for s in (self._stream, self._sink):
            try:
                s.flush()
            except Exception:                 # noqa: BLE001
                pass

    def isatty(self) -> bool:
        try:
            return bool(self._stream.isatty())
        except Exception:                     # noqa: BLE001
            return False

    @property
    def encoding(self) -> str:
        return getattr(self._stream, "encoding", "utf-8") or "utf-8"


_ACTIVE: "Logbook | None" = None    # only one Logbook may own the streams


class Logbook:
    def __init__(self, directory: Path = LOG_DIR, keep: int = KEEP):
        self.dir = Path(directory)
        self.keep = keep
        self.path: Path | None = None
        self._file = None
        self._lock = threading.Lock()
        self._saved = None

    def start(self) -> Path | None:
        """Begin capturing. Returns the log path, or None if it could not be
        opened (a read-only folder is a reason to run without a log, not to
        refuse to start).

        Wrapping twice would put a Tee around a Tee: stop() then restores the
        inner Tee instead of the real stream and leaks the first file handle,
        which on Windows keeps that log locked forever. The guard is global,
        not per-instance, because the second caller is usually a different
        object entirely."""
        global _ACTIVE
        if self._saved is not None:
            return self.path
        if _ACTIVE is not None and _ACTIVE is not self:
            return _ACTIVE.path
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            self.path = self.dir / f"simplex-{stamp}.log"
            self._file = self.path.open("w", encoding="utf-8", errors="replace")
        except OSError:
            self.path, self._file = None, None
            return None
        self._prune()
        self._saved = (sys.stdout, sys.stderr)
        sys.stdout = Tee(self._saved[0], self._file, self._lock)
        sys.stderr = Tee(self._saved[1], self._file, self._lock)
        _ACTIVE = self
        self.write_raw(f"# Simplex log - {datetime.now():%Y-%m-%d %H:%M:%S}\n"
                       f"# {sys.executable}\n# {' '.join(sys.argv)}\n\n")
        return self.path

    def write_raw(self, text: str) -> None:
        """Into the file only - used for the child process's output, which has
        already been echoed to the console by whoever read it."""
        if not self._file:
            return
        with self._lock:
            try:
                self._file.write(for_file(text))
                self._file.flush()
            except Exception:                 # noqa: BLE001
                pass

    def stop(self) -> None:
        global _ACTIVE
        if _ACTIVE is self:
            _ACTIVE = None
        if self._saved:
            sys.stdout, sys.stderr = self._saved
            self._saved = None
        if self._file:
            try:
                self._file.close()
            except Exception:                 # noqa: BLE001
                pass
            self._file = None

    def tail(self, lines: int = 40) -> str:
        if not self.path or not self.path.is_file():
            return ""
        try:
            text = self.path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return "\n".join(text.splitlines()[-lines:])

    def _prune(self) -> None:
        try:
            old = sorted(self.dir.glob("simplex-*.log"),
                         key=lambda p: p.stat().st_mtime, reverse=True)
        except OSError:
            return
        for p in old[self.keep:]:
            try:
                p.unlink(missing_ok=True)
            except OSError:
                # a log another process still has open; skip it and carry on,
                # rather than abandoning the rest of the sweep
                continue


def latest(directory: Path = LOG_DIR) -> Path | None:
    try:
        logs = sorted(Path(directory).glob("simplex-*.log"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        return logs[0] if logs else None
    except OSError:
        return None


def explain(exc: BaseException) -> tuple[str, str]:
    """(what happened, what to do) for an exception that reached the top.

    A stack trace is the right thing for a bug report and the wrong thing for
    someone who just double-clicked an icon; the trace still goes to the log."""
    name = type(exc).__name__
    text = str(exc)
    low = text.lower()
    if isinstance(exc, MemoryError):
        return ("Simplex ran out of memory.",
                "Close other programs and try again. If it keeps happening, pick a "
                "smaller model size with `windows\\start.bat profile`.")
    if isinstance(exc, PermissionError):
        return (f"Windows refused access to a file ({text}).",
                "Antivirus or OneDrive is usually holding it. Move the Simplex folder "
                "somewhere local, such as C:\\Simplex, and try again.")
    if isinstance(exc, FileNotFoundError):
        return (f"A file Simplex needs is missing ({text}).",
                "If files were moved or deleted, unzip the kit again next to your "
                ".env and models folders.")
    if isinstance(exc, OSError) and ("10048" in text or "address" in low and "use" in low):
        return ("The port Simplex uses is already taken.",
                "Another copy is probably running. Close it, or set a different PORT in .env.")
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return ("Simplex could not reach the internet.",
                "Check the connection, then start it again. A part-finished download "
                "picks up where it stopped.")
    return (f"Simplex stopped because of an unexpected error ({name}: {text}).", "")
