"""A Windows tray icon for the running server - ctypes only, no dependencies.

The point is that Simplex stops being "that console window". The icon sits by
the clock, the menu opens the chat, restarts the model, shows the log, and
quits cleanly, and a balloon tells the user when the model has finished
loading. Everything else on this machine works that way; Simplex should too.

Nothing here is required: on a non-Windows host, or if any Win32 call fails,
`Tray.start()` reports False and the launcher carries on exactly as before.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ICON = Path(__file__).resolve().parent / "simplex.ico"

IS_WINDOWS = sys.platform == "win32"

# --- Win32 constants ---------------------------------------------------------
WM_DESTROY = 0x0002
WM_CLOSE = 0x0010
WM_COMMAND = 0x0111
WM_APP = 0x8000
WM_TRAY = WM_APP + 17
WM_LBUTTONUP = 0x0202
WM_LBUTTONDBLCLK = 0x0203
WM_RBUTTONUP = 0x0205

NIM_ADD, NIM_MODIFY, NIM_DELETE = 0, 1, 2
NIF_MESSAGE, NIF_ICON, NIF_TIP, NIF_INFO = 0x01, 0x02, 0x04, 0x10
NIIF_INFO = 0x01

IMAGE_ICON = 1
LR_LOADFROMFILE, LR_DEFAULTSIZE, LR_SHARED = 0x0010, 0x0040, 0x8000

MF_STRING, MF_SEPARATOR, MF_GRAYED = 0x0000, 0x0800, 0x0001
TPM_RIGHTBUTTON, TPM_RETURNCMD, TPM_NONOTIFY = 0x0002, 0x0100, 0x0080

CW_USEDEFAULT = -2147483648
IDI_APPLICATION = 32512


class Tray:
    """One tray icon and its menu. `items` is a list of (label, callback) or
    (None, None) for a separator; the first item is also the double-click
    action, which is what people expect from a tray icon."""

    def __init__(self, title: str, tooltip: str, items, icon: Path = ICON):
        self.title, self.tooltip, self.items = title, tooltip, list(items)
        self.icon_path = Path(icon)
        self._thread: threading.Thread | None = None
        self._hwnd = None
        self._hicon = None
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._ok = False
        self._nid = None
        self._nid_lock = threading.Lock()
        self._gone = threading.Event()
        self._taskbar_created = 0

    # ------------------------------------------------------------- public --
    @staticmethod
    def available() -> bool:
        return IS_WINDOWS

    def start(self, timeout: float = 5.0) -> bool:
        """Returns True once the icon is actually in the tray."""
        if not IS_WINDOWS:
            return False
        self._thread = threading.Thread(target=self._run, name="tray", daemon=True)
        self._thread.start()
        self._ready.wait(timeout)
        return self._ok

    BASE_FLAGS = NIF_MESSAGE | NIF_ICON | NIF_TIP

    def notify(self, title: str, text: str) -> None:
        if not self._ok:
            return
        # One lock for every writer of self._nid. The struct is also read by the
        # message-loop thread when it re-adds the icon after an Explorer restart,
        # and leaving uFlags parked at NIF_INFO between two calls could register
        # an icon with no image and no click handler.
        try:
            import ctypes
            with self._nid_lock:
                nid = self._nid
                nid.szInfoTitle = title[:63]
                nid.szInfo = text[:255]
                nid.dwInfoFlags = NIIF_INFO
                nid.uFlags = self.BASE_FLAGS | NIF_INFO
                ctypes.windll.shell32.Shell_NotifyIconW(NIM_MODIFY, ctypes.byref(nid))
                nid.uFlags = self.BASE_FLAGS
                nid.szInfo = ""
        except Exception:                    # noqa: BLE001 - a balloon is never worth a crash
            pass

    def set_tip(self, text: str) -> None:
        if not self._ok:
            return
        try:
            import ctypes
            with self._nid_lock:
                self.tooltip = text
                self._nid.szTip = text[:127]
                self._nid.uFlags = self.BASE_FLAGS
                ctypes.windll.shell32.Shell_NotifyIconW(NIM_MODIFY, ctypes.byref(self._nid))
        except Exception:                    # noqa: BLE001
            pass

    def stop(self, timeout: float = 3.0) -> None:
        """Ask the icon to remove itself and wait for it to happen. Without the
        wait the process exits first and Windows leaves a dead icon by the clock
        until someone hovers over it."""
        self._stop.set()
        if not self._ok or self._hwnd is None:
            return
        try:
            import ctypes
            ctypes.windll.user32.PostMessageW(self._hwnd, WM_CLOSE, 0, 0)
        except Exception:                    # noqa: BLE001
            return
        self._gone.wait(timeout)
        if self._thread is not None:
            self._thread.join(timeout=max(0.2, timeout / 2))

    # -------------------------------------------------------------- inner --
    def _run(self) -> None:
        try:
            self._pump()
        except Exception:                    # noqa: BLE001 - never take the launcher down
            self._ok = False
        finally:
            self._ready.set()

    def _pump(self) -> None:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        shell32 = ctypes.windll.shell32
        kernel32 = ctypes.windll.kernel32

        LRESULT = ctypes.c_ssize_t
        WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wintypes.HWND, wintypes.UINT,
                                     wintypes.WPARAM, wintypes.LPARAM)

        class NOTIFYICONDATAW(ctypes.Structure):
            _fields_ = [
                ("cbSize", wintypes.DWORD),
                ("hWnd", wintypes.HWND),
                ("uID", wintypes.UINT),
                ("uFlags", wintypes.UINT),
                ("uCallbackMessage", wintypes.UINT),
                ("hIcon", wintypes.HICON),
                ("szTip", wintypes.WCHAR * 128),
                ("dwState", wintypes.DWORD),
                ("dwStateMask", wintypes.DWORD),
                ("szInfo", wintypes.WCHAR * 256),
                ("uVersion", wintypes.UINT),
                ("szInfoTitle", wintypes.WCHAR * 64),
                ("dwInfoFlags", wintypes.DWORD),
            ]

        class WNDCLASSW(ctypes.Structure):
            _fields_ = [
                ("style", wintypes.UINT),
                ("lpfnWndProc", WNDPROC),
                ("cbClsExtra", ctypes.c_int),
                ("cbWndExtra", ctypes.c_int),
                ("hInstance", wintypes.HINSTANCE),
                ("hIcon", wintypes.HICON),
                ("hCursor", wintypes.HANDLE),
                ("hbrBackground", wintypes.HBRUSH),
                ("lpszMenuName", wintypes.LPCWSTR),
                ("lpszClassName", wintypes.LPCWSTR),
            ]

        user32.DefWindowProcW.restype = LRESULT
        user32.DefWindowProcW.argtypes = [wintypes.HWND, wintypes.UINT,
                                          wintypes.WPARAM, wintypes.LPARAM]
        user32.CreateWindowExW.restype = wintypes.HWND
        user32.LoadImageW.restype = wintypes.HICON
        user32.TrackPopupMenu.restype = ctypes.c_int
        user32.CreatePopupMenu.restype = wintypes.HMENU
        # Handles are pointers. Without a restype ctypes returns c_int, which on
        # 64-bit Windows truncates the module base before it is stored in
        # WNDCLASSW.hInstance - it happens to work only because Register and
        # Create then agree on the same wrong value.
        kernel32.GetModuleHandleW.restype = wintypes.HMODULE
        kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]

        self._taskbar_created = user32.RegisterWindowMessageW("TaskbarCreated")

        # -- icon ----------------------------------------------------------
        hicon = None
        if self.icon_path.is_file():
            hicon = user32.LoadImageW(None, str(self.icon_path), IMAGE_ICON, 0, 0,
                                      LR_LOADFROMFILE | LR_DEFAULTSIZE)
        if not hicon:
            # MAKEINTRESOURCE: the id travels in the pointer, so it has to be
            # cast rather than passed as an int on 64-bit Windows
            user32.LoadIconW.restype = wintypes.HICON
            user32.LoadIconW.argtypes = [wintypes.HINSTANCE, wintypes.LPCWSTR]
            hicon = user32.LoadIconW(
                None, ctypes.cast(ctypes.c_void_p(IDI_APPLICATION), wintypes.LPCWSTR))
        self._hicon = hicon

        # -- window --------------------------------------------------------
        def wndproc(hwnd, msg, wparam, lparam):
            try:
                if msg == WM_TRAY:
                    low = lparam & 0xFFFF
                    if low in (WM_LBUTTONUP, WM_LBUTTONDBLCLK):
                        self._invoke(0)
                    elif low == WM_RBUTTONUP:
                        self._menu(hwnd, user32)
                    return 0
                if msg == WM_COMMAND:
                    self._invoke((wparam & 0xFFFF) - 1)
                    return 0
                if msg == self._taskbar_created:
                    with self._nid_lock:
                        self._nid.uFlags = self.BASE_FLAGS
                        shell32.Shell_NotifyIconW(NIM_ADD, ctypes.byref(self._nid))
                    return 0
                if msg in (WM_CLOSE, WM_DESTROY):
                    with self._nid_lock:
                        shell32.Shell_NotifyIconW(NIM_DELETE, ctypes.byref(self._nid))
                    self._gone.set()
                    user32.PostQuitMessage(0)
                    return 0
            except Exception:                # noqa: BLE001
                pass
            return user32.DefWindowProcW(hwnd, msg, wparam, lparam)

        self._proc = WNDPROC(wndproc)        # keep a reference: Windows calls it later
        cls = WNDCLASSW()
        cls.lpfnWndProc = self._proc
        cls.hInstance = kernel32.GetModuleHandleW(None)
        cls.lpszClassName = "SimplexTrayWindow"
        ERROR_CLASS_ALREADY_EXISTS = 1410
        if not user32.RegisterClassW(ctypes.byref(cls)):
            err = ctypes.get_last_error() if hasattr(ctypes, "get_last_error") else 0
            if err not in (0, ERROR_CLASS_ALREADY_EXISTS):
                raise OSError(f"could not register the tray window class (error {err})")
        hwnd = user32.CreateWindowExW(0, "SimplexTrayWindow", self.title, 0,
                                      CW_USEDEFAULT, CW_USEDEFAULT, 0, 0,
                                      None, None, cls.hInstance, None)
        if not hwnd:
            raise OSError("could not create the tray window")
        self._hwnd = hwnd

        nid = NOTIFYICONDATAW()
        nid.cbSize = ctypes.sizeof(NOTIFYICONDATAW)
        nid.hWnd = hwnd
        nid.uID = 1
        nid.uFlags = NIF_MESSAGE | NIF_ICON | NIF_TIP
        nid.uCallbackMessage = WM_TRAY
        nid.hIcon = hicon
        nid.szTip = self.tooltip[:127]
        self._nid = nid
        if not shell32.Shell_NotifyIconW(NIM_ADD, ctypes.byref(nid)):
            raise OSError("could not add the tray icon")

        self._ok = True
        self._ready.set()

        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))
        self._ok = False

    def _menu(self, hwnd, user32) -> None:
        import ctypes
        from ctypes import wintypes
        menu = user32.CreatePopupMenu()
        for i, (label, cb) in enumerate(self.items):
            if label is None:
                user32.AppendMenuW(menu, MF_SEPARATOR, 0, None)
            else:
                flags = MF_STRING | (0 if cb else MF_GRAYED)
                user32.AppendMenuW(menu, flags, i + 1, label)
        pt = wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(pt))
        # documented dance: without this the menu will not close on click-away
        user32.SetForegroundWindow(hwnd)
        cmd = user32.TrackPopupMenu(menu, TPM_RIGHTBUTTON | TPM_RETURNCMD | TPM_NONOTIFY,
                                    pt.x, pt.y, 0, hwnd, None)
        user32.PostMessageW(hwnd, 0, 0, 0)
        user32.DestroyMenu(menu)
        if cmd:
            self._invoke(cmd - 1)

    def _invoke(self, index: int) -> None:
        if not (0 <= index < len(self.items)):
            return
        _label, cb = self.items[index]
        if not cb:
            return
        threading.Thread(target=self._safe, args=(cb,), daemon=True).start()

    @staticmethod
    def _safe(cb) -> None:
        try:
            cb()
        except Exception:                    # noqa: BLE001 - a menu click must not crash the tray
            pass


def open_path(path: Path) -> None:
    """Show a file or folder in Explorer (or the platform equivalent)."""
    import subprocess
    path = Path(path)
    try:
        if IS_WINDOWS:
            if path.is_dir():
                import os
                os.startfile(str(path))                      # noqa: S606
            else:
                subprocess.Popen(["explorer", "/select,", str(path)])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-R", str(path)] if path.is_file() else ["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path if path.is_dir() else path.parent)])
    except Exception:                        # noqa: BLE001
        pass


def demo() -> int:
    """`python tools/tray.py` - a standalone icon, for checking it looks right."""
    import time
    done = threading.Event()
    tray = Tray("Simplex", "Simplex - demo",
                [("Open Simplex", lambda: print("open")),
                 ("Restart", lambda: print("restart")),
                 (None, None),
                 ("Quit", done.set)])
    if not tray.start():
        print("The tray needs Windows. Nothing to show here.")
        return 1
    tray.notify("Simplex is ready", "The model is loaded. Click to open the chat.")
    while not done.wait(0.5):
        pass
    tray.stop()
    time.sleep(0.3)
    return 0


if __name__ == "__main__":
    raise SystemExit(demo())
