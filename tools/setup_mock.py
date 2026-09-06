"""Fake machine for exercising the setup page without a GPU or a download.

    python tools/setup_mock.py --port 8899

Used by tools/test_webui.py and for taking screenshots. It patches exactly
three things - the GPU probe, the installer and the downloader - so everything
else on the page is the real code path.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import profiles                                  # noqa: E402
import setup_core as core                        # noqa: E402


class FakeGPU(profiles.GPU):
    pass


def install(cfg, log, steps, cancelled=None, speed: float = 1.0):
    plan = [
        ("venv", ["created .venv", "installed setuptools"], 1.2),
        ("pipbase", ["Collecting pip", "Successfully installed pip-25.2 wheel-0.45.1"], 1.6),
        ("torch", ["Collecting torch", "Downloading torch-2.9.0+cu130-cp313-win_amd64.whl (2.4 GB)",
                   "Successfully installed torch-2.9.0+cu130"], 4.0),
        ("triton", ["Collecting triton-windows", "Successfully installed triton-windows-3.4.0"], 1.5),
        ("engine", ["# trying the prebuilt engine: exllamav3-1.4.4-cp313-cp313-win_amd64.whl",
                    "Successfully installed exllamav3-1.4.4"], 3.0),
        ("server", ["Successfully installed aiohttp-3.12.0 huggingface_hub-0.35.0 pillow-11.3.0"], 1.4),
        ("check", ["engine 1.4.4", "cuda available: True"], 1.0),
    ]
    known = {s.id for s in steps}
    for sid, lines, secs in plan:
        if sid not in known:
            continue
        steps.start(sid)
        for line in lines:
            if cancelled and cancelled():
                return
            log(line, "out")
            time.sleep(secs / max(1, len(lines)) * speed)
        if cancelled and cancelled():
            return
        steps.finish(sid)


class FakeDownload:
    def __init__(self, total=9_700_000_000, seconds=6.0, on_progress=None):
        self.total, self.seconds = total, seconds
        self._on = on_progress
        self._cancel = threading.Event()
        self.progress = core.__dict__.get("_", None)
        import downloader
        self.progress = downloader.Progress(total_bytes=total, files_total=4, state="listing")

    def cancel(self):
        self._cancel.set()

    def run(self):
        p = self.progress
        self._emit()
        time.sleep(0.4 if self.seconds < 3 else 1.0)
        p.state = "downloading"
        files = ["config.json", "model-00001-of-00003.safetensors",
                 "model-00002-of-00003.safetensors", "tokenizer.json"]
        steps = 60
        for i in range(steps + 1):
            if self._cancel.is_set():
                p.state = "cancelled"
                self._emit()
                return
            p.done_bytes = int(self.total * i / steps)
            p.files_done = min(len(files) - 1, i * len(files) // steps)
            p.current = files[p.files_done]
            p.speed_bps = self.total / self.seconds
            self._emit()
            time.sleep(self.seconds / steps)
        p.state, p.current, p.message = "done", "", "Weights downloaded"
        p.files_done = len(files)
        self._emit()

    def _emit(self):
        if self._on:
            self._on(self.progress.as_dict())


def patch(vram_gib: float = 16.0, name: str = "NVIDIA GeForce RTX 4060 Ti",
          cc: float = 8.9, driver: str = "581.29", speed: float = 1.0,
          download_seconds: float = 6.0, fail: str = "", env_file=None):
    """Swap the expensive pieces for fakes. Returns the scratch .env path.

    That includes .env itself. A mocked run still goes through the real
    apply_choice(), which writes core.ENV_FILE - so with a pretend 16 GB card
    patched in, both the test suite and a manual `python tools/setup_mock.py`
    rewrote the kit's actual .env with a profile for a card that is not there,
    capping a 32 GB machine at 14.7 GB. Anything faking the hardware has to fake
    the file it would write, so pass env_file to choose the path or take the
    temporary one returned here."""
    import tempfile
    core.ENV_FILE = Path(env_file) if env_file else (
        Path(tempfile.mkdtemp(prefix="simplex-mock-")) / ".env")
    profiles.detect_gpu = lambda cfg=None: profiles.GPU(name, vram_gib, cc, driver)

    def fake_install(cfg, log, steps, cancelled=None):
        if fail == "install":
            steps.start("venv")
            log("Collecting exllamav3", "out")
            log("error: Microsoft Visual C++ 14.0 or greater is required", "out")
            steps.fail("venv", "the compiler is missing")
            raise core.SetupError(
                "The engine has to be compiled here, and the tools for it are missing: "
                "Visual Studio Build Tools with the \"Desktop development with C++\" workload.",
                "Easiest fix: drop a prebuilt exllamav3 wheel for cp313 into the kit's "
                "wheels\\ folder and start Simplex again - no compiler needed.")
        install(cfg, log, steps, cancelled, speed)

    core.install_environment = fake_install

    def fake_download(cfg, on_progress=None, cancelled=None, token=""):
        dl = FakeDownload(seconds=download_seconds, on_progress=on_progress)
        if cancelled is not None:
            def watch():
                while dl.progress.state in ("idle", "listing", "downloading"):
                    if cancelled():
                        dl.cancel()
                        return
                    time.sleep(0.2)
            threading.Thread(target=watch, daemon=True).start()
        return dl

    core.download_weights = fake_download
    return core.ENV_FILE
    core.weights_ready = lambda cfg: (False, ROOT / (cfg.get("MODEL_DIR") or "models"))
    core.venv_ready = lambda: (False, "no virtual environment yet")


def main(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Setup page against a fake machine")
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--vram", type=float, default=16.0)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--download-seconds", type=float, default=6.0)
    ap.add_argument("--fail", default="", choices=["", "install"])
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args(argv)

    patch(vram_gib=a.vram, speed=a.speed, download_seconds=a.download_seconds, fail=a.fail)
    import setup_web
    cfg = {"MODEL_DIR": "models/Qwen3.8-27B-EXL3-4.0bpw", "PORT": str(a.port)}
    setup, httpd, url = setup_web.serve(cfg, a.port, "127.0.0.1",
                                        console=lambda l, k: print("   ", l, flush=True))
    print(f"\n  Mock setup page: {url}\n")
    if not a.no_browser:
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception:                     # noqa: BLE001
            pass
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    httpd.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
