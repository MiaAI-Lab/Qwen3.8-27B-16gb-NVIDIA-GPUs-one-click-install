"""First-run work, with no console attached to it.

Everything setup does - probing the machine, listing the quants that fit,
writing .env, building the virtualenv, installing the engine, fetching the
weights - lives here as plain functions that report through callbacks. The
console launcher and the setup web page are two front ends over the same
pipeline, so there is one definition of what "first run" means.

Nothing in this module prints, exits, or blocks on input.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

VENV_DIR = ROOT / ".venv"
VENV_PY = VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
ENV_FILE = ROOT / ".env"
ENV_EXAMPLE = ROOT / ".env.example"

ENGINE_PACKAGE = "exllamav3"
ENGINE_VERSION = "1.4.4"


class SetupError(RuntimeError):
    def __init__(self, message: str, hint: str = ""):
        super().__init__(message)
        self.hint = hint

    def as_dict(self) -> dict:
        return {"message": str(self), "hint": self.hint}


# ------------------------------------------------------------------ steps ----

@dataclass
class Step:
    id: str
    title: str
    detail: str = ""
    state: str = "pending"      # pending | running | ok | failed | skipped
    seconds: float = 0.0
    error: str = ""
    hint: str = ""

    def as_dict(self) -> dict:
        return {"id": self.id, "title": self.title, "detail": self.detail,
                "state": self.state, "seconds": round(self.seconds, 1),
                "error": self.error, "hint": self.hint}


class Steps:
    """Ordered, observable. `on_change` fires with the whole list, because the
    page redraws the checklist as a unit anyway."""

    def __init__(self, items: list[Step], on_change=None):
        self.items = items
        self._by_id = {s.id: s for s in items}
        self._on_change = on_change
        self._t0: dict[str, float] = {}

    def __iter__(self):
        return iter(self.items)

    def get(self, sid: str) -> Step:
        return self._by_id[sid]

    def as_list(self) -> list[dict]:
        return [s.as_dict() for s in self.items]

    def _changed(self) -> None:
        if self._on_change:
            try:
                self._on_change(self.as_list())
            except Exception:                # noqa: BLE001
                pass

    def start(self, sid: str, detail: str = "") -> None:
        s = self.get(sid)
        s.state, s.detail = "running", detail or s.detail
        self._t0[sid] = time.time()
        self._changed()

    def finish(self, sid: str, detail: str = "") -> None:
        s = self.get(sid)
        s.state = "ok"
        s.seconds = time.time() - self._t0.get(sid, time.time())
        if detail:
            s.detail = detail
        self._changed()

    def skip(self, sid: str, detail: str = "") -> None:
        s = self.get(sid)
        s.state, s.detail = "skipped", detail or s.detail
        self._changed()

    def fail(self, sid: str, error: str, hint: str = "") -> None:
        s = self.get(sid)
        s.state, s.error, s.hint = "failed", error, hint
        s.seconds = time.time() - self._t0.get(sid, time.time())
        self._changed()

    def detail(self, sid: str, detail: str) -> None:
        self.get(sid).detail = detail
        self._changed()


def install_steps() -> list[Step]:
    win = sys.platform == "win32"
    items = [
        Step("venv", "Python environment", "an isolated .venv folder, nothing installed system-wide"),
        Step("pipbase", "Build tools", "pip, setuptools, wheel"),
        Step("torch", "PyTorch", "about 2-3 GB the first time"),
    ]
    if win:
        items.append(Step("triton", "Triton for Windows", "GPU kernels the engine loads at import"))
    items += [
        Step("engine", "ExLlamaV3 engine", "prebuilt wheel if one fits this Python"),
        Step("server", "Server libraries", "aiohttp, huggingface_hub, pillow"),
        Step("check", "Checking the install", "imports the engine and asks the GPU to say hello"),
    ]
    return items


# ------------------------------------------------------------------ probe ----

def _disk_free_gb(path: Path) -> float:
    try:
        return shutil.disk_usage(str(path)).free / (1024 ** 3)
    except Exception:                        # noqa: BLE001
        return 0.0


def _runnable(cmd: list[str]) -> bool:
    try:
        return subprocess.run(cmd + ["-c", "pass"], capture_output=True,
                              timeout=30).returncode == 0
    except Exception:                        # noqa: BLE001
        return False


def _system_python() -> list[str] | None:
    r"""An interpreter that can create a virtualenv.

    Two traps. The launcher puts .venv\Scripts on PATH, so `which python` can
    hand back the very environment we are trying to rebuild; and an entry on
    PATH may not run at all (a moved installation, a Store alias stub). Both
    are checked rather than assumed."""
    candidates: list[list[str]] = []
    py = shutil.which("py")
    if py and sys.platform == "win32":
        candidates.append([py, "-3"])
    for name in ("python3", "python"):
        w = shutil.which(name)
        if w and VENV_DIR not in Path(w).resolve().parents:
            candidates.append([w])
    if sys.executable and VENV_DIR not in Path(sys.executable).resolve().parents:
        candidates.append([sys.executable])
    for cmd in candidates:
        if _runnable(cmd):
            return cmd
    return None


def venv_ready() -> tuple[bool, str]:
    """(ok, reason). Reason is empty when ok, else something a person can act on."""
    if not VENV_PY.is_file():
        return False, "no virtual environment yet"
    probe = (
        "import json, sys\n"
        "out = {}\n"
        "try:\n"
        "    import torch; out['torch'] = torch.__version__; out['cuda'] = bool(torch.cuda.is_available())\n"
        "except Exception as e: out['torch_error'] = str(e)[:300]\n"
        "try:\n"
        "    from exllamav3.version import __version__ as v; out['engine'] = v\n"
        "except Exception as e: out['engine_error'] = str(e)[:300]\n"
        "for mod in ('aiohttp', 'huggingface_hub', 'PIL'):\n"
        "    try:\n"
        "        __import__(mod); out[mod] = True\n"
        "    except Exception: out[mod] = False\n"
        "print(json.dumps(out))\n"
    )
    r = subprocess.run([str(VENV_PY), "-c", probe], capture_output=True, text=True)
    try:
        out = json.loads((r.stdout or "").strip().splitlines()[-1])
    except Exception:                        # noqa: BLE001
        return False, "the environment could not be inspected"
    if "torch" not in out:
        return False, "PyTorch is not installed"
    if not out.get("cuda"):
        return False, "PyTorch cannot see a CUDA GPU (driver missing, or a CPU-only PyTorch)"
    if "engine" not in out:
        return False, "the ExLlamaV3 engine is not installed"
    if out.get("engine") != ENGINE_VERSION:
        return False, f"the engine is v{out['engine']}, this kit needs v{ENGINE_VERSION}"
    for mod, label in (("aiohttp", "aiohttp"), ("huggingface_hub", "huggingface_hub"), ("PIL", "pillow")):
        if not out.get(mod):
            return False, f"{label} is missing"
    return True, ""


def weights_ready(cfg: dict) -> tuple[bool, Path]:
    import downloader
    raw = cfg.get("MODEL_DIR") or ""
    if not raw:
        return False, ROOT / "models"
    p = Path(raw)
    if not p.is_absolute():
        p = ROOT / p
    return downloader.is_complete(p), p


def probe(cfg: dict) -> dict:
    """One snapshot of the machine, for the first screen of setup."""
    import profiles
    import wheels

    gpu = profiles.detect_gpu()
    support, notes = profiles.gpu_support(gpu)
    ok_venv, venv_reason = venv_ready()
    have_weights, model_path = weights_ready(cfg)
    tags = wheels.interpreter_tags(VENV_PY if VENV_PY.is_file() else sys.executable)
    engine_wheel = wheels.describe(ENGINE_PACKAGE, tags, cfg)

    return {
        "gpu": {"name": gpu.name, "vram_gib": round(gpu.total_gib, 1), "cc": gpu.cc,
                "arch": gpu.arch, "driver": gpu.driver, "support": support, "notes": notes},
        "python": {"version": ".".join(str(x) for x in sys.version_info[:3]),
                   "found": _system_python() is not None, "tag": tags.get("py", "")},
        "disk_free_gb": round(_disk_free_gb(ROOT), 1),
        "venv": {"ready": ok_venv, "reason": venv_reason, "exists": VENV_PY.is_file()},
        "weights": {"ready": have_weights, "path": str(model_path),
                    "model": cfg.get("MODEL_ID") or ""},
        "engine_wheel": engine_wheel,
        "source_build_blockers": [] if engine_wheel else wheels.source_build_blockers(),
        "profile": cfg.get("PROFILE") or "",
        "platform": sys.platform,
    }


def options_for(cfg: dict, vram_gib: float = 0.0, want_vision=None) -> dict:
    """The quant menu the setup page renders. Same numbers as the console menu.

    want_vision mirrors the console's images question (True / False / None for
    the automatic rule). Without it the browser flow silently planned every
    option with the automatic rule while the console asked, so the two front
    doors to the same install disagreed."""
    import profiles
    gpu = profiles.detect_gpu()
    total = vram_gib or gpu.total_gib
    if total <= 0:
        # same shape as the normal return - the page reads these keys
        # unconditionally, and a card it cannot read is exactly when it should
        # not also hit a missing-property error
        return {"budget": 0.0, "options": [], "gpu": gpu.name, "vram_gib": 0.0,
                "needs_vram": True, "vision": want_vision,
                "hidden_by_vision": [], "vision_cost": ""}
    budget, options = profiles.plan(total, want_vision)
    if want_vision and not options:
        # every quant was dropped by the images requirement, not by the card
        options = profiles.plan(total, False)[1]
        want_vision = False
    current = (cfg.get("PROFILE") or "").strip().lower()
    for o in options:
        o["id"] = o["quant"]
        o["current"] = current == f"{o['quant']}bpw-{round(o['ctx'] / 1000)}k"
        o["downloaded"] = (ROOT / o["model_dir"] / "config.json").is_file()
    hidden = []
    if want_vision:
        shown = {o["quant"] for o in options}
        hidden = [o["quant"] for o in profiles.plan(total, False)[1]
                  if o["quant"] not in shown]
    return {"budget": round(budget, 1), "options": options, "gpu": gpu.name,
            "vram_gib": round(total, 1), "needs_vram": False,
            "vision": want_vision, "hidden_by_vision": hidden,
            "vision_cost": profiles.vision_cost_line(total)}


def cards() -> list[dict]:
    """The card presets the simulation offers, newest generation first."""
    import profiles
    return [{"name": c.name, "vram_gib": c.vram_gib, "cc": c.cc, "gen": c.gen}
            for c in profiles.CARDS]


def simulate(vram_gib: float, cc: float = 0.0, driver: str = "",
             name: str = "", want_vision=None) -> dict:
    """Read-only: what this kit would offer on some other card.

    Deliberately does not touch cfg or .env - it exists so someone can check
    what a 16 GB card gets before buying one, or while setting a different one
    up, without changing their own install."""
    import profiles
    data = profiles.simulate(vram_gib, cc, driver, name, want_vision)
    for menu in data["menus"].values():
        for o in menu["options"]:
            o["id"] = o["quant"]
            o["downloaded"] = (ROOT / o["model_dir"] / "config.json").is_file()
    return data


def apply_choice(cfg: dict, quant_id: str, vram_gib: float = 0.0, want_vision=None) -> dict:
    """Write the chosen profile into .env and hand back the updated config."""
    import profiles
    data = options_for(cfg, vram_gib, want_vision)
    chosen = next((o for o in data["options"] if o["id"] == quant_id), None)
    if chosen is None:
        raise SetupError(f"{quant_id} is not one of the profiles that fit this GPU.",
                         "Reload the page - the list is built from the card that is installed.")
    gpu_name = data.get("gpu") or "unknown"
    updates = profiles.env_updates(chosen, gpu_name)
    try:
        profiles.write_env(ENV_FILE, updates)
    except PermissionError as e:
        raise SetupError("Could not write .env - the file is open in another program.",
                         "Close any editor holding .env and try again.") from e
    merged = dict(cfg)
    merged.update(updates)
    return merged


# ---------------------------------------------------------------- install ----

def _stream(cmd: list[str], log, cwd: Path = ROOT, env: dict | None = None,
            cancelled=None) -> int:
    """Run a command, forwarding every line to `log` as it appears."""
    log(f"$ {' '.join(cmd)}", "cmd")
    proc = subprocess.Popen(
        cmd, cwd=str(cwd), env=env or os.environ.copy(),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, encoding="utf-8", errors="replace",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0,
    )
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            log(line.rstrip("\n"), "out")
            if cancelled is not None and cancelled():
                proc.terminate()
                break
    finally:
        try:
            proc.stdout.close()             # type: ignore[union-attr]
        except Exception:                    # noqa: BLE001
            pass
    return proc.wait()


def _pip(*args: str) -> list[str]:
    return [str(VENV_PY), "-m", "pip", *args]


def _tail(lines: list[str], n: int = 12) -> str:
    return "\n".join(lines[-n:])


def install_environment(cfg: dict, log, steps: Steps, cancelled=None) -> None:
    """venv -> pip -> torch -> triton -> engine -> server libraries -> check.

    Raises SetupError with a message a non-developer can act on. `log(line, kind)`
    receives every line of pip output; `steps` is updated as each stage moves."""
    import wheels
    import win_start                      # console helpers reused: one definition of the rules

    recent: list[str] = []

    def tee(line: str, kind: str = "out") -> None:
        recent.append(line)
        del recent[:-200]
        log(line, kind)

    def stop() -> bool:
        return bool(cancelled and cancelled())

    # --- venv --------------------------------------------------------------
    # A venv whose base interpreter was upgraded or uninstalled still has a
    # python.exe; it just cannot run. Skipping the rebuild because the file
    # exists would then point every pip call below at a corpse.
    usable = VENV_PY.is_file() and subprocess.run(
        [str(VENV_PY), "-c", "pass"], capture_output=True).returncode == 0
    if usable:
        steps.skip("venv", "already there")
    else:
        if VENV_PY.is_file():
            log("# the existing .venv cannot run - rebuilding it", "cmd")
            steps.detail("venv", "the old one was broken, building a new one")
            shutil.rmtree(VENV_DIR, ignore_errors=True)
        steps.start("venv")
        sys_py = _system_python()
        if not sys_py:
            raise SetupError(
                "Python 3 was not found on this PC.",
                "Install 64-bit Python 3.11 or newer from python.org, tick "
                "\"Add python.exe to PATH\", then start Simplex again.")
        if _stream(sys_py + ["-m", "venv", str(VENV_DIR)], tee, cancelled=stop) != 0:
            raise SetupError("Could not create the Python environment.",
                             "Check that the folder is writable and not synced by OneDrive while in use.")
        steps.finish("venv")
    if stop():
        return

    scripts = str(VENV_DIR / ("Scripts" if os.name == "nt" else "bin"))
    os.environ["PATH"] = scripts + os.pathsep + os.environ.get("PATH", "")

    # --- pip / build tools -------------------------------------------------
    steps.start("pipbase")
    if _stream(_pip("install", "--upgrade", "pip", "setuptools", "wheel",
                    "typing_extensions", "packaging", "ninja"), tee, cancelled=stop) != 0:
        raise SetupError("pip could not update itself.",
                         "Usually a proxy or antivirus blocking pypi.org. Retry, or set "
                         "PIP_INDEX_URL in .env to a mirror you can reach.")
    steps.finish("pipbase")
    if stop():
        return

    # --- torch -------------------------------------------------------------
    steps.start("torch")
    index = win_start.torch_index_for_driver(cfg)
    steps.detail("torch", f"CUDA build {index.rsplit('/', 1)[-1]}")
    if _stream(_pip("install", "torch", "--extra-index-url", index), tee, cancelled=stop) != 0:
        raise SetupError("PyTorch could not be installed.",
                         f"The download comes from {index}. Retry on a stable connection, or set "
                         "TORCH_INDEX_URL in .env if a firewall blocks pytorch.org.")
    steps.finish("torch")
    if stop():
        return

    tags = wheels.interpreter_tags(VENV_PY)

    # --- triton (Windows only) --------------------------------------------
    if sys.platform == "win32":
        steps.start("triton")
        have = subprocess.run([str(VENV_PY), "-c", "import triton"],
                              capture_output=True, text=True).returncode == 0
        if have:
            steps.skip("triton", "already there")
        else:
            args = wheels.prebuilt_args("triton-windows", tags, cfg)
            rc = _stream(_pip(*args), tee, cancelled=stop) if args else 1
            if rc != 0:
                rc = _stream(_pip("install", "-U", "triton-windows"), tee, cancelled=stop)
            if rc != 0 or subprocess.run([str(VENV_PY), "-c", "import triton"],
                                         capture_output=True).returncode != 0:
                raise SetupError(
                    "Triton for Windows could not be installed, and the engine needs it.",
                    "Retry with the connection up. If it keeps failing, download a "
                    "triton-windows wheel matching " + tags.get("py", "your Python") +
                    " into the kit's wheels\\ folder and start again.")
            steps.finish("triton")
    if stop():
        return

    # --- engine ------------------------------------------------------------
    steps.start("engine")
    wheel_note = wheels.describe(ENGINE_PACKAGE, tags, cfg)
    installed = False
    if wheel_note:
        steps.detail("engine", wheel_note)
        tee(f"# trying the prebuilt engine: {wheel_note}", "cmd")
        args = wheels.prebuilt_args(ENGINE_PACKAGE, tags, cfg)
        if args and _stream(_pip(*args), tee, cancelled=stop) == 0:
            installed = True
        else:
            tee("# no usable prebuilt engine - falling back to building from source", "cmd")
    if stop():
        return
    if not installed:
        blockers = wheels.source_build_blockers()
        if blockers:
            raise SetupError(
                "The engine has to be compiled here, and the tools for it are missing: "
                + "; ".join(blockers) + ".",
                "Two ways out. Easiest: put a prebuilt exllamav3 wheel for "
                + tags.get("py", "this Python") + " into the kit's wheels\\ folder (or point "
                "WHEEL_INDEX in .env at one) and start again - no compiler needed. "
                "Otherwise install the missing tools and start again.")
        steps.detail("engine", "compiling CUDA kernels - 5 to 20 minutes")
        env = os.environ.copy()
        arch = cfg.get("TORCH_CUDA_ARCH_LIST") or env.get("TORCH_CUDA_ARCH_LIST")
        if arch:
            env["TORCH_CUDA_ARCH_LIST"] = arch
        cuda_home = win_start.find_cuda_home(cfg)
        if cuda_home:
            env["CUDA_HOME"] = cuda_home
            env.setdefault("CUDA_PATH", cuda_home)
        jobs = cfg.get("MAX_JOBS") or env.get("MAX_JOBS")
        if not jobs:
            n = int(env.get("NUMBER_OF_PROCESSORS", "4") or "4")
            jobs = "8" if n >= 8 else "4"
        env["MAX_JOBS"] = str(jobs)
        env["GIT_TERMINAL_PROMPT"] = "0"
        src = win_start.resolve_engine_src(cfg)
        if src.startswith("git+") and not shutil.which("git"):
            raise SetupError(
                "Git is needed to fetch the engine source, and it is not installed.",
                "Install Git for Windows, or drop a prebuilt exllamav3 wheel into the "
                "kit's wheels\\ folder to skip the build entirely.")
        if _stream(_pip("install", "--no-build-isolation", src), tee, env=env, cancelled=stop) != 0:
            raise SetupError(
                "Compiling the ExLlamaV3 engine failed.",
                "The last lines of the build log are above. The usual causes are a "
                "CUDA Toolkit that does not match the Visual Studio version, or not "
                "enough RAM for parallel compiles (set MAX_JOBS=2 in .env). A prebuilt "
                "wheel in wheels\\ avoids the build altogether.")
    steps.finish("engine", wheel_note or "compiled from source")
    if stop():
        return

    # --- server libraries --------------------------------------------------
    steps.start("server")
    if _stream(_pip("install", "aiohttp", "huggingface_hub", "pillow"), tee, cancelled=stop) != 0:
        raise SetupError("The server libraries could not be installed.",
                         "Retry - this step is a plain download from pypi.org.")
    steps.finish("server")
    if stop():
        return

    # --- verify ------------------------------------------------------------
    steps.start("check")
    ok, reason = venv_ready()
    if not ok:
        raise SetupError(f"Setup finished but the environment is still not usable: {reason}.",
                         "Delete the .venv folder and run setup again. If PyTorch cannot see "
                         "the GPU, update the NVIDIA driver first.")
    steps.finish("check", "engine v" + ENGINE_VERSION + ", GPU visible")


# ---------------------------------------------------------------- weights ----

def download_weights(cfg: dict, on_progress=None, cancelled=None, token: str = ""):
    """Fetch the chosen quant. Returns the downloader so a caller can cancel it."""
    import downloader
    repo = cfg.get("HF_TARGET_REPO") or ""
    if not repo:
        raise SetupError("No model repository is set.", "Pick a profile first.")
    _, dest = weights_ready(cfg)
    dl = downloader.Download(
        repo, dest, cfg.get("HF_REVISION") or "",
        token or cfg.get("HF_TOKEN") or os.environ.get("HF_TOKEN", ""),
        on_progress=on_progress,
    )
    if cancelled is not None:
        def watch() -> None:
            while dl.progress.state in ("idle", "listing", "downloading", "verifying"):
                if cancelled():
                    dl.cancel()
                    return
                time.sleep(0.5)
        threading.Thread(target=watch, daemon=True).start()
    return dl


def needs_setup(cfg: dict) -> tuple[bool, list[str]]:
    """(needed, reasons). The launcher's one question before it decides whether
    to show the setup page or go straight to loading the model."""
    reasons = []
    if not (cfg.get("PROFILE") or "").strip() or (cfg.get("PROFILE") or "").strip().lower() == "ask":
        reasons.append("no profile chosen yet")
    ok, why = venv_ready()
    if not ok:
        reasons.append(why)
    have, _path = weights_ready(cfg)
    if not have and cfg.get("MODEL_DIR"):
        reasons.append("the model weights are not on disk yet")
    return bool(reasons), reasons
