#!/usr/bin/env python3
"""Windows launcher (called by start.bat). Port of start.sh:

  first run  -> venv, torch, ExLlamaV3 v1.4.4 (CUDA compile), server deps
  every run  -> download weights if missing, serve, then open the built-in
                chat UI once the server is Ready
                (UI in .env: browser | server | no; see tools/chatui.py).
                Cherry Studio is still available for anyone who prefers it:
                CHERRY_AUTOSTART=ask|yes (default no; see tools/cherry.py)
"""
from __future__ import annotations

import codecs
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
ENV_FILE = ROOT / ".env"
ENV_EXAMPLE = ROOT / ".env.example"
SERVE = ROOT / "tools" / "serve_openai.py"
DEFAULT_ENGINE = "git+https://github.com/turboderp-org/exllamav3.git@v1.4.4"
DEFAULT_TORCH_INDEX = "https://download.pytorch.org/whl/cu130"
DEFAULT_REPO = "Mia-AiLab/Qwen3.8-27B-EXL3-2.0bpw"
RESTART_CODE = 87        # the server asks for a reload (see tools/webui_models.py)

# ASCII-only palette (cmd.exe OEM pages turn UTF-8 dashes into garbage).
_USE_COLOR = False


def _enable_console() -> None:
    global _USE_COLOR
    if sys.platform == "win32":
        os.system("")  # enable VT sequences in conhost / Windows Terminal
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    _USE_COLOR = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    if not _USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def cyan(t: str) -> str:
    return _c("96;1", t)


def dim(t: str) -> str:
    return _c("90", t)


def green(t: str) -> str:
    return _c("92", t)


def yellow(t: str) -> str:
    return _c("93", t)


def red(t: str) -> str:
    return _c("91;1", t)


def print_banner() -> None:
    inner = 60
    top = "+" + "-" * inner + "+"
    def line(text: str, paint=None) -> None:
        pad = text.ljust(inner)
        print(cyan("|") + (paint(pad) if paint else pad) + cyan("|"))
    print()
    print(cyan(top))
    line(" ")
    line("  Simplex - one-click Qwen3.8-27B for Windows", cyan)
    # 16 GB is what this kit is built around and what the README claims. A 12 GB
    # card technically loads the 2.0bpw baseline at 33k context and nothing else -
    # no images, no better quant - which is not what "supported" should promise.
    line("  EXL3  |  NVIDIA 16 GB+  |  up to 262k context", dim)
    line(" ")
    print(cyan(top))
    print()


def step(n: int, total: int, msg: str) -> None:
    print(f"  {yellow(f'[{n}/{total}]')}  {msg} ...", flush=True)


def step_ok(msg: str = "done", elapsed: float | None = None) -> None:
    extra = f"  {dim(f'({elapsed:.0f}s)')}" if elapsed is not None else ""
    print(f"          {green('OK')}  {msg}{extra}", flush=True)


def info(msg: str) -> None:
    print(f"  {dim('*')} {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"  {yellow('!')} {msg}", flush=True)


LOGBOOK = None          # set by main(); tools/logbook.py


def die(msg: str, code: int = 1) -> None:
    print(file=sys.stderr)
    print(red("  ERROR"), file=sys.stderr)
    for line in msg.splitlines():
        print(f"    {line}", file=sys.stderr)
    if LOGBOOK is not None and getattr(LOGBOOK, "path", None):
        print(f"    Full log: {LOGBOOK.path}", file=sys.stderr)
    print(file=sys.stderr)
    sys.exit(code)


def load_dotenv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.split("#", 1)[0].strip().strip('"').strip("'")
        if key:
            out[key] = val
    return out


def pip_cmd(*args: str) -> list[str]:
    # Always `python -m pip`: on Windows, upgrading via pip.exe cannot
    # overwrite the running pip.exe ("To modify pip, please run ... python.exe -m pip").
    return [str(VENV_PY), "-m", "pip", *args]


def run(cmd: list[str], **kw) -> None:
    print(dim("      > " + " ".join(cmd)), flush=True)
    r = subprocess.run(cmd, **kw)
    if r.returncode != 0:
        die(f"command failed ({r.returncode}): {' '.join(cmd)}", r.returncode)


def _pump_child(proc: subprocess.Popen) -> None:
    """Echo the server's output to this console. sys.stdout is a tee (see
    tools/logbook.py), so writing here also files it away for the tray menu's
    "View the log".

    Read as bytes, deliberately. A text-mode pipe is in universal-newline mode,
    where a bare CR counts as a line ending - which turns the engine's
    single-line load bar into hundreds of near-identical console lines and
    megabytes of log. Reading raw and decoding here keeps the CR a CR, so the
    bar redraws in place exactly as it does without the pipe."""
    stream = proc.stdout
    if stream is None:
        return
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    try:
        while True:
            chunk = stream.read(4096)
            if not chunk:
                break
            text = decoder.decode(chunk)
            if text:
                sys.stdout.write(text)
    except Exception:                    # noqa: BLE001 - the child went away
        pass
    finally:
        tail = decoder.decode(b"", True)
        if tail:
            try:
                sys.stdout.write(tail)
            except Exception:            # noqa: BLE001
                pass
        try:
            stream.close()
        except Exception:                # noqa: BLE001
            pass


def venv_ok() -> bool:
    if not VENV_PY.is_file():
        return False
    probe = (
        "import torch, exllamav3, aiohttp, huggingface_hub\n"
        "import sys\n"
        "sys.exit(0 if torch.cuda.is_available() else 2)\n"
    )
    r = subprocess.run([str(VENV_PY), "-c", probe], capture_output=True, text=True)
    if r.returncode == 2:
        die(
            "PyTorch in .venv cannot see a CUDA GPU. Install an NVIDIA driver,\n"
            "delete the .venv folder, set TORCH_INDEX_URL in .env if needed, and run start.bat again."
        )
    return r.returncode == 0


def ensure_triton() -> None:
    """ExLlamaV3 v1.4.4 imports Triton kernels at module load. On Windows the
    stock `triton` package is not available; without `triton-windows` you get:
    ImportError: cannot import name '_dsa_attn_split_kernel' from dsa_triton."""
    if sys.platform != "win32" or not VENV_PY.is_file():
        return
    r = subprocess.run(
        [str(VENV_PY), "-c", "import triton"],
        capture_output=True, text=True,
    )
    if r.returncode == 0:
        return
    print(f"  {yellow('[ + ]')}  Installing triton-windows  (required on Windows)")
    run(pip_cmd("install", "-U", "triton-windows"), cwd=str(ROOT))
    r2 = subprocess.run(
        [str(VENV_PY), "-c", "import triton"],
        capture_output=True, text=True,
    )
    if r2.returncode != 0:
        die(
            "Could not import Triton after installing triton-windows.\n"
            "Install a matching wheel:  .venv\\Scripts\\python.exe -m pip install -U triton-windows\n"
            + (r2.stderr or r2.stdout or "")[:800]
        )
    step_ok("triton-windows")


def ensure_pillow() -> None:
    """Image input (vision tower) decodes pictures with Pillow. Kits set up
    before images were supported have a venv without it - add it quietly."""
    if not VENV_PY.is_file():
        return
    r = subprocess.run([str(VENV_PY), "-c", "import PIL"], capture_output=True, text=True)
    if r.returncode == 0:
        return
    print(f"  {yellow('[ + ]')}  Installing pillow  (image input)")
    r2 = subprocess.run(pip_cmd("install", "pillow"), cwd=str(ROOT))
    if r2.returncode != 0:
        warn("pillow could not be installed - the server will run text-only")


def ensure_kv_patch(fmt: str) -> None:
    """CACHE_QUANT=fp8|nvfp4 need the fp8/nvfp4 KV lanes hot-patched into the
    installed ExLlamaV3 (tools/patch_kv.py; pure Python + Triton, no recompile)."""
    tool = ROOT / "tools" / "patch_kv.py"
    if not tool.is_file() or not VENV_PY.is_file():
        die(f"CACHE_QUANT={fmt} needs tools/patch_kv.py and an installed engine")
    r = subprocess.run([str(VENV_PY), str(tool), "apply"], capture_output=True, text=True, cwd=str(ROOT))
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0:
        die(f"could not enable the {fmt} KV cache:\n{out.strip()}\n"
            f"Use CACHE_QUANT=8,4 (stock engine) or fix the patch state (tools\\patch_kv.py status).")
    if "already applied" not in out:
        step_ok(f"KV cache patch applied ({fmt} available)")
    else:
        info(f"KV cache: {fmt} (patched engine)")


def find_cuda_home(cfg: dict[str, str]) -> str | None:
    for key in ("CUDA_HOME", "CUDA_PATH"):
        v = cfg.get(key) or os.environ.get(key)
        if v and Path(v).is_dir():
            return v
    base = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    if base.is_dir():
        versions = sorted((p for p in base.iterdir() if p.is_dir()), reverse=True)
        if versions:
            return str(versions[0])
    return None


def resolve_engine_src(cfg: dict[str, str]) -> str:
    """Use a local checkout only if it exists on this machine. Copied Spark
    .env files often set EXL3_REPO to a Linux aarch64 path that pip cannot
    install on Windows."""
    if (ROOT / "exllamav3" / "__init__.py").is_file():
        return str(ROOT)
    raw = (cfg.get("EXL3_REPO") or os.environ.get("EXL3_REPO") or "").strip()
    if not raw:
        return DEFAULT_ENGINE
    if raw.startswith(("git+", "http://", "https://", "file:")):
        return raw
    p = Path(raw)
    if p.is_dir() and (
        (p / "setup.py").is_file()
        or (p / "pyproject.toml").is_file()
        or (p / "exllamav3" / "__init__.py").is_file()
    ):
        return str(p.resolve())
    print(f"  {yellow('!')} EXL3_REPO is not a usable path here ({raw})")
    print(f"     falling back to {DEFAULT_ENGINE}")
    return DEFAULT_ENGINE


def bootstrap(cfg: dict[str, str]) -> None:
    print(yellow("  First-run setup") + dim("  (once; later launches skip this)"))
    print()
    py = shutil.which("py")
    sys_py = [py, "-3"] if py else None
    if sys_py is None:
        for name in ("python", "python3"):
            w = shutil.which(name)
            if w:
                sys_py = [w]
                break
    if not sys_py:
        die(
            "Python 3 was not found. Install 64-bit Python 3.11+ from python.org\n"
            "  (check 'Add python.exe to PATH'), then double-click start.bat again."
        )

    t0 = time.time()
    step(1, 5, "Creating Python virtualenv")
    run(sys_py + ["-m", "venv", str(ROOT / ".venv")], cwd=str(ROOT))
    step_ok("virtualenv", time.time() - t0)

    t1 = time.time()
    step(2, 5, "Build tools (pip, setuptools, ninja)")
    run(
        pip_cmd("install", "--upgrade", "pip", "setuptools", "wheel",
                "typing_extensions", "packaging", "ninja"),
        cwd=str(ROOT),
    )
    step_ok("build tools", time.time() - t1)

    torch_index = torch_index_for_driver(cfg)
    t2 = time.time()
    step(3, 5, "PyTorch  (~2-3 GB the first time)")
    run(
        pip_cmd("install", "torch", "--extra-index-url", torch_index),
        cwd=str(ROOT),
    )
    step_ok("PyTorch", time.time() - t2)
    ensure_triton()

    engine_src = resolve_engine_src(cfg)
    if engine_src == str(ROOT):
        note = "Local engine  - compiling CUDA kernels"
    else:
        note = "ExLlamaV3 v1.4.4  - clone + compile CUDA kernels"

    arch = cfg.get("TORCH_CUDA_ARCH_LIST") or os.environ.get("TORCH_CUDA_ARCH_LIST")
    if arch:
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch
    cuda_home = find_cuda_home(cfg)
    if cuda_home:
        os.environ["CUDA_HOME"] = cuda_home
        os.environ.setdefault("CUDA_PATH", cuda_home)
        info(f"CUDA_HOME = {cuda_home}")
    else:
        warn("CUDA toolkit not found. Set CUDA_HOME or install the NVIDIA CUDA Toolkit.")
        warn("The engine compile will likely fail without it.")

    max_jobs = cfg.get("MAX_JOBS") or os.environ.get("MAX_JOBS")
    if not max_jobs:
        n = int(os.environ.get("NUMBER_OF_PROCESSORS", "4") or "4")
        max_jobs = "8" if n >= 8 else "4"
    os.environ["MAX_JOBS"] = str(max_jobs)
    os.environ["GIT_TERMINAL_PROMPT"] = "0"

    scripts = str(ROOT / ".venv" / "Scripts")
    os.environ["PATH"] = scripts + os.pathsep + os.environ.get("PATH", "")

    if not shutil.which("git"):
        die("Git was not found on PATH. Install Git for Windows, then run start.bat again.")

    t3 = time.time()
    step(4, 5, f"{note}  (5-20 min, needs VS C++ tools)")
    run(
        pip_cmd("install", "--no-build-isolation", engine_src),
        cwd=str(ROOT),
    )
    step_ok("engine", time.time() - t3)

    t4 = time.time()
    step(5, 5, "Server dependencies (aiohttp, huggingface_hub, pillow)")
    run(pip_cmd("install", "aiohttp", "huggingface_hub", "pillow"), cwd=str(ROOT))
    step_ok("server deps", time.time() - t4)
    print()
    print(f"  {green('Setup complete.')}  Next launches start in a few seconds.")
    print()


def require_engine_version() -> None:
    r = subprocess.run(
        [str(VENV_PY), "-c",
         "from exllamav3.version import __version__ as v; print(v)"],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    ver = (r.stdout or "").strip() or "unknown"
    if r.returncode != 0 or ver != "1.4.4":
        die(
            f"this kit requires ExLlamaV3 v1.4.4, but the venv has '{ver}'.\n"
            f"Fix: delete the .venv folder and run start.bat again."
        )


def nvidia_total_mib() -> int:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            line = (r.stdout or "").strip().splitlines()[0].strip()
            return int(float(line))
    except (FileNotFoundError, ValueError, IndexError, subprocess.TimeoutExpired):
        pass
    return 0


def nvidia_cc_driver() -> tuple[float, int]:
    """(compute capability, driver major) of GPU 0, or (0, 0)."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap,driver_version",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        if r.returncode == 0 and r.stdout.strip():
            cc, drv = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")[:2]]
            return float(cc), int(drv.split(".")[0])
    except (FileNotFoundError, ValueError, IndexError, subprocess.TimeoutExpired):
        pass
    return 0.0, 0


def torch_index_for_driver(cfg: dict[str, str]) -> str:
    """PyTorch wheel index: .env/TORCH_INDEX_URL wins; otherwise pick by driver:
    CUDA 13 wheels need driver >= 580, CUDA 12.8 wheels >= 570 (and are the
    minimum for Blackwell), CUDA 12.6 for anything older."""
    explicit = cfg.get("TORCH_INDEX_URL") or os.environ.get("TORCH_INDEX_URL")
    if explicit:
        return explicit
    cc, drv = nvidia_cc_driver()
    if drv and drv < 580:
        idx = "https://download.pytorch.org/whl/cu128" if drv >= 570 else "https://download.pytorch.org/whl/cu126"
        warn(f"NVIDIA driver {drv}.x is older than CUDA 13 needs (580+): using {idx.rsplit('/', 1)[-1]} wheels")
        if cc >= 12.0 and drv < 570:
            warn("Blackwell GPUs need driver 570+ - update the driver if the engine fails to load")
        return idx
    return DEFAULT_TORCH_INDEX


def nvidia_mem_mib() -> tuple[int, int, int]:
    """(used, free, total) MiB for GPU 0 via nvidia-smi, or (0, 0, 0)."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            parts = [int(float(x)) for x in (r.stdout or "").strip().splitlines()[0].split(",")]
            if len(parts) == 3:
                return parts[0], parts[1], parts[2]
    except (FileNotFoundError, ValueError, IndexError, subprocess.TimeoutExpired):
        pass
    return 0, 0, 0


def vram_users_windows(min_mib: int = 40) -> list[tuple[int, str, int]]:
    """Processes holding dedicated VRAM, biggest first: (MiB, name, pid).
    Uses the same performance counters Task Manager reads (graphics apps such as
    the desktop compositor, browsers or Cherry Studio never show up in
    nvidia-smi on Windows, but they do here)."""
    if sys.platform != "win32":
        return []
    ps = (
        "$ErrorActionPreference='Stop';"
        "$c = Get-Counter '\\GPU Process Memory(*)\\Dedicated Usage';"
        "$acc = @{};"
        "foreach ($s in $c.CounterSamples) {"
        "  if ($s.InstanceName -match 'pid_(\\d+)') { $acc[$matches[1]] = [int64]$acc[$matches[1]] + [int64]$s.CookedValue }"
        "};"
        "foreach ($k in $acc.Keys) {"
        "  $p = Get-Process -Id ([int]$k) -ErrorAction SilentlyContinue;"
        "  $n = if ($p) { $p.ProcessName } else { '?' };"
        "  '{0}|{1}|{2}' -f [int64]($acc[$k] / 1MB), $k, $n"
        "}"
    )
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                           capture_output=True, text=True, timeout=25)
    except Exception:  # noqa: BLE001
        return []
    out: list[tuple[int, str, int]] = []
    for line in (r.stdout or "").splitlines():
        try:
            mib, pid, name = line.strip().split("|", 2)
            if int(mib) >= min_mib:
                out.append((int(mib), name, int(pid)))
        except ValueError:
            continue
    out.sort(reverse=True)
    return out


def vram_preflight(gpu_mem_gb: str, margin_gib: float = 0.3) -> None:
    """Make sure ~GPU_MEM_GB of VRAM is actually free before the long load.
    On Windows a short budget does not fail loudly: the driver can spill CUDA
    memory to system RAM and the model then runs many times slower."""
    try:
        need_mib = int(float(gpu_mem_gb) * 1024)
    except ValueError:
        return
    used, free, total = nvidia_mem_mib()
    if total <= 0:
        warn("nvidia-smi not found - skipping the free-VRAM check.")
        return
    want = need_mib + int(margin_gib * 1024)
    gib = lambda m: f"{m / 1024:.1f} GB"
    if total < want:
        warn(f"This GPU has {gib(total)} of VRAM but the recipe needs about {gib(need_mib)}.")
        warn("Lower GPU_MEM_GB and CONTEXT_SIZE in .env, or the load will fail / crawl.")
    while free < want:
        print()
        print(f"  {yellow('VRAM check')}  need ~{gib(need_mib)} free, have {gib(free)} "
              f"(of {gib(total)}, {gib(used)} in use)")
        users = vram_users_windows()
        if users:
            print("  Close these before continuing (they hold VRAM):")
            for mib, name, pid in users[:8]:
                print(f"      {mib:6d} MB  {name}  (pid {pid})")
            info("Typical culprits: browsers with many tabs, games, Discord, video/3D apps, other AI tools.")
        else:
            info("Close browsers, games, Discord, other AI tools, then re-check.")
        info("Windows may otherwise page the model into system RAM and the model runs very slowly.")
        ans = timed_input(
            f"  {green('?')} Press Enter to re-check, 'c' to continue anyway, 'q' to quit  "
            f"{dim('(auto-continue in 120s)')} ", 120)
        if ans is None or ans.lower() in ("c", "continue"):
            warn(f"Continuing with {gib(free)} free. If the load fails or is very slow, "
                 "free VRAM or lower CONTEXT_SIZE / GPU_MEM_GB in .env.")
            return
        if ans.lower() in ("q", "quit", "exit"):
            die("Stopped by user (free some VRAM and run start.bat again).", 3)
        used, free, total = nvidia_mem_mib()
    step_ok(f"VRAM  {gib(free)} free of {gib(total)}  (need ~{gib(need_mib)})")


def download_model(py: Path, repo: str, dest: Path, label: str, revision: str | None = None) -> None:
    """Fetch the weights with tools/downloader.py: no dependency on the venv,
    resumes a half-finished download, and prints a real percentage and ETA
    instead of a cursor that sits still for twenty minutes."""
    sys.path.insert(0, str(ROOT / "tools"))
    import downloader

    dest.mkdir(parents=True, exist_ok=True)
    if downloader.is_complete(dest):
        info(f"Weights already in {dest}")
        return
    print()
    print(f"  {yellow('[dl]')}  Downloading {repo}" + (f"  (revision {revision})" if revision else ""))
    info(f"into {dest}")

    last = [0.0]

    def show(p: dict) -> None:
        now = time.time()
        if p["state"] not in ("done", "error") and now - last[0] < 0.5:
            return
        last[0] = now
        eta = p["eta_seconds"]
        eta_s = "--:--" if eta < 0 else f"{int(eta) // 60:3d}:{int(eta) % 60:02d}"
        bar_w = 24
        filled = int(bar_w * p["percent"] / 100)
        bar = "#" * filled + "-" * (bar_w - filled)
        print(f"\r      [{bar}] {p['percent']:5.1f}%  "
              f"{downloader.humanize(p['done_bytes'])}/{downloader.humanize(p['total_bytes'])}  "
              f"{downloader.humanize(p['speed_bps'])}/s  ETA {eta_s}   ",
              end="", flush=True)

    token = os.environ.get("HF_TOKEN", "")
    dl = downloader.Download(repo, dest, revision or "", token, on_progress=show)
    try:
        dl.run()
    except downloader.DownloadError as e:
        print()
        die(f"{e}\n{e.hint}" if e.hint else str(e))
    print()
    step_ok(f"downloaded {downloader.humanize(dl.progress.total_bytes)}")


# ----------------------------------------------------------------------------
# First run: a page in the browser, or the console for anyone who prefers it
# ----------------------------------------------------------------------------
def setup_mode(cfg: dict[str, str]) -> str:
    """SETUP=browser (default) puts first-run setup on a web page at the same
    address the chat UI will use. SETUP=console keeps the old question-and-answer
    flow in this window - useful over SSH, or when no browser can open."""
    mode = (cfg.get("SETUP") or os.environ.get("SIMPLEX_SETUP") or "browser").strip().lower()
    if mode in ("1", "yes", "true", "on", "web", "ui"):
        return "browser"
    if mode in ("0", "no", "false", "off", "text", "terminal"):
        return "console"
    if mode not in ("browser", "console"):
        warn(f"SETUP={mode!r} is not browser/console - using 'browser'")
        return "browser"
    return mode


def run_setup_web(cfg: dict[str, str], port: str, reasons: list[str],
                  force_profile: bool = False) -> tuple[str, dict[str, str] | None]:
    """Hand first-run over to the browser.

    Returns (outcome, config): "done", "cancelled", "error", or "unavailable"
    when the page itself could not open - only the last of those is a reason
    to fall back to asking the same questions in this window."""
    sys.path.insert(0, str(ROOT / "tools"))
    import setup_web

    def console(line: str, kind: str) -> None:
        if kind == "cmd":
            print(dim("      > " + line[2:] if line.startswith("$ ") else "      " + line), flush=True)
        elif kind == "error":
            print(f"  {red('!')}  {line}", flush=True)
        elif kind == "hint":
            print(f"     {line}", flush=True)
        else:
            print(dim("      " + line), flush=True)

    def banner(url: str) -> None:
        print()
        print(f"  {cyan('Setup is open in your browser:')}  {url}")
        if reasons:
            info("because " + "; ".join(reasons))
        info("Keep this window open. It prints the same log the page shows.")
        print()

    def still_open(seconds: float) -> None:
        print()
        warn("setup could not finish - the page explains why")
        info(f"it stays open at http://127.0.0.1:{port}/ for "
             f"{int(seconds / 60)} more minutes if you want to press Try again")
        print()

    try:
        return setup_web.run(cfg, int(port), "127.0.0.1", open_browser=True,
                             console=console, banner=banner, force=force_profile,
                             retry_message=still_open)
    except KeyboardInterrupt:
        raise
    except Exception as e:  # noqa: BLE001 - fall back rather than strand the user
        warn(f"the setup page could not start ({e})")
        warn("falling back to setup in this window")
        return "unavailable", None


# ----------------------------------------------------------------------------
# Tray icon and shortcuts (Windows) - tools/tray.py, tools/shortcuts.py
# ----------------------------------------------------------------------------
def tray_mode(cfg: dict[str, str]) -> bool:
    """TRAY=no turns off the tray icon. It is on wherever Windows can show one."""
    mode = (cfg.get("TRAY") or os.environ.get("SIMPLEX_TRAY") or "auto").strip().lower()
    return mode not in ("0", "no", "false", "off", "none")


def shortcut_mode(cfg: dict[str, str]) -> str:
    """SHORTCUTS=auto (default, made once after a successful first run),
    yes (always make sure they exist), or no."""
    mode = (cfg.get("SHORTCUTS") or os.environ.get("SIMPLEX_SHORTCUTS") or "auto").strip().lower()
    if mode in ("1", "true", "y"):
        return "yes"
    if mode in ("0", "false", "n", "off", "none"):
        return "no"
    return mode if mode in ("auto", "yes", "no") else "auto"


def ensure_shortcuts(cfg: dict[str, str]) -> None:
    """Put Simplex in the Start menu the first time it runs properly, so the
    next launch does not mean hunting for start.bat in a folder."""
    mode = shortcut_mode(cfg)
    if mode == "no" or sys.platform != "win32":
        return
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        import shortcuts
    except Exception:                    # noqa: BLE001
        return
    if mode == "auto" and (shortcuts.offered() or shortcuts.looks_installed()):
        return
    try:
        made = shortcuts.create()
        shortcuts.mark_offered(bool(made))
        if made:
            info("Added Simplex to the Start menu and the desktop "
                 "(SHORTCUTS=no in .env to skip this).")
    except Exception as e:               # noqa: BLE001 - cosmetic, never fatal
        warn(f"could not create shortcuts ({e})")


def _on_ready(cfg: dict[str, str], tray):
    """What happens the moment the server answers /health. Shortcuts are made
    here rather than before the launch, because "it started properly" is the
    thing worth putting on someone's desktop - a kit that dies on the model
    load should not leave an icon behind."""
    def ready() -> None:
        ensure_shortcuts(cfg)
        if tray is not None:
            tray.notify("Simplex is ready",
                        "The model is loaded. Click here to open the chat.")
    return ready


class Runtime:
    """What the tray menu acts on: the child process and the two flags the
    launcher loop checks after it exits."""

    def __init__(self):
        self.proc: subprocess.Popen | None = None
        self.restart = False
        self.quit = False
        self.url = ""

    def stop_child(self) -> None:
        proc = self.proc
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.terminate()
        except Exception:                # noqa: BLE001
            pass


def start_tray(rt: Runtime, log_path: Path | None):
    """Returns the Tray, or None when there is no tray to put an icon in."""
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        import tray as tray_mod
    except Exception:                    # noqa: BLE001
        return None
    if not tray_mod.Tray.available():
        return None

    def open_ui() -> None:
        import webbrowser
        webbrowser.open(rt.url or "http://127.0.0.1:8888/")

    def do_restart() -> None:
        rt.restart = True
        rt.stop_child()

    def do_quit() -> None:
        rt.quit = True
        rt.stop_child()

    items = [
        ("Open Simplex", open_ui),
        ("Restart the model", do_restart),
        (None, None),
        ("Show the Simplex folder", lambda: tray_mod.open_path(ROOT)),
        ("View the log", lambda: tray_mod.open_path(log_path) if log_path else None),
        (None, None),
        ("Quit Simplex", do_quit),
    ]
    t = tray_mod.Tray("Simplex", "Simplex - starting", items)
    return t if t.start() else None


# ----------------------------------------------------------------------------
# Cherry Studio (chat app) - see tools/cherry.py
# ----------------------------------------------------------------------------
def ui_mode(cfg: dict[str, str]) -> str:
    """UI=browser (default) opens the built-in chat UI once the server is Ready;
    UI=server serves it but opens nothing; UI=no turns it off entirely."""
    mode = (cfg.get("UI") or os.environ.get("UI") or "browser").strip().lower()
    if mode in ("1", "yes", "true", "on"):
        return "browser"
    if mode in ("0", "false", "off", "none"):
        return "no"
    if mode not in ("browser", "server", "no"):
        warn(f"UI={mode!r} is not browser/server/no - using 'browser'")
        return "browser"
    return mode


def ui_after_ready(proc: subprocess.Popen, host: str, port: str, open_browser: bool,
                   on_ready=None) -> None:
    """Runs in a thread: once /health answers, point the user at the chat UI."""
    if not wait_for_ready(proc, host, port):
        return
    if on_ready is not None:
        try:
            on_ready()
        except Exception:  # noqa: BLE001 - a notification is never worth a crash
            pass
    time.sleep(0.5)  # let the Ready box finish printing
    url = f"http://127.0.0.1:{port}/"
    print()
    info(f"Chat UI ready:  {cyan(url)}")
    if open_browser:
        try:
            import webbrowser
            webbrowser.open(url)
            info("Opened it in your browser.")
        except Exception as e:  # noqa: BLE001
            warn(f"Could not open a browser ({e}) - open the address above yourself.")
    info("Other devices on this network can use it too, at this PC's LAN address.")
    info("This window is the server - keep it open while chatting. Ctrl+C or stop.bat to stop.")
    print()


CHERRY_MODES = ("ask", "yes", "no")
CHERRY_PROMPT_SECONDS = 90


def cherry_mode(cfg: dict[str, str]) -> str:
    mode = (cfg.get("CHERRY_AUTOSTART") or os.environ.get("CHERRY_AUTOSTART") or "no").strip().lower()
    if mode in ("1", "true", "y"):
        mode = "yes"
    elif mode in ("0", "false", "n", "off"):
        mode = "no"
    if mode not in CHERRY_MODES:
        warn(f"CHERRY_AUTOSTART={mode!r} is not ask/yes/no - using 'ask'")
        mode = "ask"
    return mode


def prepare_cherry(cfg: dict[str, str], port: str, context: str) -> dict | None:
    """Download (once), initialise (once) and configure Cherry Studio so it
    points at this server. Never fatal: returns None and the server still starts."""
    try:
        sys.path.insert(0, str(ROOT / "tools"))
        import cherry  # noqa: WPS433  (tools/cherry.py)
    except Exception as e:  # noqa: BLE001
        warn(f"Cherry Studio helper unavailable ({e}) - continuing without it")
        return None
    cherry.set_logger(lambda m: print(m, flush=True))
    try:
        ctx = int(context)
    except ValueError:
        ctx = None
    step(1, 1, "Cherry Studio  (chat app, pre-configured for this server)")
    try:
        prep = cherry.prepare(cfg, port, ctx)
    except Exception as e:  # noqa: BLE001
        warn("Cherry Studio could not be prepared - the server will start anyway:")
        for line in str(e).splitlines():
            print(f"    {line}")
        warn("Chat with any OpenAI client instead (see README), or fix this and run start.bat again.")
        return None
    for n in prep.get("notes", []):
        info(n)
    if prep.get("mode") == "external":
        step_ok(f"using your own install: {prep['exe']}")
    else:
        step_ok(f"ready: {prep['exe'].name}")
    return prep


def timed_input(prompt: str, timeout: float) -> str | None:
    """input() with a timeout. Returns None if nobody typed anything in time."""
    print(prompt, end="", flush=True)
    if sys.platform != "win32":
        import select
        r, _, _ = select.select([sys.stdin], [], [], timeout)
        if r:
            return sys.stdin.readline().strip()
        print()
        return None
    import msvcrt
    buf: list[str] = []
    t0 = time.time()
    while time.time() - t0 < timeout:
        if msvcrt.kbhit():
            ch = msvcrt.getwch()
            if ch in ("\r", "\n"):
                print()
                return "".join(buf).strip()
            if ch == "\x08":
                if buf:
                    buf.pop()
                    print("\b \b", end="", flush=True)
            elif ch == "\x03":
                raise KeyboardInterrupt
            elif ch.isprintable():
                buf.append(ch)
                print(ch, end="", flush=True)
        else:
            time.sleep(0.05)
    print()
    return None


def wait_for_ready(proc: subprocess.Popen, host: str, port: str) -> bool:
    h = "127.0.0.1" if host in ("0.0.0.0", "", "::") else host
    url = f"http://{h}:{port}/health"
    while proc.poll() is None:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:  # noqa: BLE001  (refused until the model is loaded)
            pass
        time.sleep(2)
    return False


def cherry_after_ready(proc: subprocess.Popen, prep: dict, host: str, port: str, mode: str) -> None:
    """Runs in a thread: once /health answers, offer to open Cherry Studio."""
    if not wait_for_ready(proc, host, port):
        return
    import cherry  # already imported by prepare_cherry
    time.sleep(0.5)  # let the Ready box finish printing
    print()
    if prep.get("mode") == "external":
        info("Cherry Studio will ask to add the provider - click Add, then add the model")
        info(f"'{prep.get('model_id', '')}' under it (Manage models).")
    else:
        info("Cherry Studio is set up: provider, model and defaults point at this server.")
    open_it = mode == "yes"
    if mode == "ask":
        ans = timed_input(
            f"  {green('?')} Open Cherry Studio and start chatting now?  [Y/n]  "
            f"{dim(f'(Enter = yes, {CHERRY_PROMPT_SECONDS}s)')} ",
            CHERRY_PROMPT_SECONDS,
        )
        if ans is None:
            info("No answer - leaving Cherry Studio closed.")
        else:
            open_it = ans.lower() in ("", "y", "yes")
    if open_it:
        try:
            cherry.open_app(prep, port)
            info("Opening Cherry Studio ... (portable build: unpacks for ~10-20 s first)")
        except Exception as e:  # noqa: BLE001
            warn(f"Could not open Cherry Studio: {e}")
    else:
        info(f"Open it any time:  .venv\\Scripts\\python.exe tools\\cherry.py open")
    info("This window is the server - keep it open while chatting. Ctrl+C or stop.bat to stop.")
    print()


def server_command(cfg: dict[str, str]):
    """Everything .env says about how to launch the server.

    Called again after a model switch, so it must be safe to re-run:
    the download check is a no-op for weights already on disk and the KV
    patch is idempotent."""
    model_dir = cfg.get("MODEL_DIR")
    if not model_dir:
        die("MODEL_DIR must be set in .env")
    model_path = Path(model_dir)
    if not model_path.is_absolute():
        model_path = ROOT / model_path

    port = cfg.get("PORT", "8888")
    host = cfg.get("HOST", "0.0.0.0")
    context = cfg.get("CONTEXT_SIZE", "199936")
    cache_quant = cfg.get("CACHE_QUANT", "none")
    cpu_cache = cfg.get("CPU_CACHE_GB", "0")
    draft = (cfg.get("DRAFT") or "mtp").strip().lower()
    gpu_mem = cfg.get("GPU_MEM_GB")
    vision_mode = (cfg.get("VISION") or "auto").strip().lower()
    if vision_mode in ("0", "false", "no", "off", "none"):
        vision_mode = "off"
    elif vision_mode not in ("auto", "off"):
        warn(f"VISION={vision_mode!r} is not auto/off - using 'auto'")
        vision_mode = "auto"
    image_max_pixels = (cfg.get("IMAGE_MAX_PIXELS") or "1048576").strip()
    if gpu_mem:
        info(f"VRAM budget  {gpu_mem} GB   context  {context}   cache  {cache_quant}   draft  {draft}   images  {vision_mode}")
    else:
        vram = nvidia_total_mib()
        if vram > 0:
            gpu_mem = str(max(vram // 1024 - 2, 8))
        else:
            gpu_mem = "14.7"
        info(f"VRAM budget  {gpu_mem} GB (auto)   context  {context}")

    if draft not in ("mtp", "none"):
        die(f"DRAFT must be mtp or none (got: {draft})")

    # KV cache format: integer bits (stock engine) or fp8 / nvfp4 (patched engine)
    cache_quant = cache_quant.strip().lower().replace(" ", "")
    if cache_quant in ("fp8", "nvfp4"):
        cc, _drv = nvidia_cc_driver()
        if cc and cc < 8.9:
            die(f"CACHE_QUANT={cache_quant} needs an Ada or Blackwell GPU (compute capability 8.9+); "
                f"this GPU is {cc}. Use CACHE_QUANT=8,4 or 4 (integer cache, any GPU).")
        ensure_kv_patch(cache_quant)
    elif cache_quant != "none":
        try:
            parts = [int(x) for x in cache_quant.split(",")]
            if len(parts) not in (1, 2) or any(not 2 <= p <= 8 for p in parts):
                raise ValueError
        except ValueError:
            die(f"CACHE_QUANT must be none, 2-8, k_bits,v_bits, fp8 or nvfp4 (got: {cache_quant})")

    repo = cfg.get("HF_TARGET_REPO") or DEFAULT_REPO
    download_model(VENV_PY, repo, model_path, "target model", cfg.get("HF_REVISION") or None)

    model_id = (cfg.get("MODEL_ID") or model_path.name).strip().lower()
    cmd = [
        str(VENV_PY), "-u", str(SERVE),
        "--model", str(model_path),
        "--model_id", model_id,
        "--host", host,
        "--port", port,
        "--cache_size", context,
        "--grid_size", gpu_mem,
        "--draft_model", draft,
    ]
    if cache_quant != "none":
        cmd.extend(["--cache_quant", cache_quant])
    if cpu_cache not in ("0", "0.0", ""):
        cmd.extend(["--cpu_cache_size", cpu_cache])
    cmd.extend(["--vision", vision_mode, "--image_max_pixels", image_max_pixels])
    ui = ui_mode(cfg)
    cmd.extend(["--ui", "off" if ui == "no" else "on"])
    if cfg.get("UI_TITLE"):
        cmd.extend(["--ui_title", cfg["UI_TITLE"]])
    return cmd, host, port, gpu_mem, ui, context


def main() -> int:
    global LOGBOOK
    _enable_console()
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        import logbook
        LOGBOOK = logbook.Logbook()
        LOGBOOK.start()
    except Exception:                    # noqa: BLE001 - a log is a convenience
        LOGBOOK = None
    print_banner()
    if not SERVE.is_file():
        die("start.bat must sit next to tools\\serve_openai.py")

    if not ENV_FILE.is_file():
        if not ENV_EXAMPLE.is_file():
            die(".env.example is missing; cannot create .env")
        shutil.copyfile(ENV_EXAMPLE, ENV_FILE)
        info("Created .env from .env.example  (16 GB NVIDIA defaults)")

    cfg = load_dotenv(ENV_FILE)
    if cfg.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = cfg["HF_TOKEN"]

    scripts = str(ROOT / ".venv" / "Scripts")
    os.environ["PATH"] = scripts + os.pathsep + os.environ.get("PATH", "")

    cc, drv = nvidia_cc_driver()
    if cc and cc < 7.5:
        die(f"This GPU (compute capability {cc}) is older than Turing; the CUDA 12.8/13 PyTorch builds\n"
            "carry no kernels for it, so the kit cannot run here. Needs an RTX 20-series / T4 or newer.")
    if cc and cc < 8.0:
        warn(f"Turing GPU (sm_{int(cc*10)}): supported, roughly half the speed of Ampere or newer.")

    # ---- first run -------------------------------------------------------
    # One page in the browser covers the profile menu, the install and the
    # weights download; `SETUP=console` keeps the old flow in this window.
    sys.path.insert(0, str(ROOT / "tools"))
    import setup_core

    want_profile = any(a.lower() in ("profile", "--profile", "setup", "--setup")
                       for a in sys.argv[1:])
    needed, reasons = setup_core.needs_setup(cfg)
    if needed or want_profile:
        port = cfg.get("PORT", "8888")
        outcome, done = ("unavailable", None)
        if setup_mode(cfg) == "browser":
            outcome, done = run_setup_web(cfg, port, reasons, force_profile=want_profile)
        if outcome == "cancelled":
            die("Setup was stopped in the browser. Nothing was lost - starting Simplex\n"
                "again picks the download up where it left off.", 3)
        if outcome == "error":
            die("Setup could not finish. The setup page explained why, and the same\n"
                "reason is in the log. Fix that and start Simplex again.", 4)
        if outcome == "unavailable":
            # console fallback: the same two stages, asked here instead
            if want_profile or not cfg.get("PROFILE") or cfg["PROFILE"].strip().lower() == "ask":
                try:
                    import profiles
                    rc = profiles.run(force=want_profile)
                except KeyboardInterrupt:
                    rc = 130
                if rc != 0:
                    die("no profile chosen - edit .env by hand or run start.bat again", rc)
            cfg = load_dotenv(ENV_FILE)
            if not venv_ok():
                bootstrap(cfg)
        else:
            cfg = done
        if cfg.get("HF_TOKEN"):
            os.environ["HF_TOKEN"] = cfg["HF_TOKEN"]

    if VENV_PY.is_file():
        ensure_triton()
        ensure_pillow()

    if not venv_ok():
        die(
            "The Python environment is still incomplete. Typical causes on Windows 11:\n"
            "  - Visual Studio Build Tools with the C++ workload not installed\n"
            "    (a prebuilt engine wheel in the wheels\\ folder avoids needing them)\n"
            "  - NVIDIA CUDA Toolkit missing (nvcc not on PATH)\n"
            "  - PyTorch CPU-only wheel (set TORCH_INDEX_URL in .env)\n"
            "Delete .venv and run start.bat again after fixing that."
        )

    require_engine_version()

    cmd, host, port, gpu_mem, ui, context = server_command(cfg)

    # Chat app: download / initialise / configure BEFORE the long model load,
    # so the console stays readable and the offer after Ready is instant.
    mode = cherry_mode(cfg)
    prep = None
    if mode != "no":
        print()
        prep = prepare_cherry(cfg, port, context)

    # Free-VRAM check right before the load (Cherry's first-run init above is
    # closed again by now, so what is left in use is other apps).
    print()
    vram_preflight(gpu_mem)

    print()
    info("Loading the model now. First start compiles kernels (a few minutes).")
    info("The Ready box appears after load finishes  - do not connect yet.")
    if ui != "no":
        info(f"The chat UI will be at http://127.0.0.1:{port}/ once it does.")
    if prep is not None:
        info("After Ready you will be asked whether to open Cherry Studio.")
    print()
    env = dict(os.environ)
    env["SIMPLEX_SUPERVISED"] = "1"     # lets the UI offer "switch model"
    # On a pipe, Python falls back to the ANSI code page; this side decodes
    # UTF-8, so say so rather than letting an accented path arrive as mojibake.
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    # Windows start.bat never `source`s .env. CUDA_VISIBLE_DEVICES in .env is
    # otherwise a comment CUDA never sees, and CUDA's default order is
    # fastest-first (5090 before 5080). Copy the pin into the child.
    if cfg.get("CUDA_DEVICE_ORDER"):
        env["CUDA_DEVICE_ORDER"] = cfg["CUDA_DEVICE_ORDER"]
    elif cfg.get("CUDA_VISIBLE_DEVICES"):
        env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    if cfg.get("CUDA_VISIBLE_DEVICES"):
        env["CUDA_VISIBLE_DEVICES"] = cfg["CUDA_VISIBLE_DEVICES"]

    rt = Runtime()
    rt.url = f"http://127.0.0.1:{port}/"
    tray = start_tray(rt, LOGBOOK.path if LOGBOOK is not None else None) \
        if tray_mode(cfg) else None
    if tray is not None:
        info("Simplex is in the notification area - right-click it for the menu.")

    first = True
    try:
        while True:
            rt.restart = False
            # The child's output is read here rather than left to the console,
            # so the same lines reach the log file the tray menu opens.
            proc = subprocess.Popen(
                cmd, cwd=str(ROOT), env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
            rt.proc = proc
            pump = threading.Thread(target=_pump_child, args=(proc,), daemon=True)
            pump.start()
            if ui != "no":
                threading.Thread(
                    target=ui_after_ready,
                    args=(proc, host, port, ui == "browser" and first),
                    kwargs={"on_ready": _on_ready(cfg, tray)},
                    daemon=True,
                ).start()
            if prep is not None and first:
                threading.Thread(
                    target=cherry_after_ready, args=(proc, prep, host, port, mode),
                    daemon=True,
                ).start()
            first = False
            if tray is not None:
                tray.set_tip(f"Simplex - {cfg.get('MODEL_ID', 'loading')}")
            try:
                while True:   # short waits so Ctrl+C is noticed promptly on Windows
                    try:
                        code = proc.wait(timeout=1)
                        break
                    except subprocess.TimeoutExpired:
                        pass
            except KeyboardInterrupt:
                # Ctrl+C reaches the server too (same console); give it a moment.
                # BaseException, not Exception: a second Ctrl+C lands here, and
                # skipping terminate() left a server holding the GPU and the port.
                try:
                    proc.wait(timeout=10)
                except BaseException:  # noqa: BLE001
                    try:
                        proc.terminate()
                    except Exception:  # noqa: BLE001
                        pass
                pump.join(timeout=3)
                raise
            # proc.wait() returns while the pipe may still hold the child's last
            # words - which is exactly when they matter, because those are the
            # lines that say why it stopped.
            pump.join(timeout=5)
            if rt.quit:
                info("Quitting - asked from the tray menu.")
                return 0
            if code != RESTART_CODE and not rt.restart:
                return code
            if rt.restart:
                print()
                info("Restarting the model - asked from the tray menu.")
                print()
            else:
                # The UI asked for another model: .env has the new settings, so
                # rebuild the command line and load again in this same window.
                cfg = load_dotenv(ENV_FILE)
                cmd, host, port, gpu_mem, ui, context = server_command(cfg)
                rt.url = f"http://127.0.0.1:{port}/"   # a switch may change PORT
                print()
                info(f"Switching to {cfg.get('MODEL_ID', 'the new model')} "
                     f"({cfg.get('CONTEXT_SIZE', '?')} ctx) - reloading ...")
                print()
            vram_preflight(gpu_mem)
            print()
    finally:
        if tray is not None:
            tray.stop()


def _crash(exc: BaseException) -> int:
    """What someone who double-clicked an icon should see: one sentence about
    what happened, one about what to do, and where the trace was written."""
    import traceback
    trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    if LOGBOOK is not None:
        LOGBOOK.write_raw("\n" + trace)
    what, todo = ("Simplex stopped because of an unexpected error.", "")
    try:
        sys.path.insert(0, str(ROOT / "tools"))
        import logbook
        what, todo = logbook.explain(exc)
    except Exception:                    # noqa: BLE001
        pass
    print(file=sys.stderr)
    print(red("  Simplex could not start"), file=sys.stderr)
    print(f"    {what}", file=sys.stderr)
    if todo:
        for line in todo.split(". "):
            if line.strip():
                print(f"    {line.strip().rstrip('.')}.", file=sys.stderr)
    if LOGBOOK is not None and LOGBOOK.path:
        print(f"    The full details are in {LOGBOOK.path}", file=sys.stderr)
    else:
        print(file=sys.stderr)
        print(trace, file=sys.stderr)
    print(file=sys.stderr)
    return 1


if __name__ == "__main__":
    try:
        code = main()
    except KeyboardInterrupt:
        print("\n  " + dim("Stopped."))
        code = 0
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    except BaseException as e:            # noqa: BLE001 - the last line of defence
        code = _crash(e)
    finally:
        if LOGBOOK is not None:
            LOGBOOK.stop()
    sys.exit(code)
