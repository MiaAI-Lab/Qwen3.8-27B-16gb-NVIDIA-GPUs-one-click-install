"""Prebuilt wheels, so a first run does not need a C++ compiler.

The expensive, fragile part of setup is compiling ExLlamaV3's CUDA kernels:
it wants the NVIDIA CUDA Toolkit and Visual Studio Build Tools, several GB of
downloads that have nothing to do with chatting to a model. A wheel built once
per (Python version x CUDA version) removes all of it.

Sources are tried in this order, and the first that yields a matching wheel wins:

  1. `wheels/` next to start.bat - what the installer drops in, or what a
     user copies off a USB stick on a machine with no internet.
  2. WHEEL_INDEX in .env - one or more `pip --find-links` targets (a GitHub
     Releases page, a file share, an internal index).
  3. PyPI - triton-windows lives there; exllamav3 does not.
  4. Compiling from source, which is what the kit did before this module.

Nothing here trusts a filename blindly: a wheel is only offered to pip when
its Python tag, ABI tag and platform tag match the interpreter that will run
it, so a cp312 wheel can never be installed into a cp313 venv.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WHEEL_DIR = ROOT / "wheels"

# Wheel filename: name-version(-build)?-pytag-abitag-plattag.whl
_WHEEL_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9_.\-]+?)-(?P<ver>[0-9][^-]*)"
    r"(?:-(?P<build>[0-9][^-]*))?"
    r"-(?P<py>[^-]+)-(?P<abi>[^-]+)-(?P<plat>[^-]+)\.whl$", re.IGNORECASE)


@dataclass
class Wheel:
    path: Path
    name: str
    version: str
    py: str
    abi: str
    plat: str

    @property
    def canonical(self) -> str:
        return re.sub(r"[-_.]+", "-", self.name).lower()


def parse_wheel(path: Path) -> Wheel | None:
    m = _WHEEL_RE.match(path.name)
    if not m:
        return None
    return Wheel(path, m["name"], m["ver"], m["py"], m["abi"], m["plat"])


# ------------------------------------------------------------------ tags -----

def interpreter_tags(python: Path | str) -> dict:
    """Ask the interpreter that will *host* the wheel what it can accept.
    Never guessed from this process: the venv may be a different Python."""
    code = (
        "import sys, sysconfig, json\n"
        "v = sys.version_info\n"
        "print(json.dumps({'py': 'cp%d%d' % (v.major, v.minor),\n"
        "                  'nodot': '%d%d' % (v.major, v.minor),\n"
        "                  'abi': (sysconfig.get_config_var('SOABI') or ''),\n"
        "                  'plat': sysconfig.get_platform().replace('-', '_').replace('.', '_'),\n"
        "                  'bits': 64 if sys.maxsize > 2**32 else 32}))\n"
    )
    try:
        r = subprocess.run([str(python), "-c", code], capture_output=True, text=True, timeout=60)
        if r.returncode == 0 and r.stdout.strip():
            import json
            return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:                       # noqa: BLE001 - fall through to this process
        pass
    v = sys.version_info
    import sysconfig
    return {"py": f"cp{v.major}{v.minor}", "nodot": f"{v.major}{v.minor}",
            "abi": sysconfig.get_config_var("SOABI") or "",
            "plat": sysconfig.get_platform().replace("-", "_").replace(".", "_"),
            "bits": 64 if sys.maxsize > 2 ** 32 else 32}


def wheel_matches(w: Wheel, tags: dict) -> bool:
    """True when this interpreter could actually import the wheel."""
    plat = (tags.get("plat") or "").lower()
    if w.plat.lower() not in ("any", plat):
        # win_amd64 wheels are also published as win32/win_amd64 pairs; only an
        # exact platform match or a pure-python wheel is safe.
        return False
    abi = w.abi.lower()
    if abi == "abi3":
        # A stable-ABI wheel runs on the minor it was built for and every
        # later one, so the Python tag is a floor rather than an equality.
        m = re.match(r"cp(\d)(\d+)$", w.py)
        return bool(m) and int(tags["nodot"]) >= int(m.group(1) + m.group(2))
    py_ok = any(t in (tags["py"], "py3", f"py{tags['nodot'][0]}")
                for t in w.py.split("."))
    if not py_ok:
        return False
    if abi == "none":
        return True
    return abi.startswith(tags["py"])


def local_wheels(tags: dict, folder: Path = WHEEL_DIR) -> list[Wheel]:
    if not folder.is_dir():
        return []
    found = []
    for p in sorted(folder.rglob("*.whl")):
        w = parse_wheel(p)
        if w and wheel_matches(w, tags):
            found.append(w)
    return found


def find_local(package: str, tags: dict, folder: Path = WHEEL_DIR) -> Wheel | None:
    want = re.sub(r"[-_.]+", "-", package).lower()
    hits = [w for w in local_wheels(tags, folder) if w.canonical == want]
    if not hits:
        return None
    # newest version wins; ties broken by filename so the choice is stable
    def key(w: Wheel):
        parts = re.findall(r"\d+", w.version)
        return ([int(x) for x in parts[:4]], w.path.name)
    return sorted(hits, key=key)[-1]


# ---------------------------------------------------------------- CUDA -------

def cuda_tag(driver_index_url: str = "") -> str:
    """'cu130' / 'cu128' - which CUDA build line the kit is installing.
    Read from the torch index URL the launcher already computed, because that
    is the single place the decision is made."""
    m = re.search(r"/(cu\d{3})\b", driver_index_url or "")
    return m.group(1) if m else ""


# ------------------------------------------------------------- pip plans -----

def index_urls(cfg: dict) -> list[str]:
    raw = (cfg.get("WHEEL_INDEX") or os.environ.get("WHEEL_INDEX") or "").strip()
    if not raw:
        return []
    return [u.strip() for u in re.split(r"[,\s]+", raw) if u.strip()]


def prebuilt_args(package: str, tags: dict, cfg: dict, folder: Path = WHEEL_DIR) -> list[str] | None:
    """pip arguments that install `package` from a prebuilt wheel and refuse
    to fall back to a source build, or None when no source can offer one.

    Returned as arguments only - running pip is the caller's job, so this
    stays testable on a machine with no network and no venv."""
    local = find_local(package, tags, folder)
    links = index_urls(cfg)
    if local is None and not links:
        return None
    args = ["install", "--only-binary", ":all:", "--no-build-isolation"]
    if local is not None:
        # An exact path is unambiguous: pip cannot decide it prefers something
        # else, and the version in the folder is the version installed.
        args += ["--no-index", str(local.path)]
        return args
    for url in links:
        args += ["--find-links", url]
    args.append(package)
    return args


def describe(package: str, tags: dict, cfg: dict, folder: Path = WHEEL_DIR) -> str:
    local = find_local(package, tags, folder)
    if local is not None:
        return f"{local.path.name}  (from wheels\\)"
    links = index_urls(cfg)
    if links:
        return f"{package} from {links[0]}"
    return ""


# --------------------------------------------------------------- toolchain ---

def have_compiler() -> bool:
    """A source build needs cl.exe (Windows) or a C++ compiler (POSIX)."""
    from shutil import which
    if sys.platform == "win32":
        if which("cl"):
            return True
        # VS is usually not on PATH until vcvars runs; vswhere tells us anyway
        vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / \
            "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
        if vswhere.is_file():
            try:
                r = subprocess.run(
                    [str(vswhere), "-latest", "-products", "*", "-requires",
                     "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                     "-property", "installationPath"],
                    capture_output=True, text=True, timeout=30)
                return bool(r.stdout.strip())
            except Exception:               # noqa: BLE001
                return False
        return False
    return bool(which("g++") or which("clang++"))


def have_cuda_toolkit() -> bool:
    from shutil import which
    if which("nvcc"):
        return True
    for key in ("CUDA_HOME", "CUDA_PATH"):
        v = os.environ.get(key)
        if v and (Path(v) / "bin").is_dir():
            return True
    base = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    return base.is_dir() and any(p.is_dir() for p in base.iterdir())


def source_build_blockers() -> list[str]:
    """Plain-English list of what a source build is missing, for the setup page."""
    out = []
    if not have_compiler():
        out.append("Visual Studio Build Tools with the \"Desktop development with C++\" workload")
    if not have_cuda_toolkit():
        out.append("NVIDIA CUDA Toolkit")
    return out


def main(argv: list[str]) -> int:
    import argparse, json
    ap = argparse.ArgumentParser(description="What prebuilt wheels can this Python use?")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--package", default="exllamav3")
    ap.add_argument("--folder", default=str(WHEEL_DIR))
    a = ap.parse_args(argv)
    tags = interpreter_tags(a.python)
    folder = Path(a.folder)
    print(json.dumps({
        "tags": tags,
        "folder": str(folder),
        "wheels_here": [w.path.name for w in local_wheels(tags, folder)],
        "match": (lambda w: w.path.name if w else None)(find_local(a.package, tags, folder)),
        "blockers": source_build_blockers(),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
