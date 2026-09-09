"""Prebuilt wheels, so a first run does not need a C++ compiler.

The expensive, fragile part of setup is compiling ExLlamaV3's CUDA kernels:
it wants the NVIDIA CUDA Toolkit and Visual Studio Build Tools, several GB of
downloads that have nothing to do with chatting to a model. A wheel built once
per (Python version x CUDA version) removes all of it.

Sources are tried in this order, and the first that yields a matching wheel wins:

  1. `wheels/` next to windows\\start.bat - what the installer drops in, or what a
     user copies off a USB stick on a machine with no internet.
  2. The engine's own GitHub release. turboderp-org/exllamav3 attaches a wheel
     per (CUDA line x torch version x Python), so the normal case needs no
     wheel-building by anyone. See ENGINE_WHEELS below for why this is an
     exact URL rather than a --find-links page.
  3. WHEEL_INDEX in .env - one or more `pip --find-links` targets (a GitHub
     Releases page, a file share, an internal index).
  4. PyPI - triton-windows lives there; exllamav3 does not.
  5. Compiling from source, which is what the kit did before this module.

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
    text = driver_index_url or ""
    m = re.search(r"/(cu\d{3})\b", text) or re.fullmatch(r"(cu\d{3})", text.strip())
    return m.group(1) if m else ""


# --------------------------------------------------- the engine's release -----

# What turboderp-org/exllamav3 v1.4.4 actually published. Written down rather
# than discovered, because setup has to be able to say what it will do before
# it has a network - and because the answer has to be exact.
#
# Why exact: the CUDA line and the torch version live in the wheel's *local
# version* ("1.4.4+cu128.torch2.10.0"), which pip does not match against
# anything. Point pip at the release with --find-links and it filters on the
# Python and platform tags only, then takes the highest version string - so a
# torch 2.10 venv is happily handed the torch2.11 build, and the failure comes
# later as an undefined-symbol ImportError that reads like a corrupt install.
# So the wheel is resolved here and pip is given one URL it cannot argue with.
ENGINE_PACKAGE = "exllamav3"
ENGINE_VERSION = "1.4.4"
ENGINE_RELEASE = ("https://github.com/turboderp-org/exllamav3/releases/download/"
                  f"v{ENGINE_VERSION}/")

# cuda line -> torch version -> the cp tags built for it (same on both platforms)
ENGINE_WHEELS: dict[str, dict[str, tuple[str, ...]]] = {
    "cu128": {
        "2.7.0":  ("cp310", "cp311", "cp312", "cp313"),
        "2.8.0":  ("cp310", "cp311", "cp312", "cp313"),
        "2.9.0":  ("cp310", "cp311", "cp312", "cp313", "cp314"),
        "2.10.0": ("cp310", "cp311", "cp312", "cp313", "cp314"),
        "2.11.0": ("cp312", "cp313", "cp314"),
    },
    "cu132": {
        "2.11.0": ("cp312", "cp313", "cp314"),
    },
}
# Only these platforms are built. A wheel for anything else does not exist, so
# the source build stays the answer there (aarch64 / GB10, for one).
ENGINE_PLATFORMS = ("win_amd64", "linux_x86_64")


def _ver_key(v: str) -> tuple:
    """'2.10.0' sorts above '2.9.0' - which it does not as a string."""
    return tuple(int(x) if x.isdigit() else 0 for x in v.split("."))


def preferred_torch(tags: dict, cuda: str) -> str:
    """The newest torch this engine release has a wheel for, on this Python and
    CUDA line - or "" when it has none and torch should float.

    This exists because the two ceilings do not have to agree. The cu128 index
    serves torch 2.11 to every Python it supports, but the engine only builds
    2.11 for cp312+; a 3.10 or 3.11 user therefore gets a torch no engine wheel
    matches and compiles for twenty minutes with no clue why. Asking the same
    table that picks the wheel which torch to install keeps the two from
    drifting - including on the day PyTorch adds 2.12 to cu128.
    """
    if (tags.get("plat") or "").lower() not in ENGINE_PLATFORMS:
        return ""                      # no wheel on any torch here; do not cap
    builds = ENGINE_WHEELS.get(cuda) or {}
    usable = [v for v, cps in builds.items() if tags.get("py") in cps]
    return max(usable, key=_ver_key) if usable else ""


def torch_requirement(tags: dict, cuda: str) -> str:
    """'torch==2.11.0' when capping buys a prebuilt engine, else plain 'torch'."""
    want = preferred_torch(tags, cuda)
    return f"torch=={want}" if want else "torch"


def torch_build(python: Path | str) -> tuple[str, str]:
    """('2.10.0', 'cu128') for the torch already installed in that venv.

    Read from the interpreter rather than from the index URL we asked for:
    pip resolves `torch` to whatever the index currently offers, so the version
    is only knowable after the fact, and it is the installed build that the
    engine wheel has to match."""
    code = ("import torch, sys\n"
            "v = torch.__version__\n"
            "base, _, local = v.partition('+')\n"
            "sys.stdout.write(base + ' ' + local)\n")
    try:
        r = subprocess.run([str(python), "-c", code], capture_output=True, text=True, timeout=120)
        if r.returncode == 0 and r.stdout.strip():
            base, _, local = r.stdout.strip().partition(" ")
            m = re.match(r"(cu\d{3})", local or "")
            return base, (m.group(1) if m else "")
    except Exception:                       # noqa: BLE001 - no torch yet, or no venv
        pass
    return "", ""


def engine_wheel_url(tags: dict, torch_version: str, cuda: str) -> str | None:
    """The one wheel that fits this venv, or None when the matrix has no entry.

    None is a normal answer - an unbuilt CUDA line, a platform nobody ships,
    a torch newer than the release - and the caller falls back to a source
    build rather than installing something that will not import."""
    plat = (tags.get("plat") or "").lower()
    if plat not in ENGINE_PLATFORMS or not torch_version or not cuda:
        return None
    builds = ENGINE_WHEELS.get(cuda)
    if not builds:
        return None
    cps = builds.get(torch_version)
    if not cps or tags.get("py") not in cps:
        return None
    py = tags["py"]
    return (f"{ENGINE_RELEASE}exllamav3-{ENGINE_VERSION}+{cuda}.torch{torch_version}"
            f"-{py}-{py}-{plat}.whl")


# ------------------------------------------------------------- pip plans -----

def index_urls(cfg: dict) -> list[str]:
    raw = (cfg.get("WHEEL_INDEX") or os.environ.get("WHEEL_INDEX") or "").strip()
    if not raw:
        return []
    return [u.strip() for u in re.split(r"[,\s]+", raw) if u.strip()]


def prebuilt_args(package: str, tags: dict, cfg: dict, folder: Path = WHEEL_DIR,
                  venv_python: Path | str | None = None) -> list[str] | None:
    """pip arguments that install `package` from a prebuilt wheel and refuse
    to fall back to a source build, or None when no source can offer one.

    Returned as arguments only - running pip is the caller's job, so this
    stays testable on a machine with no network and no venv."""
    local = find_local(package, tags, folder)
    if local is not None:
        # An exact path is unambiguous: pip cannot decide it prefers something
        # else, and the version in the folder is the version installed.
        return ["install", "--only-binary", ":all:", "--no-build-isolation",
                "--no-index", str(local.path)]
    upstream = _engine_url_for(package, tags, venv_python)
    if upstream:
        # Also exact, and for the same reason: see ENGINE_WHEELS.
        return ["install", "--only-binary", ":all:", "--no-build-isolation", upstream]
    links = index_urls(cfg)
    if not links:
        return None
    args = ["install", "--only-binary", ":all:", "--no-build-isolation"]
    for url in links:
        args += ["--find-links", url]
    args.append(package)
    return args


def _engine_url_for(package: str, tags: dict,
                    venv_python: Path | str | None) -> str | None:
    """The engine's own release wheel for this venv, when that is what is asked
    for and the venv already has a torch the release was built against."""
    if package.lower().replace("_", "-") != ENGINE_PACKAGE or venv_python is None:
        return None
    torch_version, cuda = torch_build(venv_python)
    return engine_wheel_url(tags, torch_version, cuda)


def describe(package: str, tags: dict, cfg: dict, folder: Path = WHEEL_DIR,
             venv_python: Path | str | None = None) -> str:
    local = find_local(package, tags, folder)
    if local is not None:
        return f"{local.path.name}  (from wheels\\)"
    upstream = _engine_url_for(package, tags, venv_python)
    if upstream:
        return f"{upstream.rsplit('/', 1)[-1]}  (from the engine's own release)"
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
    ap.add_argument("--torch-req", action="store_true",
                    help="print the torch requirement to install ('torch==2.11.0' or "
                         "'torch') for --cuda; what the shell launcher uses")
    ap.add_argument("--cuda", default="",
                    help="the torch index URL or a bare cuXXX tag, for --torch-req")
    ap.add_argument("--url", action="store_true",
                    help="print just the engine wheel URL for --python (empty if none) "
                         "and exit; what the shell launcher uses")
    a = ap.parse_args(argv)
    tags = interpreter_tags(a.python)
    folder = Path(a.folder)
    if a.url:
        torch_version, cuda = torch_build(a.python)
        print(engine_wheel_url(tags, torch_version, cuda) or "")
        return 0
    if a.torch_req:
        print(torch_requirement(tags, cuda_tag(a.cuda)))
        return 0
    torch_version, cuda = torch_build(a.python)
    print(json.dumps({
        "torch": torch_version, "cuda": cuda,
        "engine_wheel": engine_wheel_url(tags, torch_version, cuda),

        "tags": tags,
        "folder": str(folder),
        "wheels_here": [w.path.name for w in local_wheels(tags, folder)],
        "match": (lambda w: w.path.name if w else None)(find_local(a.package, tags, folder)),
        "blockers": source_build_blockers(),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
