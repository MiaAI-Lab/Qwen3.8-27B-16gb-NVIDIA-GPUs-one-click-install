#!/usr/bin/env python3
"""Hot-patch the installed ExLlamaV3 v1.4.4 with the fp8 / nvfp4 KV-cache lanes.

The lanes come from MiaAI-Lab/exllamav3 (their v1.4.2-based fork). The feature
is pure Python + Triton and touches only files that are identical between
upstream v1.4.2 and v1.4.4, so it is applied straight into the venv's
site-packages - no CUDA recompile:

    exllamav3/cache/fp8.py, cache/nvfp4.py            (new)
    exllamav3/cache/__init__.py                        (+2 lines)
    exllamav3/modules/attention_fn/common.py           (+2 lines: k/v scales)
    exllamav3/modules/attention_fn/dispatch.py         (fp8/nvfp4 fast path)
    exllamav3/modules/attention_fn/triton_paged.py     (online-dequant kernels)
    exllamav3/model_init.py                            (-cq fp8|nvfp4; draft cache int 8,4)

How: patches/kvcache-fp8-nvfp4-v1.4.4/ holds the complete patched files plus a
manifest with SHA-256 of the stock v1.4.4 files. `apply` verifies every target is
stock (line endings ignored - Windows checkouts are CRLF), backs it up, then
copies the patched file over it in the same line-ending style. No git involved:
`git apply` silently ignores every path when the venv happens to live inside a
git work tree, which is exactly what bit the first Windows run.

Usage (from the kit root):
    .venv\\Scripts\\python.exe tools\\patch_kv.py status
    .venv\\Scripts\\python.exe tools\\patch_kv.py apply
    .venv\\Scripts\\python.exe tools\\patch_kv.py revert

start.bat / start.sh call `apply` automatically when CACHE_QUANT is fp8 or nvfp4.
(patches/kvcache-fp8-nvfp4-v1.4.4.patch is the same change as a unified diff, for
people who build the engine from a checkout.)
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATCH_DIR = ROOT / "patches" / "kvcache-fp8-nvfp4-v1.4.4"
MANIFEST = PATCH_DIR / "manifest.json"
REQUIRED_VERSION = "1.4.4"
BACKUP_NAME = ".kvpatch-backup"
MARKER_NAME = ".kvpatch-applied"


def site_packages() -> Path:
    """Directory that contains the installed exllamav3/ package (no import side effects)."""
    spec = importlib.util.find_spec("exllamav3")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("exllamav3 is not installed in this Python (run start.bat first)")
    pkg = Path(list(spec.submodule_search_locations)[0]).resolve()
    return pkg.parent


def installed_version(sp: Path) -> str:
    try:
        for line in (sp / "exllamav3" / "version.py").read_text(encoding="utf-8").splitlines():
            if "__version__" in line and "=" in line:
                return line.split("=", 1)[1].strip().strip("\"'")
    except OSError:
        pass
    return "unknown"


def sha_norm(p: Path) -> str:
    """SHA-256 of the file with CR stripped (CRLF and LF checkouts hash the same)."""
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load_manifest() -> dict:
    if not MANIFEST.is_file():
        raise SystemExit(f"patch payload missing: {MANIFEST}")
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for rel, e in m["files"].items():
        src = PATCH_DIR / rel
        if not src.is_file():
            raise SystemExit(f"patch payload missing: {src}")
        if sha_norm(src) != e["patched_sha256"]:
            raise SystemExit(f"patch payload corrupted: {rel}")
    return m


def file_state(sp: Path, rel: str, e: dict) -> str:
    """'stock' | 'patched' | 'missing' | 'other' for one file."""
    p = sp / rel
    if not p.is_file():
        return "missing"
    h = sha_norm(p)
    if h == e["patched_sha256"]:
        return "patched"
    if e["kind"] == "modified" and h == e["stock_sha256"]:
        return "stock"
    return "other"


def state(sp: Path, m: dict) -> tuple[str, dict]:
    per = {rel: file_state(sp, rel, e) for rel, e in m["files"].items()}
    vals = set(per.values())
    if vals == {"patched"}:
        return "applied", per
    if all(s == "stock" for rel, s in per.items() if m["files"][rel]["kind"] == "modified") and \
       all(s == "missing" for rel, s in per.items() if m["files"][rel]["kind"] == "added"):
        return "clean", per
    return "mixed", per


def clear_pycache(sp: Path) -> None:
    for rel in ("exllamav3", "exllamav3/cache", "exllamav3/modules/attention_fn"):
        pc = sp / rel / "__pycache__"
        if pc.is_dir():
            shutil.rmtree(pc, ignore_errors=True)


def write_like(src: Path, dst: Path, crlf: bool) -> None:
    data = src.read_bytes().replace(b"\r\n", b"\n")
    if crlf:
        data = data.replace(b"\n", b"\r\n")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(data)


def cmd_status(sp: Path, m: dict) -> int:
    st, per = state(sp, m)
    print(f"exllamav3 {installed_version(sp)} at {sp / 'exllamav3'}")
    print(f"KV patch (fp8 / nvfp4 lanes): {st}")
    for rel, s in per.items():
        print(f"    {s:8s} {rel}")
    if st == "applied":
        print("fp8 / nvfp4 KV cache: AVAILABLE  (CACHE_QUANT=fp8 or nvfp4)")
        return 0
    if st == "clean":
        print("fp8 / nvfp4 KV cache: not installed  (run: tools\\patch_kv.py apply)")
        return 0
    print("some files are neither stock v1.4.4 nor patched: run `revert` (restores the backup)")
    print("or delete .venv and run start.bat again for a clean engine, then re-apply")
    return 2


def cmd_apply(sp: Path, m: dict) -> int:
    ver = installed_version(sp)
    if ver != REQUIRED_VERSION:
        raise SystemExit(f"this patch is for ExLlamaV3 {REQUIRED_VERSION}; installed: {ver}")
    st, per = state(sp, m)
    if st == "applied":
        print("already applied - nothing to do")
        return 0
    if st == "mixed":
        bad = [f"{s} {rel}" for rel, s in per.items() if s not in ("stock", "missing")]
        raise SystemExit("cannot apply: unexpected file state\n    " + "\n    ".join(bad)
                         + "\nrun `tools\\patch_kv.py revert`, or delete .venv and run start.bat again")
    backup = sp / "exllamav3" / BACKUP_NAME
    backup.mkdir(exist_ok=True)
    for rel, e in m["files"].items():
        if e["kind"] == "modified":
            shutil.copy2(sp / rel, backup / rel.replace("/", "__"))
    crlf = b"\r\n" in (sp / "exllamav3" / "model_init.py").read_bytes()
    for rel in m["files"]:
        write_like(PATCH_DIR / rel, sp / rel, crlf)
    clear_pycache(sp)
    st2, per2 = state(sp, m)
    if st2 != "applied":
        raise SystemExit("post-apply verification failed: " + str(per2))
    (sp / "exllamav3" / MARKER_NAME).write_text(m["engine_version"], encoding="utf-8")
    print(f"applied fp8/nvfp4 KV lanes to exllamav3 {ver}  ({len(m['files'])} files, backup: {backup})")
    print("CACHE_QUANT=fp8 and CACHE_QUANT=nvfp4 are now available")
    return 0


def cmd_revert(sp: Path, m: dict) -> int:
    st, per = state(sp, m)
    if st == "clean":
        print("not applied - nothing to revert")
        return 0
    backup = sp / "exllamav3" / BACKUP_NAME
    for rel, e in m["files"].items():
        if e["kind"] == "modified":
            b = backup / rel.replace("/", "__")
            if not b.is_file():
                raise SystemExit(f"no backup for {rel}; delete .venv and run start.bat again")
            shutil.copy2(b, sp / rel)
        else:
            (sp / rel).unlink(missing_ok=True)
    clear_pycache(sp)
    (sp / "exllamav3" / MARKER_NAME).unlink(missing_ok=True)
    st2, _ = state(sp, m)
    print("reverted to stock v1.4.4 (integer KV cache only)" if st2 == "clean"
          else f"reverted, but state is now {st2} - check `status`")
    return 0 if st2 == "clean" else 2


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "status"
    m = load_manifest()
    sp = site_packages()
    if cmd == "status":
        return cmd_status(sp, m)
    if cmd == "apply":
        return cmd_apply(sp, m)
    if cmd == "revert":
        return cmd_revert(sp, m)
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
