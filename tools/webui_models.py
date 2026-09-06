#!/usr/bin/env python3
"""Which models are downloaded, and switching between them.

A quant is loaded into VRAM at startup and sized by the settings in `.env`, so
switching is not a runtime toggle: the UI writes the new settings and the
server restarts into them (exit code 87, which the launcher treats as "start
me again"). One model at a time is the whole point - two would not fit.

The context size for the model being switched to is recomputed with the same
planner `tools/profiles.py` uses for the first-run menu, so moving from 2.0bpw
to 5.0bpw shortens the context instead of failing to load. When profiles.py or
`nvidia-smi` is unavailable the current settings are kept unchanged and only
the model itself is swapped.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

RESTART_CODE = 87        # what the launcher watches for


class Incomplete(ValueError):
    """The folder is there but the weights are not."""
MODELS_DIR = "models"
WEIGHT_SUFFIXES = (".safetensors", ".gguf", ".bin")


def _profiles():
    """tools/profiles.py, or None if this kit predates it."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        import profiles                                # noqa: WPS433
        return profiles
    except Exception:                                  # noqa: BLE001
        return None


def _size_gb(folder: Path) -> float:
    total = 0
    try:
        for f in folder.rglob("*"):
            if f.is_file() and f.suffix in WEIGHT_SUFFIXES:
                total += f.stat().st_size
    except OSError:
        return 0.0
    return round(total / 1e9, 1)


def _bpw(name: str):
    m = re.search(r"(\d+(?:\.\d+)?)\s*bpw", name, re.I)
    return float(m.group(1)) if m else None


def _quant_for(name: str):
    """Match a folder name against profiles.py's quant table."""
    p = _profiles()
    bpw = _bpw(name)
    if not p or bpw is None:
        return None
    for q in p.QUANTS:
        if abs(q.bpw - bpw) < 0.01:
            return q
    return None


def installed(root: Path, current_dir: str = "") -> list[dict]:
    """Every model folder that actually holds weights, best quality first."""
    folder = Path(root) / MODELS_DIR
    current = str(Path(current_dir).name).lower() if current_dir else ""
    out = []
    if not folder.is_dir():
        return out
    for entry in sorted(folder.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        has_weights = any(f.suffix in WEIGHT_SUFFIXES for f in entry.glob("*"))
        if not has_weights and not (entry / "config.json").is_file():
            continue
        quant = _quant_for(entry.name)
        row = {
            "complete": has_weights,
            "dir": f"{MODELS_DIR}/{entry.name}",
            "id": entry.name.lower(),
            "name": entry.name,
            "bpw": _bpw(entry.name),
            "quality": quant.note if quant else "",
            "kl": quant.kl if quant else None,
            "size_gb": _size_gb(entry),
            "current": entry.name.lower() == current,
            "fits": True, "why": "", "context": None,
        }
        # a downloaded quant is not necessarily a loadable one on this card
        if not has_weights:
            row["fits"] = False
            row["why"] = ("only part of this model is on disk - start.bat will "
                          "finish the download")
            out.append(row)
            continue
        try:
            planned = settings_for(root, row["dir"], {})
            row["context"] = int(planned.get("CONTEXT_SIZE") or 0) or None
        except ValueError as e:
            row["fits"], row["why"] = False, str(e)
        except Exception:                              # noqa: BLE001
            pass
        out.append(row)
    out.sort(key=lambda m: (m["bpw"] or 0), reverse=True)
    return out


def settings_for(root: Path, model_dir: str, cfg: dict) -> dict:
    """The .env changes that switch to `model_dir`, context recomputed to fit."""
    name = Path(model_dir).name
    updates = {"MODEL_DIR": model_dir.replace("\\", "/"), "MODEL_ID": name.lower()}
    p, quant = _profiles(), _quant_for(name)
    if not p or quant is None:
        return updates                       # unknown quant: keep the rest as-is
    try:
        gpu = p.detect_gpu(cfg)
        # profiles.GPU exposes total_gib, not .memory. Reading the wrong name
        # used to raise and land in the except below, which is what actually
        # produced this function's documented "no nvidia-smi" behaviour. Reading
        # the right name does NOT raise when there is no card - detect_gpu()
        # hands back a GPU with total_gib 0.0 - so the fallback has to be an
        # explicit check, or a machine with no readable GPU plans a -1.3 GiB
        # budget and reports every model as "does not fit in 0 GB of VRAM".
        total = float(gpu.total_gib)
    except Exception:                        # noqa: BLE001
        return updates
    if total <= 0:                           # no card, or nvidia-smi timed out
        return updates
    cache = "4"
    budget = p.budget_gib(total)
    # An explicit VISION=off is the user's answer from the setup menu; a switch
    # of model must not quietly turn images back on. "auto" means no answer was
    # given, so the planner's own rule decides.
    configured = str(cfg.get("VISION", "auto")).strip().lower()
    want_vision = False if configured in ("off", "0", "false", "no") else None
    # Same rule as the first-run menu, from the same function - see plan_one().
    picked = p.plan_one(quant, cache, budget, want_vision, floor=8192)
    if picked is None:
        raise ValueError(f"{name} does not fit in {total:.0f} GB of VRAM "
                         f"(needs more than this card has)")
    ctx, vision = picked
    need = p.need_gib(quant, ctx, cache, vision)
    updates.update({
        "PROFILE": f"{quant.id}bpw-{round(ctx / 1000)}k",
        "CONTEXT_SIZE": str(ctx),
        "CACHE_QUANT": cache,
        "GPU_MEM_GB": f"{min(budget, need + 1.5):.1f}",
        "VISION": "auto" if vision else "off",
        "HF_TARGET_REPO": quant.repo,
        "HF_REVISION": quant.revision,
    })
    return updates


def apply(root: Path, model_dir: str, cfg: dict) -> dict:
    """Validate, write .env, and return what the next start will use."""
    root = Path(root)
    target = (root / model_dir).resolve()
    models_root = (root / MODELS_DIR).resolve()
    if models_root not in target.parents:
        raise ValueError("only folders under models/ can be loaded")
    if not target.is_dir():
        raise ValueError(f"{model_dir} is not there any more")
    if not any(f.suffix in WEIGHT_SUFFIXES for f in target.glob("*")):
        raise Incomplete(f"{Path(model_dir).name} has no weight files - it looks "
                         f"like an interrupted download. Run start.bat to finish "
                         f"it before loading it.")
    updates = settings_for(root, model_dir, cfg)
    p = _profiles()
    if p is None:
        raise ValueError("tools/profiles.py is missing, so .env cannot be written")
    p.write_env(root / ".env", updates)
    return updates
