#!/usr/bin/env python3
"""Other OpenAI-compatible endpoints the UI can talk to.

The kit's own server is always there and always first. A provider is anything
else that speaks the same protocol: a second local runtime (llama.cpp, Ollama,
LM Studio, vLLM), a machine down the hall, or a hosted API. Adding one costs a
base URL, an optional key and a model id - and switching to it is instant,
because unlike a local quant nothing has to be loaded into this card's VRAM.

Providers live in `<root>/providers.json`, which holds API keys and is
gitignored. Keys never travel back to the browser: the UI sees `sk-...4f2a` and
sends an empty string to mean "leave the key alone".
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

FILE = "providers.json"
LOCAL_ID = "local"          # the kit's own server; not stored, cannot be deleted
ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")


class ProviderError(ValueError):
    """Something the user can fix, phrased for the user."""


# ------------------------------------------------------------------ store ----

def _path(root: Path) -> Path:
    return Path(root) / FILE


def load(root: Path) -> list[dict]:
    p = _path(root)
    if not p.is_file():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return []
    return [r for r in (data.get("providers") or []) if isinstance(r, dict)]


def save(root: Path, providers: list[dict]) -> None:
    p = _path(root)
    tmp = p.with_suffix(f".{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_text(json.dumps({"providers": providers}, indent=1), encoding="utf-8")
    tmp.replace(p)


def mask(key: str) -> str:
    key = key or ""
    if not key:
        return ""
    return f"{key[:3]}...{key[-4:]}" if len(key) > 10 else "..." * 2


def public(provider: dict) -> dict:
    """What the browser is allowed to see."""
    return {
        "id": provider["id"],
        "name": provider.get("name") or provider["id"],
        "base_url": provider.get("base_url", ""),
        "models": list(provider.get("models") or []),
        "default_model": provider.get("default_model", ""),
        "context_length": provider.get("context_length"),
        "efforts": list(provider.get("efforts") or []),
        "vision": provider.get("vision"),
        "vision_detected": provider.get("vision_detected"),
        "has_key": bool(provider.get("api_key")),
        "key_hint": mask(provider.get("api_key", "")),
        "checked": provider.get("checked"),
        "note": provider.get("note", ""),
    }


# ------------------------------------------------------------- validation ----

def _clean(row: dict, existing: dict | None = None) -> dict:
    name = str(row.get("name") or "").strip()
    base = str(row.get("base_url") or "").strip().rstrip("/")
    if not name:
        raise ProviderError("give the provider a name")
    if not base:
        raise ProviderError("a base URL is required, e.g. https://api.example.com/v1")
    parsed = urllib.parse.urlparse(base)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ProviderError(f"{base} is not an http(s) URL")
    if not base.rstrip("/").endswith("/v1") and "/v" not in parsed.path:
        # not fatal - some servers mount the API at the root - but it is the
        # single most common mistake, so say it plainly
        note = "this URL has no /v1 - most servers need it"
    else:
        note = ""
    models = [str(m).strip() for m in (row.get("models") or []) if str(m).strip()]
    default_model = str(row.get("default_model") or "").strip()
    if not default_model and models:
        default_model = models[0]
    if not default_model:
        raise ProviderError("a model id is required - use Test to list what the "
                            "endpoint offers, or type one in")
    if default_model not in models:
        models.insert(0, default_model)

    pid = str(row.get("id") or "").strip().lower()
    if not pid:
        pid = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:32] or "provider"
    if pid == LOCAL_ID:
        raise ProviderError(f"'{LOCAL_ID}' is the name of this computer's own server")
    if not ID_RE.match(pid):
        raise ProviderError("the id may use letters, digits, dash and underscore")

    key = str(row.get("api_key") or "")
    if not key and existing:
        key = existing.get("api_key", "")          # empty means "keep the old key"

    # How big a context this endpoint's model has. Only the user knows: /models
    # does not report it, and a remote model's window has nothing to do with
    # this machine's VRAM. Without it the UI cannot draw the context meter for a
    # provider at all - it hid the meter rather than guess, which left people
    # with no idea how full a long conversation was getting.
    raw_ctx = row.get("context_length", (existing or {}).get("context_length"))
    context_length = None
    if raw_ctx not in (None, "", 0):
        try:
            context_length = int(float(raw_ctx))
        except (TypeError, ValueError):
            raise ProviderError("context must be a number of tokens, e.g. 128000")
        if context_length < 512:
            raise ProviderError("a context under 512 tokens is not usable")
        if context_length > 100_000_000:
            raise ProviderError("that context looks like a typo")

    efforts = row.get("efforts")
    if efforts is None:
        efforts = (existing or {}).get("efforts")
    efforts = [e for e in EFFORT_ORDER if e in {str(x).lower() for x in (efforts or [])}]

    # Images. Two values, because they answer different questions: what the
    # endpoint said when probed, and what the user told us. Plenty of
    # OpenAI-compatible servers publish no modality information at all, so
    # without somewhere to say "yes it does" a working vision model would be
    # unusable here forever.
    detected = row.get("vision_detected")
    if detected is None:
        detected = (existing or {}).get("vision_detected")
    vision = row.get("vision", (existing or {}).get("vision"))
    if isinstance(vision, str):
        v = vision.strip().lower()
        vision = True if v in ("1", "true", "yes", "on") else (
            False if v in ("0", "false", "no", "off") else None)
    elif vision is not None:
        vision = bool(vision)

    return {"id": pid, "name": name, "base_url": base, "api_key": key,
            "models": models, "default_model": default_model,
            "context_length": context_length, "efforts": efforts,
            "vision": vision, "vision_detected": detected,
            "checked": (existing or {}).get("checked"), "note": note}


def upsert(root: Path, row: dict) -> dict:
    providers = load(root)
    pid = str(row.get("id") or "").strip().lower()
    existing = next((p for p in providers if p["id"] == pid), None)
    clean = _clean(row, existing)
    if existing:
        providers = [clean if p["id"] == existing["id"] else p for p in providers]
    else:
        if any(p["id"] == clean["id"] for p in providers):
            raise ProviderError(f"a provider called '{clean['id']}' already exists")
        providers.append(clean)
    save(root, providers)
    return clean


def delete(root: Path, pid: str) -> None:
    providers = [p for p in load(root) if p["id"] != pid]
    save(root, providers)


def find(root: Path, pid: str) -> dict | None:
    return next((p for p in load(root) if p["id"] == pid), None)


# ---------------------------------------------------------------- probing ----

def probe(base_url: str, api_key: str = "", timeout=15) -> list[str]:
    """Ask an endpoint what it serves. This is a user-configured address, not a
    model-chosen one, so local and private addresses are allowed here - that is
    the whole point of pointing at another runtime on this machine."""
    url = base_url.rstrip("/") + "/models"
    req = urllib.request.Request(url, headers={
        "Accept": "application/json",
        **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:200]
        if e.code in (401, 403):
            raise ProviderError(f"the endpoint refused the key (HTTP {e.code})") from e
        raise ProviderError(f"HTTP {e.code} from {url}: {detail}") from e
    except urllib.error.URLError as e:
        raise ProviderError(f"could not reach {url}: {e.reason}") from e
    except (ValueError, OSError) as e:
        raise ProviderError(f"{url} did not answer with JSON: {e}") from e
    rows = data.get("data") if isinstance(data, dict) else data
    models = [str(r.get("id")) for r in (rows or []) if isinstance(r, dict) and r.get("id")]
    if not models:
        raise ProviderError(f"{url} answered, but listed no models")
    return sorted(models)


# Reasoning effort levels differ per model - low/medium/high on some, with
# xhigh or max on others - and inventing a menu of levels an endpoint does not
# accept just produces errors or silence. /models sometimes says: OpenRouter
# lists supported_parameters, and some servers publish the enum outright. When
# nothing says, offering nothing is more honest than guessing.
EFFORT_ORDER = ["minimal", "low", "medium", "high", "xhigh", "max"]


def efforts_from_model_row(row: dict) -> list[str]:
    """Reasoning levels a /models entry claims to support, in ascending order.

    Endpoints describe this in at least three ways, so match on shape rather
    than on one blessed key: an explicit list of levels, a nested schema with an
    enum, or - OpenRouter's way - a supported_parameters list that only says
    "reasoning" is accepted without naming the levels. That last case gets the
    three every such API takes; anything more specific comes from the endpoint.
    """
    if not isinstance(row, dict):
        return []
    found: set[str] = set()
    supports_reasoning = False

    def collect(value, depth=0):
        nonlocal supports_reasoning
        if depth > 4:
            return
        if isinstance(value, str):
            v = value.strip().lower()
            if v in EFFORT_ORDER:
                found.add(v)
            elif v in ("reasoning", "reasoning_effort", "include_reasoning",
                       "thinking"):
                supports_reasoning = True
        elif isinstance(value, list):
            for item in value:
                collect(item, depth + 1)
        elif isinstance(value, dict):
            for key, item in value.items():
                if str(key).strip().lower() in ("reasoning", "reasoning_effort",
                                                "thinking"):
                    supports_reasoning = True
                collect(item, depth + 1)

    for key in ("supported_reasoning_efforts", "reasoning_efforts",
                "reasoning_effort", "supported_parameters", "capabilities",
                "reasoning", "thinking"):
        if key in row:
            collect(row[key])

    levels = [e for e in EFFORT_ORDER if e in found]
    if levels:
        return levels
    # it says it reasons but not at what settings: the common three
    return ["low", "medium", "high"] if supports_reasoning else []


def vision_from_model_row(row: dict) -> bool | None:
    """Whether a /models entry says this model takes images.

    True / False when it says, None when it does not - and None is not False.
    An endpoint that stays quiet about modalities is not the same as one that
    says text-only, so the difference is kept and the user can settle it.
    """
    if not isinstance(row, dict):
        return None
    said = False

    def modalities(value, depth=0):
        nonlocal said
        out: set[str] = set()
        if depth > 4:
            return out
        if isinstance(value, str):
            return {value.strip().lower()}
        if isinstance(value, list):
            for item in value:
                out |= modalities(item, depth + 1)
            return out
        if isinstance(value, dict):
            for key, item in value.items():
                k = str(key).strip().lower()
                if k in ("vision", "image", "images", "supports_vision",
                         "multimodal"):
                    said = True
                    if isinstance(item, bool):
                        out.add("image" if item else "text")
                        continue
                if k in ("input_modalities", "modalities", "input", "architecture",
                         "capabilities", "supported_parameters"):
                    out |= modalities(item, depth + 1)
        return out

    found = modalities(row)
    if "image" in found:
        return True
    if said or found:
        return False if ("text" in found or said) else None
    return None


def probe_capabilities(base_url: str, model: str, api_key: str = "",
                       timeout=15) -> dict:
    """What a model's own /models entry claims: reasoning levels and images.

    One request for both - they come from the same row, and asking twice for
    the same page is just slower.
    """
    url = base_url.rstrip("/") + "/models"
    req = urllib.request.Request(url, headers={
        "Accept": "application/json",
        **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
    except Exception:                              # noqa: BLE001
        return {"efforts": [], "vision": None}
    rows = data.get("data") if isinstance(data, dict) else data
    for r in rows or []:
        if isinstance(r, dict) and str(r.get("id")) == model:
            return {"efforts": efforts_from_model_row(r),
                    "vision": vision_from_model_row(r)}
    return {"efforts": [], "vision": None}


def probe_efforts(base_url: str, model: str, api_key: str = "",
                  timeout=15) -> list[str]:
    """What levels this model says it takes. [] means "it did not say"."""
    return probe_capabilities(base_url, model, api_key, timeout)["efforts"]


def test(root: Path, row: dict) -> dict:
    """Probe an endpoint the user is editing (the stored key is reused when the
    form leaves the key box empty)."""
    key = str(row.get("api_key") or "")
    if not key and row.get("id"):
        key = (find(root, str(row["id"])) or {}).get("api_key", "")
    base = str(row.get("base_url") or "").strip().rstrip("/")
    models = probe(base, key)
    # while we have its answer, ask what reasoning levels this model takes -
    # the levels differ per model, and offering ones it does not accept just
    # produces errors or silent no-ops
    wanted = str(row.get("default_model") or "").strip() or (models[0] if models else "")
    caps = probe_capabilities(base, wanted, key) if wanted else {"efforts": [], "vision": None}
    pid = str(row.get("id") or "").strip().lower()
    if pid:
        stored = find(root, pid)
        if stored:
            stored["models"] = models
            stored["efforts"] = caps["efforts"]
            stored["vision_detected"] = caps["vision"]
            stored["checked"] = time.time()
            save(root, [stored if p["id"] == pid else p for p in load(root)])
    return {"models": models, "efforts": caps["efforts"], "vision": caps["vision"]}


# ------------------------------------------------------------- resolution ----

def resolve(root: Path, pid: str, model: str, local: dict) -> dict:
    """Where a turn should actually be sent. `local` describes the kit's own
    server, which is the answer for everything except a named provider."""
    if not pid or pid == LOCAL_ID:
        return {"id": LOCAL_ID, "name": local.get("name", "This computer"),
                "base_url": local["base_url"], "api_key": local.get("api_key", "local"),
                "model": local["model"], "remote": False}
    provider = find(root, pid)
    if not provider:
        raise ProviderError(f"provider '{pid}' is gone - pick another in Settings")
    chosen = model or provider.get("default_model")
    if chosen not in (provider.get("models") or [chosen]):
        chosen = provider.get("default_model")
    return {"id": provider["id"], "name": provider["name"],
            "base_url": provider["base_url"],
            "api_key": provider.get("api_key", ""),
            "model": chosen, "remote": True}
