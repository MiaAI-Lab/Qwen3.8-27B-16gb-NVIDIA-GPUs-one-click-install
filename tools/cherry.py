#!/usr/bin/env python3
"""Cherry Studio integration for the Windows kit (used by tools/win_start.py).

Cherry Studio (https://github.com/CherryHQ/cherry-studio) is an open-source
desktop chat app with streaming, tool calling and MCP support.  This module
turns it into the kit's "one-click chat":

  1. downloads the pinned Windows x64 *portable* release once into
     apps/cherry-studio/  (gitignored; ~285 MB)
  2. first time only: lets Cherry create its data folder next to the exe
     (apps/cherry-studio/data), then closes it again
  3. writes the kit's provider + model + defaults straight into Cherry's
     SQLite store (data/Data/cherrystudio.sqlite):
        provider  "Simplex (local)"  ->  http://127.0.0.1:<PORT>/v1, key "local"
        model     qwen3.8-27b-exl3-2.0bpw  (function calling + images, ~200k context)
        default chat / translate / quick-assistant model -> that model
        Cherry Assistant -> that model, web search ON, MCP tools "auto",
        builtin MCP servers @cherry/fetch + @cherry/sequentialthinking
        installed (CHERRY_WEB_SEARCH / CHERRY_MCP_SERVERS in .env)
        onboarding skipped
  4. opens Cherry once the server reports Ready

Everything is idempotent: later launches only re-check (and re-point the
base URL if PORT changed in .env).  The data layout is validated against the
pinned CHERRY_VERSION; bump it only after checking the schema still matches.

Stand-alone use:
    .venv\\Scripts\\python.exe tools\\cherry.py prepare   # download + configure
    .venv\\Scripts\\python.exe tools\\cherry.py open      # open Cherry
    .venv\\Scripts\\python.exe tools\\cherry.py status
"""
from __future__ import annotations

import base64
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APPS_DIR = ROOT / "apps" / "cherry-studio"
SERVE = ROOT / "tools" / "serve_openai.py"

# Pinned upstream release.  The provider/model rows written below follow the
# v2.0.x data layout (user_provider / user_model / preference / assistant).
CHERRY_VERSION = "2.0.10"
CHERRY_REPO = "CherryHQ/cherry-studio"

PROVIDER_ID = "mia-qwen-local"
PROVIDER_NAME = "Simplex (local)"
FALLBACK_MODEL_ID = "qwen3.8-27b-exl3-2.0bpw"
MODEL_NAME = "Qwen3.8-27B EXL3 2.0bpw"
MODEL_GROUP = "Qwen"
API_KEY = "local"
CHERRYAI_DEFAULT_UNIQUE_MODEL_ID = "cherryai::qwen"   # upstream seed default
OPENAI_CHAT = "openai-chat-completions"

DEFAULT_MODEL_PREF_KEYS = (
    "chat.default_model_id",
    "feature.quick_assistant.model_id",
    "feature.translate.model_id",
)

# Tools on by default.  Web search = Cherry's built-in web tool group
# (web_search + web_fetch, given to the model as function tools); the stock
# keyword provider "exa-mcp" (https://mcp.exa.ai/mcp) needs no API key, URL
# fetch falls back to the built-in fetcher.  MCP: the assistant runs in
# "auto" mode (every active MCP server is offered) and the kit installs
# Cherry's own keyless builtin servers below.
DEFAULT_WEB_SEARCH = True
DEFAULT_MCP_SERVERS = "@cherry/fetch,@cherry/sequentialthinking"
# Builtin (inMemory) servers that need no configuration - mirrors
# src/shared/data/presets/mcpServers.ts in the pinned release.
BUILTIN_MCP_PRESETS = {
    "@cherry/fetch": {},
    "@cherry/sequentialthinking": {},
    "@cherry/python": {},
    "@cherry/browser": {},
}

_log = print


def set_logger(fn) -> None:
    global _log
    _log = fn


def log(msg: str) -> None:
    _log(msg)


# ---------------------------------------------------------------- paths ----

def kit_model_id(cfg: dict[str, str] | None = None) -> str:
    """Model id the server reports: .env MODEL_ID, else the model folder name
    lower-cased (exactly what serve_openai.py --model_id defaults to)."""
    cfg = cfg or {}
    mid = (cfg.get("MODEL_ID") or "").strip().lower()
    if mid:
        return mid
    md = (cfg.get("MODEL_DIR") or "").strip().strip('"')
    if md:
        return Path(md).name.lower()
    return FALLBACK_MODEL_ID


def kit_model_name(model_id: str) -> str:
    """Display name in Cherry, e.g. qwen3.8-27b-exl3-3.0bpw -> Qwen3.8-27B EXL3 3.0bpw."""
    parts = model_id.split("-")
    out = []
    for p in parts:
        if p.startswith("qwen"):
            out.append("Qwen" + p[4:])
        elif p.endswith("b") and p[:-1].replace(".", "").isdigit():
            out.append(p.upper())
        elif p == "exl3":
            out.append("EXL3")
        else:
            out.append(p)
    return " ".join(out)


def cherry_version(cfg: dict[str, str]) -> str:
    return (cfg.get("CHERRY_VERSION") or os.environ.get("CHERRY_VERSION") or CHERRY_VERSION).strip().lstrip("v")


def asset_name(version: str) -> str:
    arch = "arm64" if os.environ.get("PROCESSOR_ARCHITECTURE", "").upper() == "ARM64" else "x64"
    return f"Cherry-Studio-{version}-win-{arch}-portable.exe"


def asset_url(version: str) -> str:
    return f"https://github.com/{CHERRY_REPO}/releases/download/v{version}/{asset_name(version)}"


def portable_exe(cfg: dict[str, str]) -> Path:
    return APPS_DIR / asset_name(cherry_version(cfg))


def external_exe(cfg: dict[str, str]) -> Path | None:
    raw = (cfg.get("CHERRY_EXE") or os.environ.get("CHERRY_EXE") or "").strip().strip('"')
    if not raw:
        return None
    p = Path(raw)
    return p if p.is_file() else None


def data_dir() -> Path:
    # electron-builder portable: PORTABLE_EXECUTABLE_DIR/data is Cherry's userData
    return APPS_DIR / "data"


def db_path() -> Path:
    return data_dir() / "Data" / "cherrystudio.sqlite"


def marker_path() -> Path:
    return APPS_DIR / "kit-config.json"


def read_marker() -> dict:
    try:
        return json.loads(marker_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def write_marker(d: dict) -> None:
    APPS_DIR.mkdir(parents=True, exist_ok=True)
    marker_path().write_text(json.dumps(d, indent=2), encoding="utf-8")


def base_url(port: str | int) -> str:
    return f"http://127.0.0.1:{port}/v1"


def _truthy(v: str | None, default: bool) -> bool:
    if v is None or v.strip() == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def tools_config(cfg: dict[str, str]) -> dict:
    """What the kit turns on in Cherry (from .env CHERRY_WEB_SEARCH / CHERRY_MCP_SERVERS)."""
    web = _truthy(cfg.get("CHERRY_WEB_SEARCH") or os.environ.get("CHERRY_WEB_SEARCH"), DEFAULT_WEB_SEARCH)
    raw = cfg.get("CHERRY_MCP_SERVERS")
    if raw is None:
        raw = os.environ.get("CHERRY_MCP_SERVERS", DEFAULT_MCP_SERVERS)
    names = [n.strip() for n in raw.replace(";", ",").split(",") if n.strip()]
    unknown = [n for n in names if n not in BUILTIN_MCP_PRESETS]
    return {"web_search": web, "mcp": [n for n in names if n in BUILTIN_MCP_PRESETS], "unknown": unknown}


# ------------------------------------------------------------- download ----

def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "mia-qwen-kit/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(part, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        last = -1
        t0 = time.time()
        while True:
            chunk = resp.read(1024 * 256)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total:
                pct = done * 100 // total
                if pct != last and pct % 5 == 0:
                    last = pct
                    rate = done / max(time.time() - t0, 0.1) / 1e6
                    log(f"      {pct:3d}%  {done / 1e6:6.0f} / {total / 1e6:.0f} MB  ({rate:.1f} MB/s)")
    if total and done != total:
        part.unlink(missing_ok=True)
        raise OSError(f"short download: {done} of {total} bytes")
    part.replace(dest)


def ensure_downloaded(cfg: dict[str, str]) -> Path:
    exe = portable_exe(cfg)
    if exe.is_file() and exe.stat().st_size > 50_000_000:
        return exe
    ver = cherry_version(cfg)
    url = asset_url(ver)
    log(f"  [dl]  Cherry Studio v{ver} (portable, ~285 MB, once)")
    log(f"      {url}")
    try:
        download(url, exe)
    except Exception as e:  # noqa: BLE001
        raise OSError(f"could not download Cherry Studio: {e}\n"
                      f"      Download it yourself and put it at:\n      {exe}") from e
    log(f"      saved -> {exe}")
    return exe


# ------------------------------------------------------------- process ----

def _creationflags() -> int:
    if sys.platform != "win32":
        return 0
    return subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP


def launch(exe: Path, *args: str) -> subprocess.Popen:
    return subprocess.Popen(
        [str(exe), *args],
        cwd=str(exe.parent),
        creationflags=_creationflags(),
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        close_fds=True,
    )


def _ps(cmd: str, timeout: float = 20) -> str:
    r = subprocess.run(["powershell", "-NoProfile", "-Command", cmd],
                       capture_output=True, text=True, timeout=timeout)
    return (r.stdout or "").strip()


def child_pids(pid: int) -> list[int]:
    try:
        out = _ps(f"Get-CimInstance Win32_Process -Filter 'ParentProcessId={pid}' | "
                  f"Select-Object -ExpandProperty ProcessId")
        return [int(x) for x in out.split() if x.strip().isdigit()]
    except Exception:  # noqa: BLE001
        return []


def kill_tree(pid: int) -> None:
    try:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                       capture_output=True, text=True, timeout=30)
    except Exception:  # noqa: BLE001  (not Windows / already gone)
        try:
            os.kill(pid, 9)
        except Exception:  # noqa: BLE001
            pass


def is_running() -> bool:
    """True if any Cherry Studio process is alive (kit copy or user install)."""
    if sys.platform != "win32":
        return False
    try:
        r = subprocess.run(["tasklist", "/FI", "IMAGENAME eq Cherry Studio.exe", "/NH"],
                           capture_output=True, text=True, timeout=20)
        return "Cherry Studio.exe" in (r.stdout or "")
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------- sqlite ----

def _connect(ro: bool = False) -> sqlite3.Connection:
    uri = db_path().resolve().as_uri()      # file:///C:/... (sqlite accepts this)
    if ro:
        uri += "?mode=ro"
    con = sqlite3.connect(uri, uri=True, timeout=10)
    con.execute("PRAGMA busy_timeout=10000")
    return con


def db_bootstrapped() -> bool:
    if not db_path().is_file():
        return False
    try:
        con = _connect(ro=True)
        try:
            row = con.execute(
                "SELECT 1 FROM app_state WHERE key='seedRunner:bootstrapCompleted'").fetchone()
            if not row:
                return False
            a = con.execute("SELECT 1 FROM assistant WHERE deleted_at IS NULL LIMIT 1").fetchone()
            return a is not None
        finally:
            con.close()
    except sqlite3.Error:
        return False


# fractional-indexing (base62) port - just enough to slot a row first/last.
_DIGITS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def _int_len(head: str) -> int:
    if "a" <= head <= "z":
        return ord(head) - ord("a") + 2
    if "A" <= head <= "Z":
        return ord("Z") - ord(head) + 2
    raise ValueError(head)


def _int_part(key: str) -> str:
    n = _int_len(key[0])
    if n > len(key):
        raise ValueError(key)
    return key[:n]


def _midpoint(a: str, b: str | None) -> str:
    zero = _DIGITS[0]
    if b is not None:
        n = 0
        while (a[n] if n < len(a) else zero) == (b[n] if n < len(b) else None):
            n += 1
        if n > 0:
            return b[:n] + _midpoint(a[n:], b[n:])
    da = _DIGITS.index(a[0]) if a else 0
    dbi = _DIGITS.index(b[0]) if b else len(_DIGITS)
    if dbi - da > 1:
        return _DIGITS[round(0.5 * (da + dbi))]
    if b and len(b) > 1:
        return b[:1]
    return _DIGITS[da] + _midpoint(a[1:], None)


def _inc(x: str) -> str | None:
    head, digs = x[0], list(x[1:])
    carry = True
    for i in range(len(digs) - 1, -1, -1):
        if not carry:
            break
        d = _DIGITS.index(digs[i]) + 1
        if d == len(_DIGITS):
            digs[i] = _DIGITS[0]
        else:
            digs[i] = _DIGITS[d]
            carry = False
    if carry:
        if head == "Z":
            return "a" + _DIGITS[0]
        if head == "z":
            return None
        h = chr(ord(head) + 1)
        if h > "a":
            digs.append(_DIGITS[0])
        else:
            digs.pop()
        return h + "".join(digs)
    return head + "".join(digs)


def _dec(x: str) -> str | None:
    head, digs = x[0], list(x[1:])
    borrow = True
    for i in range(len(digs) - 1, -1, -1):
        if not borrow:
            break
        d = _DIGITS.index(digs[i]) - 1
        if d == -1:
            digs[i] = _DIGITS[-1]
        else:
            digs[i] = _DIGITS[d]
            borrow = False
    if borrow:
        if head == "a":
            return "Z" + _DIGITS[-1]
        if head == "A":
            return None
        h = chr(ord(head) - 1)
        if h < "Z":
            digs.append(_DIGITS[-1])
        else:
            digs.pop()
        return h + "".join(digs)
    return head + "".join(digs)


def key_between(a: str | None, b: str | None) -> str:
    if a is None and b is None:
        return "a0"
    if a is None:
        ib = _int_part(b)
        fb = b[len(ib):]
        if ib == "A" + "0" * 26:
            return ib + _midpoint("", fb)
        if ib < b:
            return ib
        r = _dec(ib)
        if r is None:
            raise ValueError("cannot decrement")
        return r
    if b is None:
        ia = _int_part(a)
        fa = a[len(ia):]
        i = _inc(ia)
        return ia + _midpoint(fa, None) if i is None else i
    ia, ib = _int_part(a), _int_part(b)
    fa, fb = a[len(ia):], b[len(ib):]
    if ia == ib:
        return ia + _midpoint(fa, fb)
    i = _inc(ia)
    if i is None:
        raise ValueError("cannot increment")
    return i if i < b else ia + _midpoint(fa, None)


def _first_key(con: sqlite3.Connection, table: str, where: str = "", params=()) -> str:
    row = con.execute(f"SELECT MIN(order_key) FROM {table} {where}", params).fetchone()
    try:
        return key_between(None, row[0]) if row and row[0] else "a0"
    except ValueError:
        return "Zz"


def vision_enabled(cfg: dict[str, str]) -> bool:
    v = (cfg.get("VISION") or os.environ.get("VISION") or "auto").strip().lower()
    return v not in ("0", "false", "no", "off", "none")


def model_capabilities(vision: bool) -> tuple[list[str], list[str]]:
    caps = ["function-call"] + (["image-recognition"] if vision else [])
    mods = ["text"] + (["image"] if vision else [])
    return caps, mods


def configure_db(port: str | int, model_id: str, context: int | None,
                 tools: dict | None = None, vision: bool = True) -> list[str]:
    """Upsert provider/model/defaults (+ web search / MCP tools). Returns change notes."""
    now = int(time.time() * 1000)
    unique = f"{PROVIDER_ID}::{model_id}"
    url = base_url(port)
    notes: list[str] = []
    marker = read_marker()
    first = not marker.get("configured")
    # a profile change (new quant) adds a model row and moves the defaults to it
    old_model_id = marker.get("model_id")
    retarget = bool(old_model_id) and old_model_id != model_id
    old_unique = f"{PROVIDER_ID}::{old_model_id}" if retarget else None
    tools = tools or {"web_search": DEFAULT_WEB_SEARCH, "mcp": []}
    want_tools = {"web_search": bool(tools.get("web_search")), "mcp": list(tools.get("mcp", []))}
    apply_tools = first or marker.get("tools") != want_tools

    con = _connect()
    try:
        con.execute("BEGIN IMMEDIATE")
        row = con.execute(
            "SELECT endpoint_configs FROM user_provider WHERE provider_id=?", (PROVIDER_ID,)).fetchone()
        ep = json.dumps({OPENAI_CHAT: {"baseUrl": url}})
        if row is None:
            keys = json.dumps([{"id": str(uuid.uuid4()), "key": API_KEY, "isEnabled": True}])
            con.execute(
                "INSERT INTO user_provider (provider_id, preset_provider_id, name, logo_key, "
                "endpoint_configs, default_chat_endpoint, api_keys, auth_config, provider_settings, "
                "is_enabled, order_key, created_at, updated_at) "
                "VALUES (?, NULL, ?, NULL, ?, ?, ?, NULL, NULL, 1, ?, ?, ?)",
                (PROVIDER_ID, PROVIDER_NAME, ep, OPENAI_CHAT, keys,
                 _first_key(con, "user_provider"), now, now))
            notes.append(f"provider '{PROVIDER_NAME}' -> {url}")
        else:
            try:
                cur = json.loads(row[0] or "{}").get(OPENAI_CHAT, {}).get("baseUrl")
            except ValueError:
                cur = None
            if cur != url:
                con.execute(
                    "UPDATE user_provider SET endpoint_configs=?, is_enabled=1, updated_at=? "
                    "WHERE provider_id=?", (ep, now, PROVIDER_ID))
                notes.append(f"provider base URL -> {url}")

        caps, mods = model_capabilities(vision)
        row = con.execute("SELECT context_window, capabilities FROM user_model WHERE id=?",
                          (unique,)).fetchone()
        if row is None:
            con.execute(
                "INSERT INTO user_model (id, provider_id, model_id, preset_model_id, name, description, "
                "\"group\", capabilities, input_modalities, output_modalities, endpoint_types, "
                "context_window, max_input_tokens, max_output_tokens, supports_streaming, reasoning, "
                "parameters, pricing, is_enabled, is_hidden, is_deprecated, order_key, notes, "
                "created_at, updated_at, input_modalities_explicit) "
                "VALUES (?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, NULL, ?, NULL, NULL, 1, NULL, NULL, NULL, "
                "1, 0, 0, ?, NULL, ?, ?, 1)",
                (unique, PROVIDER_ID, model_id, kit_model_name(model_id),
                 "Local EXL3 2.0bpw quant served by the Simplex kit (text only, thinking on).",
                 MODEL_GROUP, json.dumps(caps), json.dumps(mods), json.dumps(["text"]),
                 context, _first_key(con, "user_model", "WHERE provider_id=?", (PROVIDER_ID,)),
                 now, now))
            notes.append(f"model '{model_id}'" + (" (text + images)" if vision else " (text)"))
        else:
            if context and row[0] != context:
                con.execute("UPDATE user_model SET context_window=?, updated_at=? WHERE id=?",
                            (context, now, unique))
                notes.append(f"model context window -> {context}")
            try:
                have = json.loads(row[1] or "[]")
            except ValueError:
                have = []
            if sorted(have) != sorted(caps):
                con.execute("UPDATE user_model SET capabilities=?, input_modalities=?, "
                            "input_modalities_explicit=1, updated_at=? WHERE id=?",
                            (json.dumps(caps), json.dumps(mods), now, unique))
                notes.append("model capabilities -> " + ", ".join(caps))

        if first or retarget:
            for key in DEFAULT_MODEL_PREF_KEYS:
                con.execute(
                    "INSERT INTO preference (scope, key, value, created_at, updated_at) "
                    "VALUES ('default', ?, ?, ?, ?) "
                    "ON CONFLICT(scope, key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                    (key, json.dumps(unique), now, now))
            n = con.execute(
                "UPDATE assistant SET model_id=?, updated_at=? WHERE deleted_at IS NULL "
                "AND (model_id IS NULL OR model_id=? OR model_id=?)",
                (unique, now, CHERRYAI_DEFAULT_UNIQUE_MODEL_ID, old_unique or "")).rowcount
            notes.append(f"default chat model + {n} assistant(s) -> {model_id}"
                         + ("" if retarget else "; onboarding skipped"))
        if first:
            # skip the provider-setup onboarding: the provider is already there
            con.execute(
                "INSERT INTO preference (scope, key, value, created_at, updated_at) "
                "VALUES ('default', 'app.onboarding.provider_setup.status', ?, ?, ?) "
                "ON CONFLICT(scope, key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (json.dumps("completed"), now, now))
            # local-first kit: no usage analytics unless you turn it on in Cherry
            con.execute(
                "INSERT INTO preference (scope, key, value, created_at, updated_at) "
                "VALUES ('default', 'app.privacy.data_collection.enabled', ?, ?, ?) "
                "ON CONFLICT(scope, key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (json.dumps(False), now, now))

        if apply_tools:
            # --- builtin MCP servers (rows exactly like Cherry's "Install" button writes)
            server_ids: list[str] = []
            for name in want_tools["mcp"]:
                row = con.execute("SELECT id FROM mcp_server WHERE name=?", (name,)).fetchone()
                if row:
                    con.execute("UPDATE mcp_server SET is_active=1, updated_at=? WHERE id=?", (now, row[0]))
                    server_ids.append(row[0])
                    continue
                sid = str(uuid.uuid4())
                con.execute(
                    "INSERT INTO mcp_server (id, name, type, provider, sort_order, is_active, "
                    "install_source, is_trusted, trusted_at, installed_at, created_at, updated_at) "
                    "VALUES (?, ?, 'inMemory', 'CherryAI', 0, 1, 'builtin', 1, ?, ?, ?, ?)",
                    (sid, name, now, now, now, now))
                server_ids.append(sid)
                notes.append(f"MCP server {name} installed + active")
            # --- default assistant: web tools on, MCP auto, link the servers
            for aid, raw in con.execute(
                    "SELECT id, settings FROM assistant WHERE deleted_at IS NULL AND model_id=?",
                    (unique,)).fetchall():
                try:
                    st = json.loads(raw) if raw else {}
                except ValueError:
                    st = {}
                st["enableWebSearch"] = want_tools["web_search"]
                st["mcpMode"] = "auto" if want_tools["mcp"] else st.get("mcpMode", "manual")
                con.execute("UPDATE assistant SET settings=?, updated_at=? WHERE id=?",
                            (json.dumps(st), now, aid))
                for sid in server_ids:
                    con.execute(
                        "INSERT OR IGNORE INTO assistant_mcp_server (assistant_id, mcp_server_id, "
                        "created_at, updated_at) VALUES (?, ?, ?, ?)", (aid, sid, now, now))
            notes.append(
                "assistant: web search " + ("ON" if want_tools["web_search"] else "off")
                + (f", MCP auto ({len(server_ids)} server(s))" if want_tools["mcp"] else ", MCP manual"))
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()

    marker.update({"configured": True, "port": str(port), "model_id": model_id,
                   "context": context, "cherry_version": marker.get("cherry_version"),
                   "base_url": url, "tools": want_tools, "vision": bool(vision), "updated_at": now})
    write_marker(marker)
    return notes


# ------------------------------------------------------------ first run ----

def first_run_init(exe: Path, timeout: float = 240) -> None:
    """Let Cherry build its data folder once, then close it (it has nothing
    of ours yet - the configuration is written right after)."""
    log("  [ + ]  Preparing Cherry Studio for first use: it opens and closes once, do not touch it")
    proc = launch(exe)
    t0 = time.time()
    seeded_at = None
    try:
        while time.time() - t0 < timeout:
            if proc.poll() is not None:
                if db_bootstrapped():
                    break
                raise OSError(f"Cherry Studio exited early (code {proc.returncode})")
            if seeded_at is None and db_bootstrapped():
                seeded_at = time.time()
            if seeded_at is not None and time.time() - seeded_at > 8:
                break
            time.sleep(1)
        if seeded_at is None:
            raise OSError("Cherry Studio did not create its data store in time")
    finally:
        # Kill the extracted app (children of the portable stub), so the stub
        # itself can clean up its temp folder and exit.
        kids = child_pids(proc.pid)
        for k in kids:
            kill_tree(k)
        if not kids:
            kill_tree(proc.pid)
        try:
            proc.wait(timeout=30)
        except Exception:  # noqa: BLE001
            kill_tree(proc.pid)
    # let the sqlite WAL settle
    time.sleep(1.5)
    if not db_bootstrapped():
        raise OSError("Cherry Studio data store missing after first run")
    log("          OK  data folder created")


# ------------------------------------------------------------- deep link ----

def deep_link(port: str | int) -> str:
    """cherrystudio://providers/api-keys import URL (official Cherry feature).
    Used for an existing install (CHERRY_EXE): Cherry shows a confirm popup."""
    payload = {"id": PROVIDER_ID, "name": PROVIDER_NAME, "type": "openai",
               "baseUrl": base_url(port), "apiKey": API_KEY}
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    b64 = base64.b64encode(raw).decode("ascii").replace("+", "_").replace("/", "-")
    return f"cherrystudio://providers/api-keys?v=1&data={b64}"


# ----------------------------------------------------------- high level ----

def prepare(cfg: dict[str, str], port: str | int, context: int | None) -> dict:
    """Download + configure. Returns {'exe': Path, 'mode': 'portable'|'external', 'notes': [...]}.
    Raises OSError with a readable message on failure."""
    ext = external_exe(cfg)
    if ext is not None:
        return {"exe": ext, "mode": "external", "notes": [], "model_id": kit_model_id(cfg)}
    exe = ensure_downloaded(cfg)
    m = read_marker()
    if m.get("cherry_version") != cherry_version(cfg):
        m["cherry_version"] = cherry_version(cfg)
        write_marker(m)
    if not db_bootstrapped():
        first_run_init(exe)
    model_id = kit_model_id(cfg)
    notes: list[str] = []
    tools = tools_config(cfg)
    for n in tools["unknown"]:
        log(f"  ! CHERRY_MCP_SERVERS: '{n}' is not a keyless builtin server - ignored "
            f"(known: {', '.join(BUILTIN_MCP_PRESETS)})")
    m = read_marker()
    want_tools = {"web_search": tools["web_search"], "mcp": tools["mcp"]}
    vis = vision_enabled(cfg)
    need = (not m.get("configured") or m.get("port") != str(port)
            or m.get("model_id") != model_id or (context and m.get("context") != context)
            or m.get("tools") != want_tools or m.get("vision") != vis)
    if need:
        notes = configure_db(port, model_id, context, tools, vis)
        if is_running():
            notes.append("Cherry Studio was already open: restart it to pick this up")
    return {"exe": exe, "mode": "portable", "notes": notes, "model_id": model_id}


def open_app(prep: dict, port: str | int) -> None:
    exe: Path = prep["exe"]
    if prep.get("mode") == "external":
        # Official import link: Cherry pops "add provider?"; user confirms, then
        # adds the model under the provider's model list.
        launch(exe, deep_link(port))
    else:
        launch(exe)


def status(cfg: dict[str, str]) -> str:
    exe = portable_exe(cfg)
    lines = [f"exe:        {exe} ({'present' if exe.is_file() else 'missing'})",
             f"data:       {db_path()} ({'ready' if db_bootstrapped() else 'missing'})",
             f"running:    {is_running()}",
             f"marker:     {json.dumps(read_marker())}"]
    return "\n".join(lines)


def _load_env() -> dict[str, str]:
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        from win_start import load_dotenv  # type: ignore
        return load_dotenv(ROOT / ".env")
    except Exception:  # noqa: BLE001
        return {}


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "status"
    cfg = _load_env()
    port = cfg.get("PORT", "8888")
    try:
        ctx = int(cfg.get("CONTEXT_SIZE", "199936"))
    except ValueError:
        ctx = None
    if cmd == "status":
        print(status(cfg))
        return 0
    if cmd == "prepare":
        p = prepare(cfg, port, ctx)
        for n in p["notes"]:
            print("  +", n)
        print("ready:", p["exe"])
        return 0
    if cmd == "open":
        p = prepare(cfg, port, ctx)
        open_app(p, port)
        return 0
    if cmd == "link":
        print(deep_link(port))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except OSError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
