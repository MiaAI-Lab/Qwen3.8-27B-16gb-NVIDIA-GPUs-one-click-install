#!/usr/bin/env python3
"""
Minimal OpenAI- and Anthropic-compatible server for the EXL3 serving target.

Drafter: MTP by default (`-dm mtp`; the draft head lives inside the target
checkpoint, so there are no separate draft weights to download). Alternative:
no drafting at all (`-dm none`). The linux/start.sh launcher maps the .env `DRAFT`
knob onto these. This kit ships no external draft model.

Requires ExLlamaV3 >= 1.4.4: the served quant carries a quantized vision
tower (vision_bits 3), which only v1.4.4+ decodes correctly.

Images: with --vision auto (default) the vision tower is loaded next to the
text model and OpenAI `image_url` content parts (data: URLs or http(s) URLs)
are embedded through it - so chat apps can attach pictures. Images are
downscaled to --image_max_pixels first (1 MP ~ 1024 prompt tokens). If the
tower does not fit under the VRAM cap the server keeps running text-only.

Endpoints:
  GET  /                      landing page: where the APIs and the harness are
  GET  /v1/models             one model; OpenAI shape, or Anthropic shape when
                              the request carries `anthropic-version`
  GET  /v1/models/{id}        the same, for one id
  GET  /health
  POST /v1/chat/completions   OpenAI: stream and non-stream, tool calling
  POST /v1/messages           Anthropic Messages API: stream and non-stream
  POST /v1/messages/count_tokens

`stream_options: {"include_usage": true}` adds a final chunk carrying the
token counts, which is how the built-in UI reports tokens/second.

Anthropic clients (Claude Code, Cline, Continue, Zed, the Anthropic SDKs) are
served by this same process on the same port. POST /v1/messages is a
translation layer, not a second engine: the request is converted to the
OpenAI-shaped body below, generation runs once, and the result is converted
back into Anthropic content blocks (`text`, `thinking`, `tool_use`) and
Anthropic SSE events. No API key is needed; `x-api-key` is accepted and
ignored, and `anthropic-version` only selects the /v1/models shape.

Defaults match the serving convention: temperature 0.6, top-k 20, top-p 0.95,
thinking enabled (reasoning arrives inline in `<think>`), speculative
drafting active (drafter chosen via -dm, see above).
Concurrency: requests are serialized (batch-1 draft); concurrent callers queue.

Tool calling (Qwen3.8 XML format):
  - `tools` (OpenAI function specs) are rendered by the model's HF chat template
    (system "# Tools" section). `tool_choice` is accepted; required/specific
    choices are enforced with an explicit system directive.
  - assistant history with `tool_calls` is re-rendered natively by the template
    (arguments are converted JSON-string -> dict, as the template expects).
  - `role:"tool"` messages render as `<tool_response>` blocks natively.
  - Model output `<tool_call><function=name><parameter=k>v</parameter>
    </function></tool_call>` is parsed back into OpenAI `tool_calls` objects;
    generation stops at `</tool_call>`, finish_reason = "tool_calls".
  - Tool-call arguments are typed per the request's own JSON schemas
    (integer/number/boolean/array/object), strings kept on mismatch.

Launch (from repo root; 16 GB NVIDIA recipe):
  .venv/bin/python tools/serve_openai.py \
      -m models/Qwen3.8-27B-EXL3-2.0bpw -gs 14.7 -cs 199936 -cq 8,4 --port 8888
"""
import argparse, asyncio, base64, json, math, os, re, sys, time, threading, uuid
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aiohttp import web

MODEL_DIR = "models/Qwen3.8-27B-EXL3-2.0bpw"
DRAFT_DIR = "mtp"   # default drafting method: MTP head (no external draft model)
PORT = 8888
MODEL_ID = "qwen3.8-27b-exl3-2.0bpw"

gen_lock = threading.Lock()          # serialize generation (batch-1 draft)
vision = {"model": None, "max_pixels": 1048576, "reason": "not loaded"}
IMAGE_TRIPLE = "<|vision_start|><|image_pad|><|vision_end|>"   # what the chat template emits per image
stats_lock = threading.Lock()
# Cumulative counters for sparkDash live tok/s (GET /health).
stats = {
    "prompt_tokens_total": 0,
    "completion_tokens_total": 0,
    "context_length": None,
}

def _bump_stats(prompt=0, completion=0):
    if prompt <= 0 and completion <= 0:
        return
    with stats_lock:
        if prompt > 0:
            stats["prompt_tokens_total"] += int(prompt)
        if completion > 0:
            stats["completion_tokens_total"] += int(completion)

def _result_new_tokens(r):
    ids = r.get("token_ids") if isinstance(r, dict) else None
    if ids is None:
        return 0
    try:
        return int(ids.shape[-1])
    except Exception:
        return 0

TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
HOLD_BACK = 16                       # marker-safe holdback for streamed text
THINK_CLOSE = "</think>"             # where the Qwen template puts the reasoning/reply seam


class _DropTritonRemarks:
    """Triton prints one 'remark: file.py:N: 1234 instructions in function'
    line per compiled kernel and breaks the load progress bar. Drop those."""

    def __init__(self, inner):
        self._inner = inner
        self._buf = ""

    def write(self, s):
        if not isinstance(s, str):
            s = str(s)
        self._buf += s
        while True:
            rpos = self._buf.find("\r")
            npos = self._buf.find("\n")
            if rpos < 0 and npos < 0:
                break
            if rpos < 0:
                cut = npos
            elif npos < 0:
                cut = rpos
            else:
                cut = min(rpos, npos)
            line, self._buf = self._buf[:cut + 1], self._buf[cut + 1:]
            if "remark:" in line or "instructions in function" in line:
                continue
            self._inner.write(line)
        return len(s)

    def flush(self):
        if self._buf and "remark:" not in self._buf and "instructions in function" not in self._buf:
            self._inner.write(self._buf)
            self._buf = ""
        self._inner.flush()

    def isatty(self):
        return self._inner.isatty()

    def fileno(self):
        return self._inner.fileno()

    @property
    def encoding(self):
        return getattr(self._inner, "encoding", "utf-8")

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _is_triton_remark(text):
    return "remark:" in text or "instructions in function" in text


def _quiet_triton():
    """Hide Triton's per-kernel LLVM remarks (they spam stderr from C++ too,
    so wrapping sys.stderr alone is not enough)."""
    os.environ.setdefault("TRITON_PRINT_AUTOTUNING", "0")
    if getattr(_quiet_triton, "_on", False):
        return
    _quiet_triton._on = True
    try:
        rfd, wfd = os.pipe()
        saved = os.dup(2)
        os.dup2(wfd, 2)
        os.close(wfd)
    except OSError:
        sys.stderr = _DropTritonRemarks(sys.stderr)
        return

    note = {"shown": False}

    def pump():
        buf = b""
        while True:
            try:
                chunk = os.read(rfd, 8192)
            except OSError:
                break
            if not chunk:
                break
            buf += chunk
            while True:
                npos = buf.find(b"\n")
                rpos = buf.find(b"\r")
                if npos < 0 and rpos < 0:
                    break
                if npos < 0:
                    cut = rpos
                elif rpos < 0:
                    cut = npos
                else:
                    cut = min(npos, rpos)
                line, buf = buf[:cut + 1], buf[cut + 1:]
                text = line.decode("utf-8", "replace")
                if _is_triton_remark(text):
                    if not note["shown"]:
                        note["shown"] = True
                        os.write(saved, b"  compiling Triton kernels...\n")
                    continue
                os.write(saved, line)
        leftover = buf.decode("utf-8", "replace")
        if leftover and not _is_triton_remark(leftover):
            os.write(saved, buf)

    threading.Thread(target = pump, daemon = True, name = "quiet-triton").start()


def parse_split(s):
    """GPU_MEM_GB / --grid_size: one budget ("14.7") or one per GPU ("14.9,7.2")."""
    return [float(x) for x in str(s).replace(" ", "").split(",") if x]


def _cap_process_vram(budgets):
    """Hard-cap this process to budgets[i] GiB on GPU i, so a large
    unified-memory box behaves like a discrete card with that much free VRAM.
    ExLlama lifts the CUDA fraction after autosplit; pin it back to the cap."""
    import torch
    from exllamav3.util import memory as _mem
    torch.cuda.init()
    frac = {}
    for i, gb in enumerate(budgets):
        total = torch.cuda.get_device_properties(i).total_memory
        frac[i] = min(max(int(gb * 1024 ** 3) / total, 0.01), 1.0)
        print(f" == VRAM cap GPU {i} ({torch.cuda.get_device_name(i)}): {gb} GB "
              f"(fraction {frac[i]:.4f} of {total / 1024**3:.1f} GB)", flush = True)

    def _pin(devices=None):
        for i in (devices if devices is not None else frac):
            if i in frac:
                torch.cuda.set_per_process_memory_fraction(frac[i], device = i)

    _pin()
    _mem.set_memory_fraction_use = lambda use, device: _pin([device])
    _mem.set_memory_fraction_reserve = lambda reserve, device: _pin([device])
    _mem.unset_memory_fraction = lambda active: _pin(active)


# Two-GPU placement. exllamav3's layer split fills GPU 0 in order and puts
# whatever is left - the last layers, their KV cache and the 248k-vocab output
# head - on GPU 1. On an unequal pair (RTX 5080 + RTX 3050: ~960 vs ~224 GB/s)
# that is the worst choice: decode reads every weight and, at long context,
# every KV page (dequantized to fp16, several times its stored size) once per
# forward. So the slow card gets only linear-attention (GDN) blocks, which hold
# no KV; full-attention blocks, their cache and lm_head stay on the fast one.
# OFFLOAD["layers"]: None = size the offload to fit GPU 0's budget, N = offload
# exactly N GDN blocks. See README "Two GPUs" for the measurements.
# Headroom kept free on each card beyond weights + KV (GiB), measured at load and
# through a full-context prefill on 3.0 bpw: GPU 0 needs CUDA workspace plus one
# attention layer's KV dequantized to fp16 (CacheLayer_quant.get_kv allocates the
# whole cache's shape, 4 KiB/token: 1 GiB at 262k), GPU 1 the vision tower and
# its own prefill workspace (1.5 GiB in use at idle; 1.6 ran out on a 115k prompt).
OFFLOAD = {"layers": None, "reserve0_gb": 1.8, "reserve1_gb": 2.2}


def _plan_offload(model, budgets, want):
    """Module index -> device for a two-GPU text model: GDN blocks to GPU 1,
    taken as whole runs of three (one GPU hop each way per run), from the end."""
    import torch
    blocks = [(i, m) for i, m in enumerate(model.modules)
              if type(m).__name__ == "TransformerBlock"]
    has_kv = lambda m: any(sm.caps.get("kv_cache") for sm in m)
    stc = model.config.stc
    gib = lambda b: b / 1024 ** 3
    wsize = {i: gib(sum(stc.get_tensor_sizes(m.key))) for i, m in enumerate(model.modules)
             if not m.caps.get("prefer_cpu")}
    kv, deq = 0.0, 0.0
    for _, m in blocks:
        for sm in m:
            for cl in getattr(sm, "cache_layers", None) or []:
                kv += gib(cl.storage_size() + cl.overhead_size())
                if getattr(cl, "shape", None):
                    deq = max(deq, gib(2 * 2 * math.prod(cl.shape)))   # fp16 K + V
    # GPU 0 already holds the MTP head and its cache (loaded first).
    free0 = budgets[0] - gib(torch.cuda.memory_reserved(0)) - OFFLOAD["reserve0_gb"] - deq
    free1 = budgets[1] - OFFLOAD["reserve1_gb"]
    need0 = sum(wsize.values()) + kv
    gdn = [i for i, m in reversed(blocks) if not has_kv(m)]
    if want is None:
        n, excess = 0, need0 - free0
        while excess > 0 and n < len(gdn):
            excess -= wsize[gdn[n]]
            n += 1
        if excess > 0:
            raise RuntimeError(
                f"CONTEXT_SIZE does not fit: KV cache {kv:.2f} GiB leaves GPU 0 "
                f"{excess:.2f} GiB short even with every linear-attention block on "
                f"GPU 1. Lower CONTEXT_SIZE.")
    else:
        n = min(int(want), len(gdn))
    moved = sorted(gdn[:n])
    on1 = sum(wsize[i] for i in moved)
    print(f" == split plan: weights {sum(wsize.values()):.2f} GiB + KV {kv:.2f} GiB, "
          f"GPU 0 room {free0:.2f} GiB -> {n} GDN blocks ({on1:.2f} GiB) to GPU 1", flush = True)
    if on1 > free1:
        raise RuntimeError(
            f"CONTEXT_SIZE does not fit: GPU 1 would need {on1:.2f} GiB of layers "
            f"but has room for {free1:.2f} GiB. Lower CONTEXT_SIZE.")
    return {i: 1 for i in moved}


def _install_offload_loader(budgets):
    """Wrap exllamav3's layer-split loader so the text model follows
    _plan_offload(). The stock measuring loop still runs (it allocates each
    block's KV and does a reference forward per block, so an OOM surfaces at
    load time); only its device choice is replaced. A block that still does
    not fit on GPU 0 makes the loop spill everything after it to GPU 1, the
    stock behaviour, and is reported."""
    from exllamav3.model import model_ls
    orig = model_ls.Model_LSMixin._load_autosplit

    class _Planned(list):
        # active_devices[current_device_i]: 0 = "no OOM yet" -> the plan's pick
        def __getitem__(self, i):
            if isinstance(i, int) and i == 0:
                return self.plan.get(self.cur, 0)
            return list.__getitem__(self, i)

    def patched(self, progressbar, reserve_per_device, use_per_device, active_devices,
                max_chunk_size, max_output_size, max_output_factor, callback_sync,
                *rest):
        if getattr(self, "component", "text") != "text" or list(active_devices) != [0, 1]:
            yield from orig(self, progressbar, reserve_per_device, use_per_device,
                            active_devices, max_chunk_size, max_output_size,
                            max_output_factor, callback_sync, *rest)
            return
        dev = _Planned([0, 1])
        dev.plan, dev.cur = _plan_offload(self, budgets, OFFLOAD["layers"]), 0

        def track(idx, n):
            dev.cur = idx
            if callback_sync:
                callback_sync(idx, n)

        yield from orig(self, progressbar, reserve_per_device, use_per_device, dev,
                        max_chunk_size, max_output_size, max_output_factor, track, *rest)
        self.active_devices = [0, 1]
        where = [str(m.device.index) if m.device is not None and m.device.type == "cuda" else "c"
                 for m in self.modules]
        spilled = [i for i, m in enumerate(self.modules)
                   if where[i] == "1" and i not in dev.plan]
        print(f" == layer map (c=cpu): {''.join(where)}", flush = True)
        if spilled:
            print(f" !! {len(spilled)} modules did not fit GPU 0 and spilled to GPU 1 "
                  f"(lower CONTEXT_SIZE or raise GPU_OFFLOAD_LAYERS)", flush = True)

    model_ls.Model_LSMixin._load_autosplit = patched


def build_model(argv, use_draft = True, draft_kw = None):
    from argparse import ArgumentParser

    # The one-time JIT build of the CUDA extension can look like a hang;
    # say so before the import below blocks on it.
    try:
        import importlib.util, os
        if importlib.util.find_spec("exllamav3_ext") is None:
            _root = os.environ.get("TORCH_EXTENSIONS_DIR",
                                   os.path.expanduser("~/.cache/torch_extensions"))
            if not (os.path.isdir(_root) and
                    any(d == "exllamav3_ext"
                        for _, _dirs, _ in os.walk(_root) for d in _dirs)):
                print(" == compiling the CUDA extension "
                      "(one-time; a few minutes of silence is normal) ...", flush = True)
    except Exception:
        pass

    from exllamav3 import model_init, Generator
    parser = ArgumentParser()
    model_init.add_args(parser, add_draft_model_args = use_draft)
    args = parser.parse_args(argv)
    if use_draft:
        model, config, cache, tokenizer, draft_model, draft_config, draft_cache = \
            model_init.init(args, progress = True)
        generator = Generator(
            model, cache, tokenizer,
            draft_model = draft_model, draft_cache = draft_cache,
            # num_draft_tokens defaults to the draft model's arch-declared
            # default_draft_size (MTP head: 4). Must
            # match model_init's max_history sizing, which reads the same caps
            # - so an override goes to both (-ndt in argv, and here).
            **(draft_kw or {}),
        )
    else:
        model, config, cache, tokenizer = model_init.init(args, progress = True)
        generator = Generator(model, cache, tokenizer)
    return generator, tokenizer, config


def load_vision(config, max_pixels, device = 0):
    """Load the checkpoint's vision tower (Qwen3.8: 27 layers, 3-bit, ~0.3 GB)
    after the text model. Never fatal: on failure the server stays text-only.
    With two GPUs it goes on the second: it only runs while a picture is
    encoded, so it should not take room from the fast card's KV cache."""
    vision["max_pixels"] = int(max_pixels)
    if not getattr(config, "vision", None):
        vision["reason"] = "checkpoint has no vision tower"
        return None
    try:
        from PIL import Image  # noqa: F401  (pillow is needed to decode images)
    except ImportError:
        vision["reason"] = "pillow not installed (pip install pillow)"
        print(" == images: OFF - " + vision["reason"], flush = True)
        return None
    try:
        from exllamav3 import Model
        vm = Model.from_config(config, component = "vision")
        vm.load(device = f"cuda:{device}", progressbar = True)
        vision["model"] = vm
        vision["reason"] = "ok"
        return vm
    except Exception as e:  # OOM under the VRAM cap is the realistic failure
        vision["reason"] = f"vision tower failed to load: {type(e).__name__}: {str(e)[:200]}"
        print(" == images: OFF - " + vision["reason"], flush = True)
        print(" == (lower CONTEXT_SIZE in .env, e.g. 180224, to free VRAM for it)", flush = True)
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        return None


def decode_image(url):
    """OpenAI image_url -> PIL.Image (RGB), downscaled to vision['max_pixels'].
    Accepts data: URLs and http(s) URLs. Local file paths are refused on
    purpose (the server may be reachable from the LAN)."""
    import base64, io, math, urllib.request
    from PIL import Image
    url = (url or "").strip()
    if url.startswith("data:"):
        _, _, b64 = url.partition(",")
        raw = base64.b64decode(b64)
    elif url.startswith(("http://", "https://")):
        req = urllib.request.Request(url, headers = {"User-Agent": "simplex-kit/1.0"})
        with urllib.request.urlopen(req, timeout = 20) as r:
            raw = r.read(48 * 1024 * 1024 + 1)
        if len(raw) > 48 * 1024 * 1024:
            raise ValueError("image larger than 48 MB")
    else:
        raise ValueError("image_url must be a data: URL or an http(s) URL")
    img = Image.open(io.BytesIO(raw))
    img.load()
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    if w * h > vision["max_pixels"]:
        k = math.sqrt(vision["max_pixels"] / float(w * h))
        img = img.resize((max(32, int(w * k)), max(32, int(h * k))), Image.LANCZOS)
    return img


def extract_images(messages):
    """Pull image_url parts out of the OpenAI messages. Returns
    (messages with the parts rewritten as {"type": "image"}, [urls]).
    The chat template turns each {"type": "image"} into IMAGE_TRIPLE."""
    urls = []
    out = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            parts = []
            for part in c:
                if isinstance(part, dict) and part.get("type") in ("image_url", "image"):
                    iu = part.get("image_url")
                    url = iu.get("url") if isinstance(iu, dict) else (iu or part.get("image"))
                    if url:
                        urls.append(url)
                        parts.append({"type": "image"})
                        continue
                parts.append(part)
            m = dict(m, content = parts)
        out.append(m)
    return out, urls


def normalize_messages(messages):
    """OpenAI history -> template-compatible dicts (tool_calls args str->dict)."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("role") == "assistant" and m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                fn = dict(c.get("function") or {})
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except ValueError:
                        args = {}
                fn["arguments"] = args
                calls.append({"function": fn})
            m["tool_calls"] = calls
        out.append(m)
    return out


def split_reasoning(text):
    """Split Qwen reasoning from content. Generation starts inside <think>
    (the chat template ends with it), so text before </think> is reasoning.
    Returns (reasoning, content) with markers stripped."""
    close = text.find("</think>")
    if close >= 0:
        reasoning = text[:close]
        content = text[close + len("</think>"):]
        return reasoning.lstrip().removeprefix("<think>").strip(), content.strip("\n")
    if text.lstrip().startswith("<think>"):
        return text.lstrip()[len("<think>"):].strip(), ""
    return "", text


def build_tool_schemas(tools):
    """OpenAI tools list -> {function_name: {param_name: json-schema type}}."""
    schemas = {}
    for t in tools or []:
        fn = (t or {}).get("function") or {}
        name = fn.get("name")
        props = ((fn.get("parameters") or {}).get("properties")) or {}
        if name and isinstance(props, dict):
            schemas[name] = {k: v.get("type") for k, v in props.items()
                             if isinstance(v, dict)}
    return schemas


def _coerce_value(value, jtype):
    """Coerce one XML string parameter to the schema-declared JSON type.
    Lossless: on any mismatch the original string is returned unchanged."""
    v = value.strip()
    if not v:
        return value
    try:
        if jtype == "integer":
            return int(v)
        if jtype == "number":
            try:
                return int(v)
            except ValueError:
                return float(v)
        if jtype == "boolean":
            if v.lower() == "true": return True
            if v.lower() == "false": return False
        if jtype == "array":
            parsed = json.loads(v)
            if isinstance(parsed, list):
                return parsed
        if jtype == "object":
            parsed = json.loads(v)
            if isinstance(parsed, dict):
                return parsed
    except (ValueError, json.JSONDecodeError):
        pass
    return value


def coerce_tool_args(args, fn_schema):
    """Qwen's XML tool format delivers every parameter value as a string;
    OpenAI tool_calls arguments are typed JSON. Coerce each value using the
    request's own tool schema; undeclared params and failed coercions keep
    the raw string."""
    if not fn_schema:
        return args
    out = {}
    for k, v in args.items():
        t = fn_schema.get(k)
        types = t if isinstance(t, list) else [t]
        for tt in types:
            if isinstance(tt, str) and tt in ("integer", "number", "boolean",
                                              "array", "object"):
                cv = _coerce_value(v, tt)
                if not isinstance(cv, str):
                    v = cv
                    break
        out[k] = v
    return out


def parse_tool_calls(text, tool_schemas = None):
    """Parse Qwen XML tool calls. Returns (content_without_calls, [calls]).
    A <tool_call> block left unterminated is treated as complete: the
    </tool_call> stop-condition strips the closing tag from generated text."""
    calls = []
    content = text

    def parse_block(block):
        fm = re.search(r"<function=([^>]+)>", block)
        if not fm:
            return None
        name = fm.group(1).strip()
        args = {}
        for pm in re.finditer(r"<parameter=([^>]+)>\n?(.*?)\n?</parameter>",
                              block[fm.end():], flags = re.S):
            args[pm.group(1).strip()] = pm.group(2)
        if tool_schemas:
            args = coerce_tool_args(args, tool_schemas.get(name))
        return {
            "id": f"call_{uuid.uuid4().hex[:12]}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }

    while True:
        i = content.find(TOOL_CALL_OPEN)
        if i < 0:
            break
        j = content.find(TOOL_CALL_CLOSE, i)
        if j < 0:
            # truncated close (stop string consumed): parse the remainder
            call = parse_block(content[i + len(TOOL_CALL_OPEN):])
            if call:
                calls.append(call)
            content = content[:i]
            break
        call = parse_block(content[i + len(TOOL_CALL_OPEN):j])
        if call:
            calls.append(call)
        content = content[:i] + content[j + len(TOOL_CALL_CLOSE):]
    return content, calls


def tool_choice_directive(tool_choice, tools):
    """OpenAI tool_choice -> (tools_to_render, extra system directive or None).
    The Qwen template has no tool_choice support, so required/specific are
    enforced with an explicit instruction appended to the history."""
    if tool_choice in (None, "auto"):
        return tools, None
    if tool_choice == "none":
        return None, None
    names = [t["function"]["name"] for t in (tools or [])
             if isinstance(t, dict) and t.get("type") == "function"]
    if isinstance(tool_choice, dict):
        name = (tool_choice.get("function") or {}).get("name")
        return tools, (f"You must call the function `{name}` now. Reply ONLY with "
                       f"the <tool_call> block for `{name}` and nothing else.")
    if tool_choice == "required":
        one_of = " or ".join(f"`{n}`" for n in names)
        return tools, (f"You must call one of the available functions ({one_of}) "
                       "now. Reply ONLY with the <tool_call> block and nothing else.")
    return tools, None


# Effort levels this model's chat template understands, probed once. The
# template is the authority: Qwen3's raises on anything outside its own set
# (xhigh / medium / low - note "high" is NOT one of them), and a template that
# ignores reasoning_effort entirely must not be advertised as taking levels.
_EFFORT_CANDIDATES = ("minimal", "low", "medium", "high", "xhigh", "max")
_efforts_cache = None


def supported_efforts(tokenizer):
    """The subset of _EFFORT_CANDIDATES this template both accepts and acts on."""
    global _efforts_cache
    if _efforts_cache is not None:
        return _efforts_cache
    probe = [{"role": "user", "content": "hi"}]
    try:
        base = tokenizer.hf_render_chat_template(probe, add_generation_prompt = True,
                                                 enable_thinking = True)
    except Exception:                                    # noqa: BLE001
        _efforts_cache = []
        return _efforts_cache
    ok, seen = [], set()
    for level in _EFFORT_CANDIDATES:
        try:
            out = tokenizer.hf_render_chat_template(
                probe, add_generation_prompt = True, enable_thinking = True,
                reasoning_effort = level)
        except Exception:                                # noqa: BLE001
            continue                                     # the template refused it
        ok.append(level)
        seen.add(out)
    # If every level renders identically the template is ignoring the argument,
    # and offering the user a choice that changes nothing is worse than none.
    _efforts_cache = ok if len(seen) > 1 else []
    return _efforts_cache


def template_effort(tokenizer, effort):
    """Kwargs to pass through to the template for `effort`, or nothing.

    "high" is the word the OpenAI API uses and the word the UI offers; this
    template spells the same idea "xhigh" and raises on "high". Translate
    rather than let a valid-looking request blow up in the renderer."""
    if not effort:
        return {}
    levels = supported_efforts(tokenizer)
    if not levels:
        return {}
    want = str(effort).strip().lower()
    if want not in levels:
        want = {"high": "xhigh", "xhigh": "high", "max": "xhigh",
                "minimal": "low", "none": None, "off": None}.get(want)
    return {"reasoning_effort": want} if want in levels else {}


def apply_tool_choice(messages, tools, tool_choice):
    """tool_choice -> (messages, tools) with any forced-choice nudge applied.

    Counted as part of the prompt: counting tokens on the untouched history
    would price a different prompt than the one that gets generated."""
    tools, directive = tool_choice_directive(tool_choice, tools)
    if not directive:
        return messages, tools
    messages = list(messages)
    if messages and messages[0].get("role") == "system":
        # Qwen template allows only ONE leading system message — merge
        first = dict(messages[0])
        c = first.get("content") or ""
        if isinstance(c, list):          # content parts (multimodal-style clients)
            first["content"] = list(c) + [{"type": "text", "text": "\n\n" + directive}]
        else:
            first["content"] = c.rstrip() + "\n\n" + directive
        messages[0] = first
    else:
        messages = [{"role": "system", "content": directive}] + messages
    return messages, tools


def build_inputs(tokenizer, messages, tools, enable_thinking, reasoning_effort):
    """History -> (input_ids, image embeddings or None).

    The one place a prompt is rendered, so generation and token counting
    cannot drift apart. Images are embedded through the vision tower first and
    their placeholder slots replaced by the embedding's own text alias."""
    messages, image_urls = extract_images(messages)
    if not image_urls:
        return tokenizer.hf_chat_template(
            messages, add_generation_prompt = True,
            enable_thinking = enable_thinking, tools = tools,
            **template_effort(tokenizer, reasoning_effort)), None
    vm = vision["model"]
    if vm is None:
        raise ValueError(f"this server is running text-only ({vision['reason']}); "
                         "remove the image or restart with VISION=auto")
    images = [decode_image(u) for u in image_urls]
    # GPU work: keep it out of the way of a running generation.
    with gen_lock:
        embeddings = [vm.get_image_embeddings(tokenizer = tokenizer, image = img)
                      for img in images]
    rendered = tokenizer.hf_render_chat_template(
        messages, add_generation_prompt = True,
        enable_thinking = enable_thinking, tools = tools,
        **template_effort(tokenizer, reasoning_effort))
    n = rendered.count(IMAGE_TRIPLE)
    if n != len(embeddings):
        raise ValueError(f"chat template rendered {n} image slot(s) for "
                         f"{len(embeddings)} image(s)")
    for e in embeddings:   # alias -> <|vision_start|> + N image tokens + <|vision_end|>
        rendered = rendered.replace(IMAGE_TRIPLE, e.text_alias, 1)
    return tokenizer.encode(rendered, encode_special_tokens = True,
                            embeddings = embeddings), embeddings


class StreamSplitter:
    """Raw generated text -> ("reasoning"|"content"|"call", payload) events.

    The model emits reasoning, reply and tool calls in one undelimited stream:
    generation starts inside `<think>` when thinking is on, calls arrive as XML
    in the middle of the text, and `<tool_call>` can span chunk boundaries.
    Both wire formats need the same three answers out of it, so the parsing -
    including the holdback that keeps a half-arrived marker out of the reply -
    lives here once and each protocol renders the events in its own shape.

    Feed chunks with feed(); feed(final = True) releases the holdback."""
    def __init__(self, enable_thinking, schemas = None):
        self.pending = ""
        # With thinking on the template ends the prompt with "<think>" and
        # generation starts inside it. With thinking off it emits an empty
        # "<think></think>" pair instead, so the first token is already the
        # answer - starting in_think True there would swallow the reply into a
        # reasoning block nobody asked for.
        self.in_think = bool(enable_thinking)
        self.schemas = schemas or None

    def feed(self, chunk = "", final = False):
        self.pending += chunk
        while True:
            if self.in_think:
                close = self.pending.find(THINK_CLOSE)
                if close >= 0:
                    head = self.pending[:close]
                    self.pending = self.pending[close + len(THINK_CLOSE):]
                    if head.strip():
                        yield ("reasoning", head.lstrip("\n"))
                    self.in_think = False
                    continue
                cut = len(self.pending) if final else max(0, len(self.pending) - HOLD_BACK)
                piece, self.pending = self.pending[:cut], self.pending[cut:]
                # `if piece`, not `if piece.strip()`: a chunk that is nothing
                # but whitespace is still part of the reasoning, and dropping
                # it silently glues the words around it together.
                if piece:
                    yield ("reasoning", piece)
                return
            if TOOL_CALL_OPEN in self.pending:
                head, rest = self.pending.split(TOOL_CALL_OPEN, 1)
                if head.strip() or (final and head):
                    yield ("content", head)
                if TOOL_CALL_CLOSE in rest:
                    block, self.pending = rest.split(TOOL_CALL_CLOSE, 1)
                    _, calls = parse_tool_calls(
                        TOOL_CALL_OPEN + block + TOOL_CALL_CLOSE, self.schemas)
                    for c in calls:
                        yield ("call", c)
                    continue
                # unterminated call: final -> implicit close, else hold
                if final and "<function=" in rest:
                    _, calls = parse_tool_calls(TOOL_CALL_OPEN + rest, self.schemas)
                    for c in calls:
                        yield ("call", c)
                    self.pending = ""
                else:
                    self.pending = TOOL_CALL_OPEN + rest
                return
            cut = len(self.pending) if final else max(0, len(self.pending) - HOLD_BACK)
            piece, self.pending = self.pending[:cut], self.pending[cut:]
            if piece:
                yield ("content", piece)
            return


def generate_full(generator, tokenizer, messages, max_tokens, temperature,
                  top_p, top_k, seed, tools, tool_choice = None, stop = None,
                  on_text = None, enable_thinking = True, should_stop = None,
                  reasoning_effort = None, on_prompt = None):
    """Blocking generation; returns (text, tool_calls, finish, p_toks, o_toks,
    reasoning, content).

    on_prompt is called once with the prompt token count, before the first
    token: a streaming client that reports input tokens in its opening frame
    has no other way to know them."""
    schemas = build_tool_schemas(tools)
    messages, tools = apply_tool_choice(messages, tools, tool_choice)
    input_ids, embeddings = build_inputs(tokenizer, messages, tools,
                                         enable_thinking, reasoning_effort)
    prompt_toks = int(input_ids.shape[-1])
    if on_prompt is not None:
        on_prompt(prompt_toks)
    from exllamav3.generator.sampler.presets import ComboSampler
    from exllamav3 import Job
    forced_choice = tool_choice not in (None, "auto", "none")
    reason = "max_new_tokens"
    text = ""

    def run_once():
        nonlocal text, reason
        text = ""
        reason = "max_new_tokens"
        sampler = ComboSampler(temperature = temperature, top_k = top_k, top_p = top_p)
        stop_conditions = ["<|im_end|>", tokenizer.eos_token_id] + (stop or [])
        job = Job(input_ids = input_ids, max_new_tokens = max_tokens,
                  stop_conditions = stop_conditions,
                  sampler = sampler, seed = seed,
                  embeddings = embeddings)
        prefill_seen = 0
        with gen_lock:
            generator.enqueue(job)
            while generator.num_remaining_jobs():
                # Stop has to reach the GPU, not just the socket. One check per
                # decode step is a few milliseconds of latency and releases the
                # job's cache pages immediately, so the next request is not
                # queued behind a reply nobody is reading any more.
                if should_stop is not None and should_stop():
                    generator.cancel(job)
                    reason = "cancelled"
                    break
                for r in generator.iterate():
                    if r.get("stage") == "prefill":
                        curr = int(r.get("curr_progress") or 0)
                        if curr > prefill_seen:
                            _bump_stats(prompt=curr - prefill_seen)
                            prefill_seen = curr
                    elif _result_new_tokens(r):
                        _bump_stats(completion=_result_new_tokens(r))
                    chunk = r.get("text", "")
                    if chunk:
                        text += chunk
                        if on_text is not None:
                            on_text(chunk)
                    if r.get("eos"):
                        reason = r.get("eos_reason", reason)
            if reason != "cancelled" and prefill_seen < prompt_toks:
                _bump_stats(prompt=prompt_toks - prefill_seen)
        return job

    job = run_once()
    # Forced tool_choice is a prompt nudge; at temperature > 0 the model can
    # occasionally skip the call. One greedy retry makes it deterministic -
    # but not after a cancel, or Stop would start a second generation.
    if reason != "cancelled" and forced_choice and not parse_tool_calls(text, schemas)[1]:
        temperature = 0.0
        job = run_once()
    seq = job.sequences[0]
    out_toks = int(seq.sequence_ids.seq_len - prompt_toks)
    content, calls = parse_tool_calls(text, schemas)
    if calls:
        finish = "tool_calls"
    else:
        finish = {"max_new_tokens": "length", "eos": "stop",
                  "stop_condition": "stop", "banned": "content_filter",
                  "cancelled": "stop"}.get(reason, "stop")
    reasoning, content = split_reasoning(content)
    return text, calls, finish, prompt_toks, out_toks, reasoning, content


async def models(request):
    """The model's own entry - and what it can actually do.

    This used to publish an id and a context length and nothing else, which
    made the server opaque to every client that asks /models what it supports:
    a client probing for reasoning levels and modalities found neither, and
    reported "this provider did not say which levels it takes" - while the
    template on this side accepts three of them.

    The key names are the shapes hosted APIs use; nothing here is invented for
    this kit alone. tools/dsh.py reads exactly this row to write the harness's
    provider route, so what the harness offers is what actually loaded rather
    than what .env hoped for.
    """
    ctx = stats.get("context_length")
    row = {
        "id": MODEL_ID,
        "object": "model",
        "owned_by": "exl3",
        **({"max_model_len": ctx} if ctx else {}),
    }
    tokenizer = request.app.get("tokenizer")
    if tokenizer is not None:
        try:
            levels = supported_efforts(tokenizer)
        except Exception:                            # noqa: BLE001
            levels = []
        if levels:
            row["supported_reasoning_efforts"] = levels
    row["architecture"] = {
        "input_modalities": ["text", "image"] if vision["model"] else ["text"],
        "output_modalities": ["text"],
    }
    # One row, two envelopes. An Anthropic client asking what is loaded gets
    # the Anthropic one; everything else gets the OpenAI one - including
    # tools/dsh.py, which reads this row to configure the harness.
    if wants_anthropic(request):
        return anthropic_json(anthropic_model_list(row))
    return web.json_response({"object": "list", "data": [row]})


async def health(request):
    with stats_lock:
        return web.json_response({
            "ok": True,
            "busy": gen_lock.locked(),
            "backend": "exl3",
            "prompt_tokens_total": stats["prompt_tokens_total"],
            "completion_tokens_total": stats["completion_tokens_total"],
            "context_length": stats["context_length"],
            "vision": vision["model"] is not None,
        })


def parse_request(body):
    messages = body.get("messages")
    if not messages or not isinstance(messages, list):
        return None, "`messages` (list) is required"
    max_tokens = int(body.get("max_tokens") or
                     body.get("max_completion_tokens") or 1024)
    temperature = float(body.get("temperature", 0.6))
    top_p = float(body.get("top_p", 0.95))
    top_k = int(body.get("top_k", 20))
    seed = body.get("seed")
    tools = body.get("tools") or None
    stop = body.get("stop")
    if isinstance(stop, str):
        stop = [stop]
    elif not isinstance(stop, list):
        stop = None

    # How much the model should think, in two spellings a client may already
    # have: chat_template_kwargs.enable_thinking (what vLLM and SGLang accept,
    # and what this model's own template reads) and OpenAI's reasoning_effort.
    # Only "off" is exact - the template emits an empty <think></think> pair and
    # the model has nothing to reason in. The effort levels are a request the
    # model can decline: this engine has no way to cap thinking mid-generation,
    # so they are passed on as guidance rather than enforced. Saying so here
    # keeps the UI from promising a hard limit it cannot deliver.
    kwargs = body.get("chat_template_kwargs") or {}
    enable_thinking = kwargs.get("enable_thinking")
    effort = str(body.get("reasoning_effort") or "").strip().lower()
    if enable_thinking is None:
        enable_thinking = effort not in ("none", "off", "minimal")
    return dict(
        messages = normalize_messages(messages),
        max_tokens = max_tokens, temperature = temperature,
        top_p = top_p, top_k = top_k,
        seed = int(seed) if seed is not None else None,
        tools = tools,
        tool_choice = body.get("tool_choice"),
        stop = stop,
        stream = bool(body.get("stream", False)),
        include_usage = bool((body.get("stream_options") or {}).get("include_usage")),
        model_id = body.get("model", MODEL_ID),
        enable_thinking = bool(enable_thinking),
        reasoning_effort = effort or None,
    ), None


async def run_generation(req, generator, tokenizer):
    """One non-streaming request, run off the event loop.

    Returns (result, error): a failed prompt is a 400, not a 500 - the cache
    and the context are the two things a caller can actually do something
    about - while anything else keeps propagating."""
    try:
        return await asyncio.to_thread(
            generate_full, generator, tokenizer, req["messages"],
            req["max_tokens"], req["temperature"], req["top_p"], req["top_k"],
            req["seed"], req["tools"], req["tool_choice"], req["stop"],
            None, req["enable_thinking"], None,
            req["reasoning_effort"]), None
    except AssertionError as e:
        return None, f"context/cache: {e}"
    except ValueError as e:
        return None, str(e)


async def client_watcher(request, gone):
    """A reply that is still in prefill writes nothing, so a failed write would
    never notice the browser had gone. Watch the socket instead.

    `gone` is what the worker thread polls, so this is also the only path by
    which the Stop button reaches the GPU. Both protocols use it."""
    try:
        while not gone.is_set():
            transport = request.transport
            if transport is None or transport.is_closing():
                gone.set()
                return
            await asyncio.sleep(0.2)
    except asyncio.CancelledError:
        pass


async def chat_completions(request):
    app = request.app
    generator, tokenizer = app["generator"], app["tokenizer"]
    try:
        body = await request.json()
    except web.HTTPRequestEntityTooLarge:
        # aiohttp enforces client_max_size inside request.json(); without this
        # branch it falls into the generic handler below and gets misreported
        # as "invalid JSON" (400) even though the body parsed fine.
        return web.json_response(
            {"error": {"message": f"request body exceeds {request.app['max_body_mb']} MiB limit",
                       "type": "invalid_request_error",
                       "code": "request_entity_too_large"}},
            status = 413)
    except Exception:
        return web.json_response({"error": {"message": "invalid JSON"}}, status = 400)
    req, err = parse_request(body)
    if err:
        return web.json_response({"error": {"message": err}}, status = 400)

    if not req["stream"]:
        result, err = await run_generation(req, generator, tokenizer)
        if err:
            return web.json_response(
                {"error": {"message": err, "type": "invalid_request_error"}},
                status = 400)
        text, calls, finish, ptoks, otoks, reasoning, content = result
        msg = {"role": "assistant", "content": content or None}
        if reasoning:
            msg["reasoning_content"] = reasoning
        if calls:
            msg["tool_calls"] = calls
        return web.json_response({
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion", "created": int(time.time()),
            "model": req["model_id"],
            "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
            "usage": {"prompt_tokens": ptoks, "completion_tokens": otoks,
                      "total_tokens": ptoks + otoks},
        })

    # ---- streaming (SSE) ----
    resp = web.StreamResponse(headers = {
        "Content-Type": "text/event-stream", "Cache-Control": "no-cache",
        "Connection": "keep-alive"})
    await resp.prepare(request)
    cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    model_id = req["model_id"]
    req_schemas = build_tool_schemas(req["tools"])

    async def run():
        loop = asyncio.get_event_loop()
        queue = asyncio.Queue()
        # The client hanging up IS the Stop button - there is no other message
        # for it in the OpenAI protocol. Nothing used to tell the worker thread,
        # so it generated to max_tokens on the GPU with nobody listening, and
        # because generation is serialised the user's *next* message queued
        # behind the reply they had just cancelled.
        gone = threading.Event()

        def on_text(chunk):
            loop.call_soon_threadsafe(queue.put_nowait, ("delta", chunk))

        forced_choice = req["tool_choice"] not in (None, "auto", "none")

        def worker():
            try:
                # generate_full's own tuple goes on the queue unrepacked: both
                # protocols unpack it the same way, so neither can agree with a
                # reordering the other never made.
                result = generate_full(
                    generator, tokenizer, req["messages"], req["max_tokens"],
                    req["temperature"], req["top_p"], req["top_k"],
                    req["seed"], req["tools"], req["tool_choice"], req["stop"],
                    on_text = None if forced_choice else on_text,
                    enable_thinking = req["enable_thinking"],
                    reasoning_effort = req["reasoning_effort"],
                    should_stop = gone.is_set)
                loop.call_soon_threadsafe(queue.put_nowait, ("done", result))
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, ("error", str(e)))
        loop.run_in_executor(None, worker)

        async def send(delta, finish = None):
            obj = {"id": cid, "object": "chat.completion.chunk",
                   "created": int(time.time()), "model": model_id,
                   "choices": [{"index": 0, "delta": delta,
                                "finish_reason": finish}]}
            try:
                await resp.write(f"data: {json.dumps(obj)}\n\n".encode())
            except (ConnectionError, RuntimeError):
                gone.set()          # tell the GPU, not just the event loop
                raise

        finish, calls_emitted = None, False
        call_idx = [0]
        splitter = StreamSplitter(req["enable_thinking"], req_schemas)

        async def send_call(c):
            nonlocal calls_emitted
            calls_emitted = True
            await send({"tool_calls": [dict(c, index = call_idx[0])]})
            call_idx[0] += 1

        async def flush_pending(chunk = "", final = False):
            """Render what the splitter can parse; it holds back the tail that
            might still be half a marker."""
            for kind, payload in splitter.feed(chunk, final = final):
                if kind == "call":
                    await send_call(payload)
                elif kind == "reasoning":
                    await send({"reasoning_content": payload})
                else:
                    await send({"content": payload})

        async def consume():
            while True:
                kind, payload = await queue.get()
                if kind == "error":
                    await resp.write(
                        f'data: {json.dumps({"error": {"message": payload}})}\n\n'.encode())
                    break
                if kind == "delta":
                    await flush_pending(payload)
                elif kind == "done":
                    _text, calls, finish, ptoks, otoks, reasoning, content = payload
                    await flush_pending(final = True)
                    if forced_choice:
                        # Buffered path (no deltas were streamed): emit the
                        # authoritative complete result as deltas.
                        if reasoning:
                            await send({"reasoning_content": reasoning})
                        if content:
                            await send({"content": content})
                    if not calls_emitted and calls:
                        for c in calls:
                            await send_call(c)
                    await send({}, finish = finish)
                    if req["include_usage"]:
                        tail = {"id": cid, "object": "chat.completion.chunk",
                                "created": int(time.time()), "model": model_id,
                                "choices": [],
                                "usage": {"prompt_tokens": ptoks,
                                          "completion_tokens": otoks,
                                          "total_tokens": ptoks + otoks}}
                        await resp.write(f"data: {json.dumps(tail)}\n\n".encode())
                    await resp.write(b"data: [DONE]\n\n")
                    break
            await resp.write_eof()

        watcher = asyncio.ensure_future(client_watcher(request, gone))
        try:
            await consume()
        finally:
            gone.set()              # the turn is over either way
            watcher.cancel()
    try:
        await run()
    except (ConnectionError, RuntimeError):
        # the client went away mid-write; `gone` has already told the worker
        pass
    return resp


# ---------------------------------------------------------------------------
# Anthropic Messages API
#
# The same model on the same port, in a second wire format. Claude Code,
# Cline, Continue, Zed and the Anthropic SDKs speak the Messages API, not Chat
# Completions. This is a translation layer rather than a second engine: the
# request is converted to the OpenAI-shaped body parse_request() already
# validates, generation runs once, and the result is converted back into
# Anthropic content blocks - and, when asked for, Anthropic SSE events.
#
# What a strict client checks was read off the client, not guessed at. From
# anthropic-sdk-python 1.7.0: every frame must carry an `event:` line (its SSE
# reader dispatches on the event name, not on the JSON), message_start must be
# first, message_start.message.usage must carry input_tokens *and*
# output_tokens, message_delta.usage must carry output_tokens, every thinking
# block needs a signature, and content_block_stop must arrive for each index
# that was opened.
#
# No authentication is invented here. `x-api-key` and `Authorization: Bearer`
# are accepted and ignored, exactly as /v1 ignores them, and
# `anthropic-version` is not required - it only selects the /v1/models shape.
ANTHROPIC_VERSION = "2023-06-01"

# finish_reason -> stop_reason. Anthropic's enum is closed; these are the
# members this engine can produce. A stop of ours that matches a caller's
# stop_sequence is reported as end_turn: the engine does not say which of the
# job's stop conditions fired, and guessing would be worse than the plain
# answer. stop_sequence stays null to match.
_STOP_REASONS = {"tool_calls": "tool_use", "length": "max_tokens",
                 "content_filter": "refusal", "stop": "end_turn"}


class _Unsupported(Exception):
    """A request asks for something this server cannot render."""


def _request_id():
    return f"req_{uuid.uuid4().hex[:24]}"


def anthropic_json(payload, status = 200):
    """A reply in the Anthropic shape, with the request id Anthropic always
    sends - the SDKs keep it as `_request_id` and prompts to quote it."""
    return web.json_response(payload, status = status,
                             headers = {"request-id": _request_id()})


def anthropic_error(message, etype = "invalid_request_error", status = 400):
    """The Anthropic error envelope, which is not the OpenAI one. The request
    id is in the body too, which is where it gets pasted from."""
    rid = _request_id()
    return web.json_response(
        {"type": "error", "error": {"type": etype, "message": message},
         "request_id": rid},
        status = status, headers = {"request-id": rid})


def _anth_signature():
    """A stand-in for the signature Anthropic puts on a thinking block.

    The field is opaque - clients store it and hand it back verbatim, and the
    SDK models it as a required plain string - so a value of the right shape is
    all a client can check. Nothing verifies it here: thinking blocks in the
    history are rendered back to the model as its own reasoning rather than
    checked against this."""
    return base64.b64encode(os.urandom(48)).decode("ascii")


def _anth_blocks(content):
    """An Anthropic `content` -> a list of blocks. A bare string is one text
    block, which is what the Messages API says it is shorthand for."""
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return [b for b in content if isinstance(b, dict)]


def _anth_text_of(content):
    """Anthropic content -> plain text (text blocks joined, others dropped)."""
    if isinstance(content, str):
        return content
    return "\n".join(str(b.get("text") or "") for b in _anth_blocks(content)
                     if b.get("type") == "text")


def _anth_image_part(block):
    """Anthropic image block -> an OpenAI image_url part.

    Both spellings become a data: URL or an http(s) URL, which is exactly what
    the vision path already decodes, so an attached picture needs no second
    code path to reach the tower."""
    src = block.get("source") or {}
    kind = src.get("type")
    if kind == "base64":
        media = src.get("media_type") or "image/png"
        return {"type": "image_url",
                "image_url": {"url": f"data:{media};base64,{src.get('data') or ''}"}}
    if kind == "url":
        return {"type": "image_url", "image_url": {"url": src.get("url") or ""}}
    raise _Unsupported(f"image source {kind!r} is not supported "
                       "(the Files API is not served here; send base64 or a URL)")


def _anth_tool_result_content(content):
    """tool_result `content` -> an OpenAI tool message's content.

    A result may carry images - an agent that reads a screenshot returns one -
    and the chat template renders pictures inside <tool_response> as readily as
    anywhere else, so they travel as content parts instead of being flattened
    away."""
    blocks = _anth_blocks(content)
    if not blocks:
        return ""
    if all(b.get("type") == "text" for b in blocks):
        return _anth_text_of(blocks)
    parts = []
    for b in blocks:
        if b.get("type") == "text":
            parts.append({"type": "text", "text": str(b.get("text") or "")})
        elif b.get("type") == "image":
            parts.append(_anth_image_part(b))
        else:
            raise _Unsupported(f"{b.get('type')!r} inside a tool_result is not supported")
    return parts


def _anth_system(system):
    """Top-level `system` -> one system message's text.

    The Messages API has no system *role*: the prompt is a top-level field, as
    a string or as text blocks. The Qwen template allows exactly one leading
    system message, so several blocks are joined into it."""
    if not system:
        return ""
    if isinstance(system, str):
        return system
    for b in _anth_blocks(system):
        if b.get("type") != "text":
            raise _Unsupported(f"{b.get('type')!r} in `system` is not supported")
    return _anth_text_of(system)


def anthropic_to_openai(body, require_max_tokens = True):
    """Anthropic Messages request -> OpenAI chat-completions body.

    Returns (body, error). The conversion stops at the same door every other
    client goes through - parse_request() - so the two wire formats cannot
    disagree about defaults, limits or validation."""
    messages = body.get("messages")
    if not messages or not isinstance(messages, list):
        return None, "`messages` (list) is required"

    max_tokens = body.get("max_tokens")
    if max_tokens is None and require_max_tokens:
        return None, "`max_tokens` is required"
    if max_tokens is None:
        max_tokens = 1024           # count_tokens never generates; any value will do
    try:
        max_tokens = int(max_tokens)
    except (TypeError, ValueError):
        return None, "`max_tokens` must be an integer"
    if require_max_tokens and max_tokens < 1:
        return None, "`max_tokens` must be at least 1"

    try:
        out = []
        # The Qwen template allows exactly one system message and only at the
        # start ("System message must be at the beginning"), but the Messages
        # API puts the prompt in a top-level field and Claude Code adds further
        # system messages *inside* `messages` (the mid-conversation-system
        # beta). Their text is collected here and joined into the one system
        # message the template will accept - as system-level instruction, not
        # as a user turn, which would put words in the user's mouth.
        system_parts = [_anth_system(body.get("system"))]
        for m in messages:
            m = m if isinstance(m, dict) else {}
            role = m.get("role")
            if role not in ("user", "assistant", "system"):
                return None, f"`messages[].role` must be user or assistant (got {role!r})"
            if role == "system":
                system_parts.append(_anth_text_of(m.get("content")))
                continue
            blocks = _anth_blocks(m.get("content"))
            if role == "assistant":
                texts, calls, thinking = [], [], []
                for b in blocks:
                    t = b.get("type")
                    if t == "text":
                        texts.append(str(b.get("text") or ""))
                    elif t == "thinking":
                        thinking.append(str(b.get("thinking") or ""))
                    elif t == "redacted_thinking":
                        continue        # encrypted; there is nothing usable inside
                    elif t == "tool_use":
                        calls.append({
                            "id": b.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                            "type": "function",
                            "function": {"name": b.get("name"),
                                         "arguments": json.dumps(b.get("input") or {})}})
                    else:
                        raise _Unsupported(f"{t!r} in an assistant turn is not supported")
                msg = {"role": "assistant", "content": "\n".join(texts)}
                # The template renders a `reasoning_content` on an assistant
                # turn as that turn's <think> block, so returned thinking goes
                # back in as the reasoning it was rather than being dropped.
                if thinking:
                    msg["reasoning_content"] = "\n".join(thinking)
                if calls:
                    msg["tool_calls"] = calls
                out.append(msg)
                continue

            # user: tool results become their own turns, and they come first -
            # the same order the Messages API requires of the blocks.
            results, parts = [], []
            for b in blocks:
                t = b.get("type")
                if t == "text":
                    parts.append({"type": "text", "text": str(b.get("text") or "")})
                elif t == "image":
                    parts.append(_anth_image_part(b))
                elif t == "tool_result":
                    results.append({"role": "tool",
                                    "tool_call_id": b.get("tool_use_id") or "",
                                    "content": _anth_tool_result_content(b.get("content"))})
                elif t == "thinking":
                    continue            # only valid on an assistant turn; ignore
                else:
                    raise _Unsupported(f"{t!r} in a user turn is not supported")
            out.extend(results)
            if parts:
                if all(p.get("type") == "text" for p in parts):
                    out.append({"role": "user",
                                "content": "\n".join(p["text"] for p in parts)})
                else:
                    out.append({"role": "user", "content": parts})
    except _Unsupported as e:
        return None, str(e)
    system = "\n\n".join(p for p in system_parts if p.strip())
    if system:
        out.insert(0, {"role": "system", "content": system})

    tools = []
    for t in body.get("tools") or []:
        t = t if isinstance(t, dict) else {}
        ttype = t.get("type")
        if ttype not in (None, "custom"):
            return None, (f"tool type {ttype!r} is not supported - this server "
                          "runs client tools only")
        tools.append({"type": "function",
                      "function": {
                          "name": t.get("name"),
                          "description": t.get("description") or "",
                          "parameters": t.get("input_schema")
                                        or {"type": "object", "properties": {}}}})

    choice = body.get("tool_choice") or {}
    if not isinstance(choice, dict):
        choice = {}
    ctype = choice.get("type")
    if ctype == "any":
        tool_choice = "required"        # "any" is Anthropic's must-call-something
    elif ctype == "tool":
        tool_choice = {"type": "function", "function": {"name": choice.get("name")}}
    elif ctype == "none":
        tool_choice = "none"
    else:
        tool_choice = "auto"

    openai = {"model": body.get("model") or MODEL_ID,
              "messages": out, "max_tokens": max_tokens,
              "stream": bool(body.get("stream"))}
    for key in ("temperature", "top_p", "top_k"):
        if body.get(key) is not None:
            openai[key] = body[key]
    if body.get("stop_sequences"):
        openai["stop"] = body["stop_sequences"]
    if tools:
        openai["tools"] = tools
        openai["tool_choice"] = tool_choice
    thinking = body.get("thinking") if isinstance(body.get("thinking"), dict) else {}
    if thinking.get("type") == "disabled":
        openai["chat_template_kwargs"] = {"enable_thinking": False}
    elif thinking.get("type") in ("enabled", "adaptive"):
        openai["chat_template_kwargs"] = {"enable_thinking": True}
    # output_config.effort is the same request the OpenAI path spells
    # reasoning_effort, and this model's template acts on it - so it is passed
    # on rather than dropped.
    effort = (body.get("output_config") or {}).get("effort") \
        if isinstance(body.get("output_config"), dict) else None
    if effort:
        openai["reasoning_effort"] = effort
    # `metadata`, `service_tier`, `cache_control`, `context_management` and
    # unknown future fields are deliberately not forwarded: nothing here can
    # act on them, and accepting a knob we ignore is how a caller ends up
    # trusting it.
    return openai, None


def anthropic_thinking_omitted(body):
    """`thinking.display: "omitted"` - the client wants the block and its
    signature but not the reasoning text. Claude Code asks for this, and
    honouring it is the difference between every later turn carrying this
    turn's whole thought process and carrying none of it."""
    t = body.get("thinking")
    return bool(isinstance(t, dict) and t.get("display") == "omitted")


def anthropic_no_parallel(body):
    """Does the request forbid parallel tool use?

    The engine cannot cap that mid-generation, so it is enforced where it can
    be: the response keeps the first tool call and drops the rest."""
    choice = body.get("tool_choice")
    return bool(isinstance(choice, dict) and choice.get("disable_parallel_tool_use"))


def anthropic_message(model_id, content, reasoning, calls, finish, ptoks, otoks,
                      no_parallel = False, thinking_omitted = False):
    """A finished generation -> an Anthropic Message object."""
    blocks = []
    if reasoning:
        blocks.append({"type": "thinking",
                       "thinking": "" if thinking_omitted else reasoning,
                       "signature": _anth_signature()})
    if content:
        blocks.append({"type": "text", "text": content})
    for c in (calls[:1] if no_parallel else calls):
        fn = c.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except ValueError:
            args = {}
        blocks.append({"type": "tool_use",
                       "id": c.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                       "name": fn.get("name"),
                       "input": args if isinstance(args, dict) else {}})
    if not blocks:
        blocks.append({"type": "text", "text": ""})   # a reply is never no blocks
    return {"id": f"msg_{uuid.uuid4().hex[:24]}", "type": "message",
            "role": "assistant", "model": model_id, "content": blocks,
            "stop_reason": _STOP_REASONS.get(finish, "end_turn"),
            "stop_sequence": None,
            # No cache fields: this server has no prompt cache, and reporting
            # zeroes for one would describe a mechanism the caller cannot use.
            "usage": {"input_tokens": ptoks, "output_tokens": otoks}}


def wants_anthropic(request):
    """Is this a client that speaks the Messages API?

    The version header is the one every Anthropic client sends and no OpenAI
    client does, which is what /v1/models keys off to pick a shape."""
    return "anthropic-version" in request.headers


def anthropic_model_list(row):
    """The OpenAI /v1/models row -> the Anthropic list response."""
    entry = anthropic_model_row(row)
    return {"data": [entry], "has_more": False,
            "first_id": entry["id"], "last_id": entry["id"]}


def anthropic_model_row(row):
    """The OpenAI /v1/models row -> an Anthropic ModelInfo.

    created_at is an epoch value, which the docs allow when a release date is
    unknown - the honest answer for a quant someone exported themselves."""
    entry = {"type": "model", "id": row["id"], "display_name": row["id"],
             "created_at": "1970-01-01T00:00:00Z"}
    if row.get("max_model_len"):
        entry["max_input_tokens"] = row["max_model_len"]
    return entry


async def model_retrieve(request):
    """GET /v1/models/{id} - served in whichever shape asked for it."""
    model_id = request.match_info["model_id"]
    row = {"id": MODEL_ID}
    ctx = stats.get("context_length")
    if ctx:
        row["max_model_len"] = ctx
    if wants_anthropic(request):
        if model_id != MODEL_ID:
            return anthropic_error(f"model {model_id!r} not found",
                                   "not_found_error", 404)
        return anthropic_json(anthropic_model_row(row))
    if model_id != MODEL_ID:
        return web.json_response({"error": {"message": f"model {model_id!r} not found"}},
                                 status = 404)
    return web.json_response({"id": MODEL_ID, "object": "model", "owned_by": "exl3"})


async def anthropic_stream(request, req, generator, tokenizer, no_parallel,
                           thinking_omitted = False):
    """The Anthropic SSE protocol, rendered from the same generation.

    The event order is fixed and the SDKs depend on it: message_start, then per
    block start/delta*/stop, then message_delta - which is where the stop
    reason lives, not in message_start - and message_stop. input_tokens is
    unknowable before the prompt is rendered, so generation reports the count
    before the first token rather than after the last, and the opening frame
    carries the real number instead of a guess."""
    resp = web.StreamResponse(headers = {
        "Content-Type": "text/event-stream", "Cache-Control": "no-cache",
        "Connection": "keep-alive"})
    await resp.prepare(request)
    mid = f"msg_{uuid.uuid4().hex[:24]}"
    model_id = req["model_id"]
    schemas = build_tool_schemas(req["tools"])

    async def send(etype, payload):
        await resp.write(f"event: {etype}\ndata: {json.dumps(payload)}\n\n".encode())

    async def run():
        loop = asyncio.get_event_loop()
        queue = asyncio.Queue()
        # The client hanging up is the Stop button in both protocols, and the
        # only thing that tells the GPU is this event.
        gone = threading.Event()
        forced_choice = req["tool_choice"] not in (None, "auto", "none")

        def on_text(chunk):
            loop.call_soon_threadsafe(queue.put_nowait, ("delta", chunk))

        def worker():
            try:
                result = generate_full(
                    generator, tokenizer, req["messages"], req["max_tokens"],
                    req["temperature"], req["top_p"], req["top_k"], req["seed"],
                    req["tools"], req["tool_choice"], req["stop"],
                    on_text = None if forced_choice else on_text,
                    enable_thinking = req["enable_thinking"],
                    reasoning_effort = req["reasoning_effort"],
                    should_stop = gone.is_set,
                    on_prompt = lambda n: loop.call_soon_threadsafe(
                        queue.put_nowait, ("prompt", n)))
                loop.call_soon_threadsafe(queue.put_nowait, ("done", result))
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, ("error", str(e)))
        loop.run_in_executor(None, worker)

        splitter = StreamSplitter(req["enable_thinking"], schemas)
        state = {"index": -1, "open": None, "started": False, "calls": 0}

        async def close_block():
            if state["open"] is None:
                return
            if state["open"] == "thinking":
                # Exactly one signature_delta, immediately before the stop.
                # This is the only place a client ever gets a signature, and a
                # thinking block without one is a block it cannot hand back.
                await send("content_block_delta",
                           {"type": "content_block_delta", "index": state["index"],
                            "delta": {"type": "signature_delta",
                                      "signature": _anth_signature()}})
            await send("content_block_stop",
                       {"type": "content_block_stop", "index": state["index"]})
            state["open"] = None

        async def open_block(kind, block):
            await close_block()
            state["index"] += 1
            state["open"] = kind
            await send("content_block_start",
                       {"type": "content_block_start", "index": state["index"],
                        "content_block": block})

        async def emit(kind, payload):
            if kind == "reasoning":
                if state["open"] != "thinking":
                    await open_block("thinking", {"type": "thinking", "thinking": "",
                                                  "signature": ""})
                if not thinking_omitted:
                    await send("content_block_delta",
                               {"type": "content_block_delta", "index": state["index"],
                                "delta": {"type": "thinking_delta", "thinking": payload}})
                return
            if state["open"] != "text":
                await open_block("text", {"type": "text", "text": ""})
            await send("content_block_delta",
                       {"type": "content_block_delta", "index": state["index"],
                        "delta": {"type": "text_delta", "text": payload}})

        async def emit_call(c):
            """A tool call is its own block, closed as soon as its input is out
            - so two calls in a row stay two blocks."""
            if state["calls"] and no_parallel:
                return
            state["calls"] += 1
            fn = c.get("function") or {}
            await open_block("tool_use",
                             {"type": "tool_use",
                              "id": c.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                              "name": fn.get("name"), "input": {}})
            await send("content_block_delta",
                       {"type": "content_block_delta", "index": state["index"],
                        "delta": {"type": "input_json_delta",
                                  "partial_json": fn.get("arguments") or "{}"}})
            await close_block()

        async def drain(chunk = "", final = False):
            for kind, payload in splitter.feed(chunk, final = final):
                if kind == "call":
                    await emit_call(payload)
                else:
                    await emit(kind, payload)

        async def start_message(ptoks):
            state["started"] = True
            await send("message_start", {
                "type": "message_start",
                "message": {"id": mid, "type": "message", "role": "assistant",
                            "model": model_id, "content": [],
                            "stop_reason": None, "stop_sequence": None,
                            "usage": {"input_tokens": ptoks, "output_tokens": 0}}})

        async def consume():
            while True:
                kind, payload = await queue.get()
                if kind == "prompt":
                    if not state["started"]:
                        await start_message(payload)
                    continue
                if kind == "error":
                    await send("error", {"type": "error",
                                         "error": {"type": "api_error",
                                                   "message": payload}})
                    break
                if kind == "delta":
                    await drain(payload)
                    continue
                _text, calls, finish, ptoks, otoks, reasoning, content = payload
                if not state["started"]:
                    await start_message(ptoks)
                await drain(final = True)
                if forced_choice:
                    # Buffered path (no deltas were streamed): render the
                    # authoritative complete result instead.
                    if reasoning:
                        await emit("reasoning", reasoning)
                    if content:
                        await emit("content", content)
                if not state["calls"] and calls:
                    for c in calls:
                        await emit_call(c)
                await close_block()
                await send("message_delta", {
                    "type": "message_delta",
                    "delta": {"stop_reason": _STOP_REASONS.get(finish, "end_turn"),
                              "stop_sequence": None},
                    # output_tokens only. Usage on this event is cumulative and
                    # would overwrite message_start's, so re-sending the same
                    # input count a frame later buys nothing and risks saying
                    # something else.
                    "usage": {"output_tokens": otoks}})
                await send("message_stop", {"type": "message_stop"})
                break
            await resp.write_eof()

        watcher = asyncio.ensure_future(client_watcher(request, gone))
        try:
            await consume()
        finally:
            gone.set()              # the turn is over either way
            watcher.cancel()
    try:
        await run()
    except (ConnectionError, RuntimeError):
        # the client went away mid-write; `gone` has already told the worker
        pass
    return resp


async def anthropic_messages(request):
    """POST /v1/messages."""
    app = request.app
    generator, tokenizer = app["generator"], app["tokenizer"]
    try:
        body = await request.json()
    except web.HTTPRequestEntityTooLarge:
        return anthropic_error(
            f"request body exceeds {app['max_body_mb']} MiB limit",
            "request_too_large", 413)
    except Exception:
        return anthropic_error("invalid JSON")
    openai, err = anthropic_to_openai(body)
    if err:
        return anthropic_error(err)
    req, err = parse_request(openai)
    if err:
        return anthropic_error(err)
    no_parallel = anthropic_no_parallel(body)
    thinking_omitted = anthropic_thinking_omitted(body)

    if not req["stream"]:
        result, err = await run_generation(req, generator, tokenizer)
        if err:
            return anthropic_error(err)
        text, calls, finish, ptoks, otoks, reasoning, content = result
        return anthropic_json(anthropic_message(
            req["model_id"], content, reasoning, calls, finish, ptoks, otoks,
            no_parallel, thinking_omitted))
    return await anthropic_stream(request, req, generator, tokenizer, no_parallel,
                                  thinking_omitted)


async def anthropic_count_tokens(request):
    """POST /v1/messages/count_tokens - what the prompt costs, without running it.

    A real count, not an estimate: the same template renders the same prompt
    generation would use, including any forced-tool-choice directive, and the
    ids are counted once. Claude Code calls this before a turn to decide
    whether it has to compact, so "roughly" would be the wrong answer."""
    app = request.app
    tokenizer = app["tokenizer"]
    try:
        body = await request.json()
    except web.HTTPRequestEntityTooLarge:
        return anthropic_error(
            f"request body exceeds {app['max_body_mb']} MiB limit",
            "request_too_large", 413)
    except Exception:
        return anthropic_error("invalid JSON")
    openai, err = anthropic_to_openai(body, require_max_tokens = False)
    if err:
        return anthropic_error(err)
    req, err = parse_request(openai)
    if err:
        return anthropic_error(err)
    try:
        messages, tools = apply_tool_choice(req["messages"], req["tools"],
                                            req["tool_choice"])
        input_ids, _ = build_inputs(tokenizer, messages, tools,
                                    req["enable_thinking"], req["reasoning_effort"])
    except AssertionError as e:
        return anthropic_error(f"context/cache: {e}")
    except ValueError as e:
        return anthropic_error(str(e))
    return anthropic_json({"input_tokens": int(input_ids.shape[-1])})


LANDING = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{model}</title>
<style>
  :root {{ color-scheme: light dark; --fg:#1c1b1a; --dim:#6b6864; --bg:#faf9f7;
           --card:#fff; --line:#e6e2dc; --accent:#d97757; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --fg:#eeece8; --dim:#9a958e; --bg:#141413; --card:#1d1d1b;
             --line:#302f2c; }} }}
  * {{ box-sizing:border-box }}
  body {{ margin:0; min-height:100vh; display:grid; place-items:center;
          background:var(--bg); color:var(--fg); font:15px/1.55 ui-sans-serif,
          system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; padding:24px }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:14px;
           padding:28px 32px; max-width:34rem; width:100% }}
  h1 {{ font-size:1.15rem; margin:0 0 .35rem; font-weight:600 }}
  p {{ margin:.55rem 0; color:var(--dim) }}
  code {{ font:13px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
          background:color-mix(in srgb, var(--fg) 7%, transparent);
          padding:.12em .4em; border-radius:5px; color:var(--fg) }}
  .dot {{ display:inline-block; width:.5rem; height:.5rem; border-radius:50%;
          background:var(--accent); margin-right:.45rem; vertical-align:.05rem;
          animation:pulse 1.4s ease-in-out infinite }}
  @keyframes pulse {{ 0%,100%{{opacity:1}} 50%{{opacity:.25}} }}
  a {{ color:var(--accent) }}
  hr {{ border:0; border-top:1px solid var(--line); margin:1.25rem 0 }}
</style></head><body><div class="card">
<h1>{model}</h1>
<p>The model is loaded and serving <code>{api}</code> (OpenAI) and
   <code>{anthropic}</code> (Anthropic) &mdash; API key <code>local</code>,
   ignored either way.</p>
<hr>
<p id="s"><span class="dot"></span>Waiting for the DeepSeek Harness at
   <code>{harness}</code> &hellip; it opens here by itself.</p>
<p><small>The harness is a separate process. If it is not running,
   <code>python tools/dsh.py --open</code> starts it, and <code>UI=no</code> in
   <code>.env</code> stops the launcher starting one at all.</small></p>
</div>
<script>
  // The harness mints a token every launch and refuses a browser that arrives
  // without it, so this hands over through /harness - which knows the address
  // the launcher actually read off dsh - rather than to the bare port.
  const say = (html) => {{ document.getElementById("s").innerHTML = html; }};
  async function poll() {{
    try {{
      const r = await fetch("/harness/status", {{ cache: "no-store" }});
      const s = await r.json();
      if (s.ready && s.local) {{ location.href = "/harness"; return; }}
      if (s.ready && !s.local) {{
        say("The harness is running, but it answers only on the computer it "
          + "runs on \u2014 its agent runs commands there and there is no login.");
        return;
      }}
    }} catch (e) {{ /* the server is still coming up */ }}
    setTimeout(poll, 1500);
  }}
  poll();
</script>
</body></html>
"""


def mount_landing(app, args):
    """Serve a small page at `/` saying where everything is.

    The model server used to serve a chat UI here. It serves the harness's
    address instead: first-run setup ends on this port, so something has to be
    at `/` afterwards, and "nothing" would read as a failed launch. The page
    forwards to the harness as soon as the harness answers.

    /v1 is untouched, and a failure here must never stop the model serving."""
    try:
        api = f"http://127.0.0.1:{args.port}/v1"
        anthropic = f"http://127.0.0.1:{args.port}"
        harness = f"http://127.0.0.1:{args.harness_port}/"
        body = LANDING.format(model = MODEL_ID, api = api, anthropic = anthropic,
                              harness = harness).encode("utf-8")

        async def landing(_request):
            return web.Response(body = body, content_type = "text/html",
                                charset = "utf-8",
                                headers = {"Cache-Control": "no-store"})

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        url_file = os.path.join(root, ".dsh", "url")

        def harness_url():
            """The address the launcher read off dsh, token and all, or None.

            Read per request rather than cached: the harness can be started,
            stopped and restarted while this server runs, and each launch has
            its own token."""
            try:
                with open(url_file, encoding="utf-8") as fh:
                    return fh.read().strip() or None
            except OSError:
                return None

        def from_this_computer(request):
            peer = (request.remote or "").strip().strip("[]").split("%")[0]
            return peer in ("127.0.0.1", "::1", "::ffff:127.0.0.1")

        async def harness_status(request):
            return web.json_response({"ready": harness_url() is not None,
                                      "local": from_this_computer(request)},
                                     headers = {"Cache-Control": "no-store"})

        async def harness_open(request):
            """Send a browser to the harness, authenticated.

            Only to a browser on this computer. HOST may be 0.0.0.0 so that
            /v1 serves the network, and the token in that URL is a session on
            an agent that runs commands here - handing it to the network would
            undo the loopback bind the harness itself insists on."""
            if not from_this_computer(request):
                return web.Response(status = 403, content_type = "text/plain",
                                    text = "The harness answers only on the "
                                           "computer it runs on.\n")
            target = harness_url()
            if target is None:
                return web.Response(status = 503, content_type = "text/plain",
                                    text = "The harness is not running.\n")
            raise web.HTTPFound(target, headers = {"Cache-Control": "no-store"})

        for path in ("/", "/index.html"):
            app.router.add_get(path, landing)
        app.router.add_get("/harness", harness_open)
        app.router.add_get("/harness/status", harness_status)
        return True
    except Exception as e:                # noqa: BLE001  (never fatal)
        print(f" !! landing page disabled: {type(e).__name__}: {e}", flush = True)
        print("    The OpenAI API on /v1 is unaffected.", flush = True)
        return False


def main():
    global MODEL_DIR, DRAFT_DIR, PORT, MODEL_ID
    _quiet_triton()
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default = MODEL_DIR)
    ap.add_argument("-dm", "--draft_model", default = DRAFT_DIR,
                    help = "'mtp' for MTP drafting (head inside the "
                           "main checkpoint: no extra weights, much smaller KV footprint) "
                           "or 'none' to disable drafting")
    ap.add_argument("-gs", "--grid_size", type = str, default = "14.7",
                    help = "GPU memory budget in GB (autosplit + process cap), "
                           "or one per GPU, e.g. 14.9,7.2 to split across two. "
                           "14.7 is the 16 GB-card recipe")
    ap.add_argument("--num_draft_tokens", type = int, default = 0,
                    help = "MTP draft length per round (0 = the head's default, 4)")
    ap.add_argument("--dynamic_draft", type = float, default = 0,
                    help = "cut each draft where the head's confidence drops "
                           "below this (e.g. 0.4); 0 = always draft the full length")
    ap.add_argument("--offload_layers", type = str, default = "auto",
                    help = "two GPUs: how many linear-attention blocks go to "
                           "GPU 1 (auto = just enough to fit GPU 0's budget; "
                           "stock = exllamav3's own in-order split)")
    ap.add_argument("-cs", "--cache_size", type = int, default = 199936,
                    help = "KV cache size in tokens (default 199936 ~200k for "
                           "16 GB VRAM; must be a multiple of 256)")
    ap.add_argument("-cq", "--cache_quant", type = str, default = None,
                    help = "Quantized KV cache bits, e.g. 8 or 8,4 (k_bits[,v_bits])")
    ap.add_argument("--model_id", type = str, default = None,
                    help = "id reported by /v1/models and accepted in requests "
                           "(default: the model folder name, lower-cased)")
    ap.add_argument("-p", "--port", type = int, default = PORT)
    ap.add_argument("--host", type = str, default = "0.0.0.0",
                    help = "Interface to bind (use 127.0.0.1 for local-only)")
    ap.add_argument("-ccs", "--cpu_cache_size", type = float, default = 0.0,
                    help = "CPU second-tier cache size in GB (pages spill from "
                           "GPU when the GPU cache is full)")
    ap.add_argument("--vision", type = str, default = "auto", choices = ["auto", "off"],
                    help = "auto: load the checkpoint's vision tower so image_url "
                           "content parts work (falls back to text-only if it "
                           "does not fit); off: text only")
    ap.add_argument("--image_max_pixels", type = int, default = 1048576,
                    help = "downscale images to at most this many pixels before "
                           "encoding (1 MP ~ 1024 prompt tokens)")
    ap.add_argument("--ui", type = str, default = "on", choices = ["on", "off"],
                    help = "serve a landing page at http://<host>:<port>/ that "
                           "says where the API and the harness are, and forwards "
                           "to the harness once it answers. /v1 is a plain "
                           "OpenAI endpoint either way")
    ap.add_argument("--harness_port", type = int, default = 3080,
                    help = "port the DeepSeek Harness listens on; the landing "
                           "page forwards there (see tools/dsh.py)")
    ap.add_argument("--max_body_mb", type = int, default = 64,
                    help = "max request body size in MiB (aiohttp's built-in "
                           "default is 1 MiB, far too small for a full tool "
                           "set + a long transcript)")
    args = ap.parse_args()
    MODEL_ID = (args.model_id or os.path.basename(os.path.normpath(args.model))).strip().lower() or MODEL_ID
    budgets = parse_split(args.grid_size)
    _cap_process_vram(budgets)
    if len(budgets) > 1 and args.offload_layers != "stock":
        OFFLOAD["layers"] = None if args.offload_layers == "auto" else int(args.offload_layers)
        _install_offload_loader(budgets)
    _draft = args.draft_model.lower()
    use_mtp = _draft == "mtp"
    use_draft = _draft not in ("none", "", "-")
    argv = ["-m", args.model,
            "-gs", ",".join(str(b) for b in budgets), "-cs", str(args.cache_size)]
    draft_kw = {}
    if use_draft and args.num_draft_tokens:
        argv += ["-ndt", str(args.num_draft_tokens)]
        draft_kw["num_draft_tokens"] = args.num_draft_tokens
    if use_draft and args.dynamic_draft:
        draft_kw.update(dynamic_draft_tokens = True, draft_confidence = args.dynamic_draft)
    if use_mtp:
        argv += ["-mtp"]
    elif use_draft:
        argv += ["-dm", args.draft_model]
    if args.cache_quant:
        argv += ["-cq", args.cache_quant]
    if args.cpu_cache_size:
        argv += ["-ccs", str(args.cpu_cache_size)]

    print(f" == loading {args.model}"
          + (" + MTP head" if use_mtp else
             (f" + draft {args.draft_model}" if use_draft else " (no draft)"))
          + " ...", flush = True)
    generator, tokenizer, config = build_model(argv, use_draft = use_draft, draft_kw = draft_kw)
    stats["context_length"] = int(args.cache_size)
    if args.vision == "auto":
        print(" == loading vision tower (images) ...", flush = True)
        load_vision(config, args.image_max_pixels, device = len(budgets) - 1)
    else:
        vision["reason"] = "disabled (--vision off)"
    try:
        import torch
        for i, cap in enumerate(budgets):
            a = torch.cuda.memory_allocated(i) / 1024 ** 3
            r = torch.cuda.memory_reserved(i) / 1024 ** 3
            print(f" == cuda:{i} allocated {a:.2f} GiB, reserved {r:.2f} GiB "
                  f"(cap {cap} GB)", flush = True)
    except Exception as e:
        print(f" == cuda memory stats unavailable: {e}", flush = True)
    # Build and mount first, bind the port next, and only then say Ready: a
    # box that appears before any of that can promise an address that never
    # answers, which is the single most confusing way for a launch to fail.
    app = web.Application(client_max_size = args.max_body_mb * 1024 * 1024)
    app["generator"] = generator
    app["tokenizer"] = tokenizer
    app["max_body_mb"] = args.max_body_mb
    app.router.add_get("/v1/models", models)
    app.router.add_get("/v1/models/{model_id}", model_retrieve)
    app.router.add_get("/health", health)
    app.router.add_post("/v1/chat/completions", chat_completions)
    app.router.add_post("/v1/messages", anthropic_messages)
    app.router.add_post("/v1/messages/count_tokens", anthropic_count_tokens)
    landing_on = mount_landing(app, args) if args.ui == "on" else False

    def ready_box():
        inner = 52
        print(flush = True)
        print("  +" + "-" * inner + "+")
        for line in (
            "  Ready",
            f"  OpenAI:    http://127.0.0.1:{args.port}/v1",
            f"  Anthropic: http://127.0.0.1:{args.port}",
            f"  model: {MODEL_ID}"[:inner],
            "  API key: local (ignored)",
            ("  Images: ON  (max %.1f MP per image)" % (vision["max_pixels"] / 1e6))
            if vision["model"] is not None else "  Images: off (text only)",
            (f"  Chat: http://127.0.0.1:{args.harness_port}/  (harness)"[:inner]
             if landing_on else "  Chat: any OpenAI client"),
            "  Ctrl+C to stop",
        ):
            print("  |" + line.ljust(inner) + "|")
        print("  +" + "-" * inner + "+")
        print(flush = True)

    async def serve():
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, args.host, args.port)
        try:
            await site.start()          # the port is actually bound here
        except OSError as e:
            await runner.cleanup()
            raise SystemExit(
                f"\n  Could not listen on {args.host}:{args.port} - {e}.\n"
                f"  Another copy of the server is probably already running.\n"
                f"  Close it, or set a different PORT in .env.\n") from e
        ready_box()
        try:
            while True:
                await asyncio.sleep(0.5)   # short ticks so Ctrl+C is noticed on Windows
        finally:
            await runner.cleanup()

    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        print("\n  Stopped.", flush = True)


if __name__ == "__main__":
    main()
