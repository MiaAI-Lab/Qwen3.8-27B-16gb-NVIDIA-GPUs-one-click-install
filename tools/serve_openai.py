#!/usr/bin/env python3
"""
Minimal OpenAI-compatible server for the EXL3 serving target.

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
  GET  /                      built-in chat UI (--ui off to disable)
  GET  /v1/models
  GET  /health
  POST /v1/chat/completions   (stream and non-stream, tool calling)

`stream_options: {"include_usage": true}` adds a final chunk carrying the
token counts, which is how the built-in UI reports tokens/second.

After each request the server prints one ` == stats ...` line: prefill/decode
tok/s from the engine's per-stage timings (which exclude the queue), wall
time (which includes it), and speculative-draft acceptance; the same numbers
are mirrored in GET /health as "last_request".

While a request runs it also prints transient ` .. ` progress lines - prefill
ingress, decode tok/s - at most one per PROGRESS_EVERY seconds per phase (.env
knob, default 1 s, 0 = off). They are for tail -f, not the record: the
` == stats` line stays authoritative, and phases shorter than the interval
print no live line at all.

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
import argparse, asyncio, json, os, re, sys, time, threading, uuid
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
    # Snapshot of the last finished request (see _report_request): what the
    # per-request " == stats" line said, for GET /health. Additive - the
    # cumulative counters above are what sparkDash consumes and stay as-is.
    "last_request": None,
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

def _rate(n, t):
    """n / t rounded to 0.1, or None where the rate is undefined. A completion
    of 0 tokens and a prefill served whole out of the free cache
    (time_prefill ~ 0) both divide by zero; a '-' in the stats line beats a
    0.0 or inf that reads like a measurement."""
    return round(n / t, 1) if n and t and t > 1e-6 else None


def _req_tag(rid, streaming):
    """The request's name in log lines: the caller-minted id (or "request")
    plus its mode. Shared by the live progress lines and the stats summary so
    both spell the same request identically; "retry"/"cancelled" suffixes are
    summary-only, since neither is known while the request is still running."""
    return (rid or "request") + (" stream" if streaming else "")


def _report_request(rid, streaming, prompt_toks, out_toks, attempt, retried,
                    cancelled):
    """One ' == stats' line per finished request, plus the same numbers as
    stats["last_request"] for GET /health.

    Rates come from the engine's EOS result dict, which measures stages on
    the GPU side: time_prefill = first prefill -> first token, time_generate
    = first -> last token, time_enqueued = the in-engine queue. None of them
    include the wait for gen_lock (requests are serialized), which is why
    `wall` - measured around the whole iterate loop, lock wait included - is
    the only number that grows under concurrency. A cancelled job never
    receives the EOS dict (generator.cancel drops it), so after a cancel the
    per-stage rates print '-' and only `wall` is real.

    cached_tokens is clamped to the prompt length: a requeued long generation
    resubmits prompt + generated-so-far as one new prompt and all of it hits
    the cache, which would otherwise claim more cached tokens than the
    request had prompt tokens. prompt_tokens from the same dict is skipped
    for the same reason (it counts the requeued input, not the caller's)."""
    eos = attempt.get("eos") or {}
    wall = attempt.get("wall") or 0.0
    cached = min(int(eos.get("cached_tokens") or 0), int(prompt_toks))
    gen = int(eos.get("new_tokens") or 0) or int(out_toks)
    prefill_tps = _rate(prompt_toks - cached, eos.get("time_prefill"))
    decode_tps = _rate(gen, eos.get("time_generate"))
    queued = eos.get("time_enqueued") or 0.0
    # absent entirely when no drafter is attached (-dm none)
    acc = int(eos.get("accepted_draft_tokens") or 0)
    rej = int(eos.get("rejected_draft_tokens") or 0)
    drafted = acc + rej

    tag = (_req_tag(rid, streaming)
           + (" retry" if retried else "")
           + (" cancelled" if cancelled else ""))
    parts = [f"prompt {prompt_toks} tok" + (f" ({cached} cached)" if cached else ""),
             f"completion {out_toks} tok",
             "prefill " + (f"{prefill_tps} tok/s" if prefill_tps is not None
                           else "- (fully cached)" if cached else "-"),
             "decode " + (f"{decode_tps} tok/s" if decode_tps is not None else "-"),
             f"wall {wall:.2f} s (queue {queued:.2f})"]
    if drafted:
        parts.append(f"draft accepted {acc}/{drafted} ({round(100.0 * acc / drafted)}%)")
    print(f" == stats {tag}: " + ", ".join(parts), flush = True)

    with stats_lock:
        stats["last_request"] = {
            "id": rid, "streaming": bool(streaming), "retried": bool(retried),
            "cancelled": bool(cancelled), "ts": int(time.time()),
            "prompt_tokens": int(prompt_toks), "cached_prompt_tokens": cached,
            "completion_tokens": int(out_toks),
            "prefill_tok_s": prefill_tps, "decode_tok_s": decode_tps,
            "queue_s": round(queued, 3), "wall_s": round(wall, 3),
            "draft_accepted": acc, "draft_rejected": rej,
        }


PROGRESS_EVERY = 1.0    # seconds between live progress lines within a phase


class _LiveProgress:
    """Transient ` .. ` lines while ONE generation runs, so tail -f on the log
    shows where a request is. Additive to the ` == stats` summary, which stays
    the record: phases shorter than PROGRESS_EVERY print nothing (short chats
    stay quiet), rates are phase averages over the whole phase (clock started
    when the generator lock is acquired - the same window the summary's
    engine-side timings use), and lines are plain one-shot prints - no CR
    redraws, because tools/logbook.py strips redrawn lines from the file copy,
    which is where these matter. Nothing polls: the engine's iterate() events
    are the only wakeups, each carrying a timestamp check that mostly says
    nothing.

    curr_progress counts free prompt-cache page hits as ingested, so a live
    prefill rate can read high on a warm cache; the summary's cached_tokens is
    the corrected number. One instance per run_once() attempt: a greedy
    tool_choice retry is a real second generation and prints its own lines.
    No cancel line is needed - the summary already says "cancelled"."""

    def __init__(self, tag, prompt_toks):
        self.tag = tag
        self.prompt_toks = prompt_toks   # authoritative total (not max_progress)
        self.phase = None                # "prefill" | "decode"; set by queued()
        self.t_phase = 0.0               # start of the current phase
        self.last = 0.0                  # last live line (throttle)
        self.curr = 0                    # latest curr_progress (for "prefill done")
        self.toks = 0                    # decode tokens since phase entry
        self.printed_prefill = False

    def _say(self, msg):
        print(f" .. {self.tag}: {msg}", flush = True)
        self.last = time.time()

    def queued(self, wait):
        """Called the moment gen_lock is finally held. The generation's clock
        starts here either way: prefill is the next thing the engine does once
        the lock is ours, so t_phase set now covers the WHOLE phase - matching
        the summary's engine-side time_prefill - rather than starting at the
        first event we happen to see, which would exclude the first, largest
        chunk and report a duration the summary contradicts. The print itself
        (the previous owner can be another request or image embeddings) waits
        for 0.5 s so an idle box stays quiet."""
        now = time.time()
        self.phase, self.t_phase, self.last = "prefill", now, now
        if PROGRESS_EVERY <= 0 or wait < 0.5:
            return
        self._say(f"queued {wait:.1f} s for the generator")

    def prefill(self, curr):
        if PROGRESS_EVERY <= 0:
            return
        self.curr = curr
        now = time.time()
        if now - self.last >= PROGRESS_EVERY:
            tps = _rate(curr, now - self.t_phase)
            self._say(f"prefill {curr}/{self.prompt_toks} tok"
                      + (f" ({tps} tok/s)" if tps is not None else ""))
            self.printed_prefill = True

    def decode(self, n):
        if PROGRESS_EVERY <= 0:
            return
        now = time.time()
        if self.phase != "decode":
            # First decode token = prefill is over. Only a prefill long enough
            # to have printed gets a closing line; a short or fully cached one
            # said nothing and must not start now.
            if self.phase == "prefill" and self.printed_prefill:
                self._say(f"prefill done, {self.curr} tok "
                          f"in {now - self.t_phase:.1f} s")
            self.phase, self.t_phase, self.last = "decode", now, now
            self.toks = n
            return
        self.toks += n
        if now - self.last >= PROGRESS_EVERY:
            tps = _rate(self.toks, now - self.t_phase)
            self._say(f"decoding, {self.toks} tok"
                      + (f" ({tps} tok/s)" if tps is not None else ""))


TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
HOLD_BACK = 16                       # marker-safe holdback for streamed text


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


def _cap_process_vram(gb):
    """Hard-cap this process to `gb` GiB so a large unified-memory box
    behaves like a discrete card with that much free VRAM. ExLlama lifts
    the CUDA fraction after autosplit; pin it back to the same cap."""
    import torch
    from exllamav3.util import memory as _mem
    torch.cuda.init()
    total = torch.cuda.get_device_properties(0).total_memory
    cap_bytes = int(float(gb) * 1024 ** 3)
    frac = min(max(cap_bytes / total, 0.01), 1.0)

    def _pin(devices=None):
        for i in (devices if devices is not None else [0]):
            torch.cuda.set_per_process_memory_fraction(frac, device = i)

    _pin()
    _mem.set_memory_fraction_use = lambda use, device: _pin([device])
    _mem.set_memory_fraction_reserve = lambda reserve, device: _pin([device])
    _mem.unset_memory_fraction = lambda active: _pin(active)
    print(f" == VRAM cap: {gb} GB "
          f"({cap_bytes / 1024**3:.2f} GiB, fraction {frac:.4f} of "
          f"{total / 1024**3:.1f} GB device)", flush = True)


def build_model(argv, use_draft = True):
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
            # match model_init's max_history sizing, which reads the same caps.
        )
    else:
        model, config, cache, tokenizer = model_init.init(args, progress = True)
        generator = Generator(model, cache, tokenizer)
    return generator, tokenizer, config


def load_vision(config, max_pixels):
    """Load the checkpoint's vision tower (Qwen3.8: 27 layers, 3-bit, ~0.3 GB)
    after the text model. Never fatal: on failure the server stays text-only."""
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
        vm.load(progressbar = True)
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


def generate_full(generator, tokenizer, messages, max_tokens, temperature,
                  top_p, top_k, seed, tools, tool_choice = None, stop = None,
                  on_text = None, enable_thinking = True, should_stop = None,
                  reasoning_effort = None, rid = None, streaming = False):
    """Blocking generation; returns (text, tool_calls, finish, p_toks, o_toks,
    reasoning, content). rid/streaming only label the per-request stats line
    (see _report_request); defaulted so existing callers are unaffected."""
    schemas = build_tool_schemas(tools)
    tools, directive = tool_choice_directive(tool_choice, tools)
    if directive:
        messages = list(messages)
        if messages and messages[0].get("role") == "system":
            # Qwen template allows only ONE leading system message — merge
            first = dict(messages[0])
            c = first.get("content") or ""
            if isinstance(c, list):      # content parts (multimodal-style clients)
                first["content"] = list(c) + [{"type": "text", "text": "\n\n" + directive}]
            else:
                first["content"] = c.rstrip() + "\n\n" + directive
            messages[0] = first
        else:
            messages = [{"role": "system", "content": directive}] + messages
    messages, image_urls = extract_images(messages)
    embeddings = None
    if image_urls:
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
        input_ids = tokenizer.encode(rendered, encode_special_tokens = True,
                                     embeddings = embeddings)
    else:
        input_ids = tokenizer.hf_chat_template(
            messages, add_generation_prompt = True,
            enable_thinking = enable_thinking, tools = tools,
            **template_effort(tokenizer, reasoning_effort))
    prompt_toks = int(input_ids.shape[-1])
    from exllamav3.generator.sampler.presets import ComboSampler
    from exllamav3 import Job
    forced_choice = tool_choice not in (None, "auto", "none")
    reason = "max_new_tokens"
    text = ""
    attempt = {}   # metrics of the run that produced the reply: {"eos": {...}, "wall": s}

    def run_once():
        nonlocal text, reason, attempt
        text = ""
        reason = "max_new_tokens"
        sampler = ComboSampler(temperature = temperature, top_k = top_k, top_p = top_p)
        stop_conditions = ["<|im_end|>", tokenizer.eos_token_id] + (stop or [])
        job = Job(input_ids = input_ids, max_new_tokens = max_tokens,
                  stop_conditions = stop_conditions,
                  sampler = sampler, seed = seed,
                  embeddings = embeddings)
        prefill_seen = 0
        eos = {}      # engine per-stage timings, filled from the final EOS result
        live = _LiveProgress(_req_tag(rid, streaming), prompt_toks)
        t0 = time.time()   # before the lock: wall must include waiting for it
        with gen_lock:
            live.queued(time.time() - t0)
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
                        live.prefill(curr)
                        if curr > prefill_seen:
                            _bump_stats(prompt=curr - prefill_seen)
                            prefill_seen = curr
                    elif _result_new_tokens(r):
                        n = _result_new_tokens(r)
                        _bump_stats(completion=n)
                        live.decode(n)
                    chunk = r.get("text", "")
                    if chunk:
                        text += chunk
                        if on_text is not None:
                            on_text(chunk)
                    if r.get("eos"):
                        reason = r.get("eos_reason", reason)
                        # The EOS dict is the only place the engine reports
                        # per-stage timings, and it is only sent on a natural
                        # finish: cancel() drops it, hence the empty-dict
                        # fallback path in _report_request.
                        eos = {
                            "time_prefill": float(r.get("time_prefill") or 0.0),
                            "time_generate": float(r.get("time_generate") or 0.0),
                            "time_enqueued": float(r.get("time_enqueued") or 0.0),
                            "new_tokens": int(r.get("new_tokens") or 0),
                            "cached_tokens": int(r.get("cached_tokens") or 0),
                            "accepted_draft_tokens": int(r.get("accepted_draft_tokens") or 0),
                            "rejected_draft_tokens": int(r.get("rejected_draft_tokens") or 0),
                        }
            if reason != "cancelled" and prefill_seen < prompt_toks:
                _bump_stats(prompt=prompt_toks - prefill_seen)
        attempt = {"eos": eos, "wall": time.time() - t0}
        return job

    job = run_once()
    retried = False
    # Forced tool_choice is a prompt nudge; at temperature > 0 the model can
    # occasionally skip the call. One greedy retry makes it deterministic -
    # but not after a cancel, or Stop would start a second generation.
    if reason != "cancelled" and forced_choice and not parse_tool_calls(text, schemas)[1]:
        temperature = 0.0
        job = run_once()
        retried = True
    seq = job.sequences[0]
    out_toks = int(seq.sequence_ids.seq_len - prompt_toks)
    # One stats line per request, for the run that produced the returned
    # reply: a greedy retry's discarded first attempt is summarized by the
    # "retry" tag rather than a second line, because the caller only ever
    # receives one reply.
    _report_request(rid, streaming, prompt_toks, out_toks, attempt, retried,
                    reason == "cancelled")
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
            "last_request": stats.get("last_request"),
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

    import asyncio
    if not req["stream"]:
        # Minted before generation so the per-request stats line inside
        # generate_full can name the request; the response reuses it.
        cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        try:
            text, calls, finish, ptoks, otoks, reasoning, content = await asyncio.to_thread(
                generate_full, generator, tokenizer, req["messages"],
                req["max_tokens"], req["temperature"], req["top_p"], req["top_k"],
                req["seed"], req["tools"], req["tool_choice"], req["stop"],
                None, req["enable_thinking"], None,
                req["reasoning_effort"], rid = cid)
        except AssertionError as e:
            return web.json_response(
                {"error": {"message": f"context/cache: {e}", "type": "invalid_request_error"}},
                status = 400)
        except ValueError as e:
            return web.json_response(
                {"error": {"message": str(e), "type": "invalid_request_error"}},
                status = 400)
        msg = {"role": "assistant", "content": content or None}
        if reasoning:
            msg["reasoning_content"] = reasoning
        if calls:
            msg["tool_calls"] = calls
        return web.json_response({
            "id": cid,
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
                text, calls, finish, ptoks, otoks, reasoning, content = generate_full(
                    generator, tokenizer, req["messages"], req["max_tokens"],
                    req["temperature"], req["top_p"], req["top_k"],
                    req["seed"], req["tools"], req["tool_choice"], req["stop"],
                    on_text = None if forced_choice else on_text,
                    enable_thinking = req["enable_thinking"],
                    reasoning_effort = req["reasoning_effort"],
                    should_stop = gone.is_set,
                    rid = cid, streaming = True)
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    ("done", (calls, finish, reasoning, content, ptoks, otoks)))
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

        pending, finish, calls_emitted = "", None, False
        call_idx = [0]
        # With thinking on, the template ends the prompt with "<think>" and the
        # generation therefore starts inside it. With thinking off it emits an
        # empty "<think></think>" pair instead, so the first token is already
        # the answer - starting in_think True there would swallow the reply into
        # a reasoning block nobody asked for.
        in_think = [bool(req["enable_thinking"])]
        THINK_CLOSE = "</think>"

        async def send_call(c):
            nonlocal calls_emitted
            calls_emitted = True
            await send({"tool_calls": [dict(c, index = call_idx[0])]})
            call_idx[0] += 1

        async def flush_pending(final = False):
            """Emit everything parseable from pending; keep marker-safe tail."""
            nonlocal pending
            while True:
                if in_think[0]:
                    close = pending.find(THINK_CLOSE)
                    if close >= 0:
                        head, pending = pending[:close], pending[close + len(THINK_CLOSE):]
                        if head.strip():
                            await send({"reasoning_content": head.lstrip("\n")})
                        in_think[0] = False
                        continue
                    cut = len(pending) if final else max(0, len(pending) - HOLD_BACK)
                    piece = pending[:cut]
                    if piece.strip():
                        await send({"reasoning_content": piece})
                    pending = pending[cut:]
                    return
                if TOOL_CALL_OPEN in pending:
                    head, rest = pending.split(TOOL_CALL_OPEN, 1)
                    if head.strip() or (final and head):
                        await send({"content": head})
                    if TOOL_CALL_CLOSE in rest:
                        block, pending = rest.split(TOOL_CALL_CLOSE, 1)
                        _, calls = parse_tool_calls(
                            TOOL_CALL_OPEN + block + TOOL_CALL_CLOSE,
                            req_schemas)
                        for c in calls:
                            await send_call(c)
                        continue
                    # unterminated call: final -> implicit close, else hold
                    if final and "<function=" in rest:
                        _, calls = parse_tool_calls(TOOL_CALL_OPEN + rest,
                                                    req_schemas)
                        for c in calls:
                            await send_call(c)
                        pending = ""
                    else:
                        pending = TOOL_CALL_OPEN + rest
                    return
                cut = len(pending) if final else max(0, len(pending) - HOLD_BACK)
                await send({"content": pending[:cut]})
                pending = pending[cut:]
                return

        async def watch_client():
            """A reply that is still in prefill writes nothing, so a failed
            write would never notice the browser had gone. Watch the socket."""
            try:
                while not gone.is_set():
                    transport = request.transport
                    if transport is None or transport.is_closing():
                        gone.set()
                        return
                    await asyncio.sleep(0.2)
            except asyncio.CancelledError:
                pass

        async def consume():
            nonlocal pending            # flush_pending owns it too
            while True:
                kind, payload = await queue.get()
                if kind == "error":
                    await resp.write(
                        f'data: {json.dumps({"error": {"message": payload}})}\n\n'.encode())
                    break
                if kind == "delta":
                    pending += payload
                    await flush_pending()
                elif kind == "done":
                    calls, finish, reasoning, content, ptoks, otoks = payload
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

        watcher = asyncio.ensure_future(watch_client())
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
<p>The model is loaded and serving the OpenAI API at
   <code>{api}</code> &mdash; API key <code>local</code>.</p>
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
        harness = f"http://127.0.0.1:{args.harness_port}/"
        body = LANDING.format(model = MODEL_ID, api = api,
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
    global MODEL_DIR, DRAFT_DIR, PORT, MODEL_ID, PROGRESS_EVERY
    _quiet_triton()
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default = MODEL_DIR)
    ap.add_argument("-dm", "--draft_model", default = DRAFT_DIR,
                    help = "'mtp' for MTP drafting (head inside the "
                           "main checkpoint: no extra weights, much smaller KV footprint) "
                           "or 'none' to disable drafting")
    ap.add_argument("-gs", "--grid_size", type = float, default = 14.7,
                    help = "GPU memory budget in GB (autosplit + process cap). "
                           "14.7 is the 16 GB-card recipe")
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
    ap.add_argument("--progress_every", type = float, default = 1.0,
                    help = "seconds between live ' .. ' progress lines per "
                           "phase while a request runs; 0 disables them (the "
                           "end-of-request ' == stats' line always prints)")
    args = ap.parse_args()
    MODEL_ID = (args.model_id or os.path.basename(os.path.normpath(args.model))).strip().lower() or MODEL_ID
    PROGRESS_EVERY = args.progress_every
    _cap_process_vram(args.grid_size)
    _draft = args.draft_model.lower()
    use_mtp = _draft == "mtp"
    use_draft = _draft not in ("none", "", "-")
    argv = ["-m", args.model,
            "-gs", str(args.grid_size), "-cs", str(args.cache_size)]
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
    generator, tokenizer, config = build_model(argv, use_draft = use_draft)
    stats["context_length"] = int(args.cache_size)
    if args.vision == "auto":
        print(" == loading vision tower (images) ...", flush = True)
        load_vision(config, args.image_max_pixels)
    else:
        vision["reason"] = "disabled (--vision off)"
    try:
        import torch
        a = torch.cuda.memory_allocated(0) / 1024 ** 3
        r = torch.cuda.memory_reserved(0) / 1024 ** 3
        print(f" == cuda allocated {a:.2f} GiB, reserved {r:.2f} GiB "
              f"(cap {args.grid_size} GB)", flush = True)
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
    app.router.add_get("/health", health)
    app.router.add_post("/v1/chat/completions", chat_completions)
    landing_on = mount_landing(app, args) if args.ui == "on" else False

    def ready_box():
        inner = 52
        print(flush = True)
        print("  +" + "-" * inner + "+")
        for line in (
            "  Ready",
            f"  http://127.0.0.1:{args.port}/v1",
            f"  model: {MODEL_ID}"[:inner],
            "  OpenAI-compatible  |  API key: local",
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
