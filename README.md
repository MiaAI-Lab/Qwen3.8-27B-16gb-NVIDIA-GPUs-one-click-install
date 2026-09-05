---
library_name: transformers
license: apache-2.0
pipeline_tag: image-text-to-text
tags:
  - exl3
  - quantized
  - qwen3.8
---

# Qwen3.8-27B-EXL3 2.0bpw — 16 GB NVIDIA recipe

EXL3 2.0 bpw quant of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), **sized to load and run on NVIDIA GPUs with 16 GB VRAM** (RTX 4060 Ti 16 GB, RTX 5060 Ti 16 GB, Tesla T4, and similar).

Quantization is **[turboderp](https://huggingface.co/turboderp)**'s `SC_2.00bpw_H3_V3` (original: [turboderp/Qwen3.8-27B-exl3](https://huggingface.co/turboderp/Qwen3.8-27B-exl3)). This repo adds a serving kit and the context/VRAM settings that actually fit a 16 GB card.

A 16 GB board typically has about **14.7 GB free** after the driver. That is the budget this recipe uses. Native 262144-token context does **not** fit.

## Validated 16 GB settings

| Knob | Value | Why |
| --- | --- | --- |
| `GPU_MEM_GB` | `14.7` | Process cap matching ~14.7 GB free on a 16 GB card |
| `CONTEXT_SIZE` | `199936` | ~200k tokens (must be a multiple of the 256-token page size) |
| `CACHE_QUANT` | `8,4` | int8 K / int4 V |
| `DRAFT` | `mtp` (default) | MTP head inside the checkpoint; ~50 MB extra weights |

Measured at load under that cap (MTP on, `8,4` KV):

* CUDA **allocated 12.35 GiB**, **reserved 13.37 GiB**
* Native **`CONTEXT_SIZE=262144` fails to boot** (`Insufficient VRAM in split for model and cache`)

Weights on disk are ~9.7 GB. The rest is KV (16 full-attention layers), the MTP draft cache, GDN recurrent state, and CUDA workspace.

## Quick start

**Windows 11:** double-click `start.bat`. A page opens in your browser and does the rest: it shows what it found on the card, offers the model sizes that fit it, then installs and downloads with a progress bar and a live log. Nothing is asked in the console. When it finishes, the same page turns into the chat UI at `http://127.0.0.1:8888/`.

The only thing you have to install first is **64-bit Python 3.11+** — and `start.bat` says so plainly if it is missing. Everything else Simplex installs into its own `.venv` folder. If a prebuilt engine wheel is available (in `wheels\`, or at a `WHEEL_INDEX` you set in `.env`) the first run is a plain download; without one it compiles the engine, which additionally needs Git, the NVIDIA CUDA Toolkit and Visual Studio Build Tools with the C++ workload. See [Prebuilt wheels](#prebuilt-wheels-no-compiler-needed).

The download is resumable: closing the window, losing the connection or rebooting costs you nothing, because it picks up from the byte it stopped at.

Later launches skip setup. Right before the model loads, `start.bat` checks that about **14.7 GB of VRAM is actually free** (the whole recipe needs it) and, if not, lists the programs holding VRAM — browsers, games, Discord, other AI tools — and waits for you to close them (Enter re-checks, `c` continues anyway, `q` quits). Take that seriously on Windows: with too little free VRAM the driver pages the model into system RAM instead of failing, and it then runs many times slower.

While it runs, Simplex puts an icon in the notification area — right-click for **Open Simplex**, **Restart the model**, **Show the Simplex folder**, **View the log** and **Quit**. Every launch also writes a full transcript to `logs\`, so a crash that scrolls past is still readable afterwards. `.venv`, `models/`, `logs/` and `apps/` stay on your machine and are not part of the git tree.

To go back to the old console questions instead of the web page, set `SETUP=console` in `.env`.

**Linux:**

```bash
cp .env.example .env    # already set for 16 GB
./start.sh              # first run: venv + ExLlamaV3 v1.4.4, then serve
# http://localhost:8888/v1
```

Requires **ExLlamaV3 v1.4.4** (quantized vision tower). PyPI skips 1.4.4 (`1.4.2` → `1.4.5`); the launchers install the git tag. Engine: [ExLlamaV3](https://github.com/turboderp-org/exllamav3).

## Profiles: the launcher picks quants for your GPU

The first start (or `start.bat profile` / `./start.sh profile`, or `PROFILE=ask` in `.env`) runs `tools/profiles.py`: it reads the card's VRAM with `nvidia-smi`, computes what fits under a budget of *VRAM − max(1.3 GB, 8 %)* using the measured weight sizes, KV cost per token (`8,4` 26 KB, `4` 18 KB, MTP draft cache +1/16), the vision tower (measured: 0.87 GB, or 0.17 GB for the 3-bit-quantised 2.0 bpw one) and 2.6 GB of runtime overhead (measured against real prefill peaks; it was 1.7, which under-predicted every quant by more than the safety margin covered), and offers the best-quality and longest-context options (plus a middle one when they are far apart). Enter takes the recommendation; if a model is already downloaded, "keep current" is the default so an unattended start never triggers a surprise download. The choice is written into `.env` (`MODEL_DIR`, `HF_TARGET_REPO`, `HF_REVISION`, `MODEL_ID`, `CONTEXT_SIZE`, `CACHE_QUANT`, `GPU_MEM_GB`, `VISION`) and everything downstream — server, Cherry's model entry and defaults — follows it.

| VRAM | offered (KV cache is always the stock int4 — measured within 0.001 KL of fp16, and it has no hardware requirement: it runs on every supported GPU. Only the fp8 / nvfp4 lanes need Ada or newer) |
| --- | --- |
| 12 GB | 2.0 bpw @ 33k, text-only — the floor, and the whole menu |
| 16 GB | **3.0 bpw @ 66k** · 2.5 bpw @ 180k · 2.0 bpw @ 229k with images |
| 24 GB | **5.0 bpw @ 147k** · 4.0 bpw @ 262k with images · everything below it at 262k |
| 32 GB+ | **5.0 bpw @ 180k** with images · 6.0 bpw @ 262k once its prefill is verified |

Bold is what the launcher pre-selects: the best quality that still has real context
(≥ 64k), not the longest context — on a 16 GB card only the two weakest quants can
reach 128k, so optimising for context there means shipping three times the
quantisation error to buy room almost nobody uses. The setup page's **Simulation
mode** shows this table for any specific card without owning one.

Quants other than the 2.0 bpw baseline are pulled from turboderp's branches (`HF_REVISION`); their vision towers are unquantised (0.87 GB measured), which is why images are off on the tight profiles. The KL figures behind the menu's quality words are turboderp's (mean KL vs bf16): 2.0 → 0.35 *fair*, 2.5 → 0.30 *good*, 3.0 → 0.11 *better*, 3.5 → 0.08 *very good*, 4.0 → 0.05 *very good*, 5.0 → 0.014 *excellent*, 6.0 → 0.007 *near-lossless*. `python tools/profiles.py --list --vram 24` previews the menu for any card.

## Chat with the model

The server is **OpenAI-compatible**. Leave `start.bat` / `start.sh` running, then point a client at it. There is **no API key**; many apps still require a dummy value such as `local`.

| | |
| --- | --- |
| Base URL | `http://127.0.0.1:8888/v1` |
| API key | `local` (ignored) |
| Model | `qwen3.8-27b-exl3-2.0bpw` |

### The built-in chat UI

The server serves its own chat app at **`http://127.0.0.1:8888/`** — the same
process, no extra download, nothing to configure. `start.bat` opens it once the
**Ready** box appears (`UI=browser|server|no` in `.env`), and `/v1` stays a
plain OpenAI endpoint for every other client.

The UI answers **only this computer** by default, even with `HOST=0.0.0.0`:
`/v1` serving your network is one thing, but the UI's Agent mode writes files
and runs commands on this PC and there is no login. `UI_LAN=1` in `.env` opens
it to the network when you want it on your phone.

It streams, folds the model's thinking into a collapsed block, renders Markdown
and code (with copy buttons), takes pasted or dropped images (when the vision
tower is loaded), keeps conversations in `sessions/` as JSON, follows your
system's dark/light theme with a toggle to override it, works on a phone, and
has two modes:

| Mode | Tools | What it is for |
| --- | --- | --- |
| **Chat** (default) | `web_search`, `web_fetch` | Everyday questions. Read-only: no filesystem, no shell. |
| **Agent** | the web tools plus `list_dir`, `read_file`, `find_files`, `search_text`, `write_file`, `edit_file`, `run_python`, `run_command`, `job_output`, `job_kill`, `update_plan`, `ask_user` | Work on files in **one folder you pick**. Everything outside that folder is refused; writes and commands ask you in the browser before they run (Allow once / Always allow / Deny). |

Several habits in the agent are borrowed from
[DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) (MIT), which
has already learned them the hard way:

* **Read before you mutate.** An unseen file can be created but never silently
  replaced, and `edit_file` refuses outright until the file has been read —
  with the remedy in the error, so the model fixes itself instead of flailing.
* **A visible plan.** `update_plan` takes the whole checklist every call (no
  partial updates), keeps exactly one step `in_progress`, and the browser shows
  it as a progress strip under the top bar that survives a reload.
* **A real question.** `ask_user` puts a question with labelled options in
  front of you and waits — for decisions that are actually yours, not for
  permission to run a tool.
* **Honest commands.** Each call is a fresh shell (pass `workdir`, not `cd`),
  the exit code is reported as `[exit code: N]`, long output keeps its *tail*
  and spills the rest to a file the model can read back, and anything slow goes
  in the background with `run_in_background` and is polled with `job_output`.
* **External text is data.** Search results and fetched pages arrive labelled
  as untrusted content, so a web page cannot quietly issue instructions.

Switching mode starts a fresh chat, so a transcript never mixes toolsets. The
agent's folder is `AGENT_WORKSPACE` in `.env` (default `workspace/`); the chip
in the top bar opens **the normal Windows folder dialog** to change it
(`FOLDER_PICKER=browser` forces the kit's own browser instead, which is also
what other devices on the network get — a system dialog would open on this
PC's screen, not theirs).

**Work outlives the window.** A turn belongs to the conversation, not to the
browser tab that started it. Close the tab, reload, or walk to another device
mid-answer and the model keeps generating, tools keep running and the answer is
still saved. Reopening replays that turn from the beginning and carries on live;
conversations that are still working show a pulsing dot in the sidebar, and
reloading the page picks one up automatically. **Stop** is the only thing that
ends a turn early. One turn per conversation at a time — sending into a busy
conversation attaches to what is already running instead of starting a second.

**Living with it.** Hovering an answer offers **Copy** and **Retry**; the last
question you asked offers **Edit**, which puts it back in the composer and
forgets the answer it produced. Dropping or pasting a **text file** (code,
Markdown, CSV, JSON, logs) pastes its contents into the prompt — images still
go through the vision tower. **Right-click any conversation** for Rename, Pin to top, Save as Markdown and
Delete (the same menu is on the button at the end of the row). Pinned
conversations sit in their own group above the day headings. The **search box**
searches names *and* what was actually said — matches in the messages come back
with the surrounding sentence, and the index is cached per file so typing does
not re-read the whole folder. A **context meter** under the model shows how much of the window the
current conversation is using, and warns before it runs out. When an answer
finishes while the tab is in the background the title flags it, and Settings
can turn on a desktop notification as well.

**Other endpoints.** The kit's own server is the default, but the UI can talk
to anything else that speaks the OpenAI API — another runtime on this machine
(llama.cpp, Ollama, LM Studio, vLLM), a box on the network, or a hosted service.
**Settings → Providers and endpoints** takes a name, a base URL, an optional API
key and a model id; **Test and list models** asks the endpoint what it serves
and fills the list. Provider models then appear in the Model picker alongside
the local quants and switch **instantly** — nothing is loaded into VRAM — while
tools still run on this computer and approvals still apply. Keys are stored in
`providers.json` beside `.env` (gitignored) and are never sent back to the
browser: the UI shows `sk-...4f2a`, and an empty key box means "keep the saved
one". Each conversation remembers which endpoint answered it.

**Reaching it from outside (Tailscale).** The UI answers only this computer by
default. To use it from a phone or a laptop elsewhere, keep the server on
loopback and let Tailscale do the exposure and the identity:

```
# .env
HOST=127.0.0.1
UI_HOSTS=yourbox.tailXXXX.ts.net        # tailscale status prints the name
```

then, on this PC:

```
tailscale serve --bg 8888
```

Open `https://yourbox.tailXXXX.ts.net` from any device in your tailnet.
Tailscale terminates TLS and only your devices can connect; the port is never
exposed on the local network, and because the proxy connects from loopback the
"only this computer" rule still holds — `UI_HOSTS` only says which *name* is
allowed through it.

The blunter alternative binds the tailnet interface directly and drops the
same-machine rule: `HOST=100.x.y.z` with `UI_LAN=1`. Every device in the
tailnet can then use it, Agent mode included.

Two warnings. Do **not** put this behind `tailscale funnel` — that publishes it
to the internet, and Agent mode runs commands on this PC with no login. And
note the shipped `HOST=0.0.0.0` already exposes `/v1` (the API, not the UI) to
whatever network you are on; set `HOST=127.0.0.1` if that network is not yours.

**Switching model.** Every quant you have downloaded under `models/` is listed
under **Model** in the sidebar. Picking one writes the matching settings into
`.env` — including a context size re-planned for that quant on your card, using
the same planner as the first-run profile menu — and restarts the server into
it, which takes about a minute; conversations are kept. A quant too large for
the card is listed but greyed out with the reason, rather than failing at load.
Only the computer running the server may switch, and only when the server was
started by `start.bat` / `start.sh` (something has to start it again).

**Speed.** The sidebar shows **tok/s**, and the top bar shows a live figure while the model generates
(sampled from `/health`'s counters), and each finished answer keeps its exact
numbers — tokens, tok/s, seconds — on the row under it. The figure counts
generation time only, so tool calls and web fetches do not drag it down. The
server reports token counts to any client that asks with the OpenAI-standard
`stream_options: {"include_usage": true}`.

Web search needs no API key: it uses DuckDuckGo by scraping, which can break
without warning. Point `SEARXNG_URL` at your own SearXNG, or set
`TAVILY_API_KEY`, for a backend that does not. `WEB_TOOLS=0` removes the web
tools; `AGENT_EXEC=0` keeps the agent to reading and editing files.

Work on the UI without loading the model:

```bash
python tools/chatui.py --mock --open      # scripted replies, port 8890
python tools/test_webui.py                # loop, approvals, sandbox checks
```

### Cherry Studio (optional, off by default)

The kit used to ship [Cherry Studio](https://github.com/CherryHQ/cherry-studio) as its chat app; the built-in UI above replaced it. It is still wired up for anyone who wants Cherry's assistants, knowledge bases and MCP servers — set `CHERRY_AUTOSTART=ask` (or `yes`) in `.env` and it works exactly as before:

* `start.bat` downloads the pinned **portable** build (v2.0.10, ~285 MB, once) into `apps/cherry-studio/`. Nothing is installed system-wide; Cherry keeps its data in `apps/cherry-studio/data/`.
* The first time, Cherry opens and closes once by itself to create that data folder. The kit then writes its configuration straight into Cherry's store: provider **Simplex (local)** → `http://127.0.0.1:8888/v1`, key `local`, model `qwen3.8-27b-exl3-2.0bpw` (tool calling + image input on, ~200k context), set as the default chat model and on the default assistant, onboarding skipped. Usage analytics is switched off (Cherry → Settings → Privacy to change).
* After the **Ready** box: *"Open Cherry Studio and start chatting now? [Y/n]"*. Enter/`y` opens it (the portable build unpacks for ~10–20 s), `n` or no answer within 90 s leaves it closed. Keep the server window open while chatting; `stop.bat` stops the server only.
* Changing `PORT` in `.env` re-points the provider on the next start. Open Cherry later without the prompt: `.venv\Scripts\python.exe tools\cherry.py open` (`status` shows what the kit thinks).
* `.env` knobs: `CHERRY_AUTOSTART=ask|yes|no` (`no` also skips the download), `CHERRY_VERSION` (pinned; the store layout is checked against v2.0.x), `CHERRY_EXE=<path>` to use a Cherry Studio you already installed — the kit then only sends Cherry's official import link (`cherrystudio://providers/api-keys`), you confirm the popup and add the model id under the new provider.
* **Web search and tools are on by default.** The default assistant gets `web_search` + `web_fetch` as function tools (Cherry's stock keyless search provider, Exa MCP at `mcp.exa.ai`; change it under Settings → Web Search), runs MCP in *auto* mode, and the kit installs Cherry's keyless builtin MCP servers `@cherry/fetch` and `@cherry/sequentialthinking`. The model decides when to call them; the server turns Qwen's XML tool calls into OpenAI `tool_calls` (below). Tune with `CHERRY_WEB_SEARCH=1|0` and `CHERRY_MCP_SERVERS=` (also `@cherry/python`, `@cherry/browser`; empty = none) in `.env` — re-applied on the next start when you change them. New assistants you create in Cherry start with Cherry's own defaults (web search off) unless you copy the default one. Cherry Studio is [AGPL-3.0](https://github.com/CherryHQ/cherry-studio/blob/main/LICENSE) (see its README for the commercial-use terms); the kit downloads the official release binary and does not redistribute it.

The built-in UI is served on Linux too (`./start.sh`, then open `http://127.0.0.1:8888/`). Manual setup for any other OpenAI client (Chatbox, Open WebUI, Continue, Cursor): add an **OpenAI-compatible** provider with base `http://127.0.0.1:8888/v1` (or host `http://127.0.0.1:8888` if the app appends `/v1` itself), API key `local`, model id `qwen3.8-27b-exl3-2.0bpw`. Any other OpenAI client works the same way (Chatbox, Open WebUI, Continue, Cursor custom endpoint). Open WebUI is stronger if you want a big tools/RAG UI and are fine running Docker.

**Tool calling.** Send OpenAI `tools` (function name + JSON schema) on `POST /v1/chat/completions`. The model emits Qwen XML; the server parses it into `tool_calls`. Your app must run the function and POST a follow-up with `role: "tool"` (and the previous assistant `tool_calls`). `tool_choice` of `auto`, `required`, or a named function is supported. Example:

```bash
curl http://127.0.0.1:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27b-exl3-2.0bpw",
    "messages": [{"role": "user", "content": "What is the weather in Tel Aviv?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Current weather for a city",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string"}},
          "required": ["city"]
        }
      }
    }]
  }'
```

Plain chat (no tools):

```bash
curl http://127.0.0.1:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-27b-exl3-2.0bpw","messages":[{"role":"user","content":"Hi"}]}'
```

**Images.** The quant keeps Qwen3.8's vision tower (3-bit), and the server loads it by default (`VISION=auto`), so you can send OpenAI `image_url` content parts — `data:` URLs or http(s) links — and Cherry's attach-picture button works out of the box. Pictures are downscaled to `IMAGE_MAX_PIXELS` (1 MP ≈ 1024 prompt tokens) before encoding. If the tower does not fit next to the ~200k context under the 14.7 GB cap, the Ready box says `Images: off` and the server keeps running text-only — lower `CONTEXT_SIZE` (e.g. `180224`) and restart. Video is not supported.

Defaults: temperature 0.6, top-p 0.95, top-k 20, thinking on. One request at a time; extras queue.

## Installing Simplex on a Windows PC

Four routes, all of which end at the same first-run page.

| route | what you do |
| --- | --- |
| **Zip** | unzip anywhere, double-click `start.bat` |
| **Installer** | run `Simplex-x.y.z-setup.exe`, click Next |
| **winget** | `winget install <publisher>.Simplex` |
| **Scoop** | `scoop install simplex` |

The installer is per-user and needs no administrator rights: it goes into `%LOCALAPPDATA%\Programs\Simplex`, adds Start-menu and (optionally) desktop shortcuts and a sign-in entry, and appears in Add or Remove Programs. Uninstalling asks whether to keep the downloaded model, your settings and your conversations, and keeps them unless you say otherwise.

Build the artifacts with `packaging\build.ps1`; `packaging/README.md` covers the manifests and code signing.

If you unzipped instead of installing, the first successful launch adds the Start-menu and desktop shortcuts for you (`SHORTCUTS=no` in `.env` to skip that).

### Prebuilt wheels: no compiler needed

Compiling the ExLlamaV3 CUDA kernels is the slowest and most fragile part of setup: it wants the CUDA Toolkit and Visual Studio Build Tools, several GB of downloads that have nothing to do with chatting to a model. A wheel built once per Python version removes all of it.

Simplex looks for one in this order:

1. **`wheels\`** next to `start.bat` — what an installer drops in, or what you copy off a USB stick.
2. **`WHEEL_INDEX`** in `.env` — one or more `pip --find-links` targets (a GitHub Releases page, a file share, an internal index).
3. **PyPI**, which has `triton-windows` but not `exllamav3`.
4. **Compiling from source**, which is what it did before.

A wheel is only used when its Python, ABI and platform tags match the interpreter it is going into, so a `cp313` wheel can never land in a `cp312` environment. Check what would be picked:

```
.venv\Scripts\python.exe tools\wheels.py --package exllamav3
```

`wheels/README.md` has the recipe for building one.

### Install it as an app

The chat UI is a progressive web app: in Chrome or Edge an **Install** button appears in the toolbar, and Simplex gets its own window, its own taskbar icon and no browser chrome. On a phone reaching it over Tailscale, "Add to Home Screen" does the same.

Installing works on `localhost` (a secure context). Over a plain `http://` LAN or Tailscale address browsers refuse to register a service worker, so there the UI stays an ordinary page — it still works, it just cannot be installed.

## KV cache formats: int vs fp8 / nvfp4

`CACHE_QUANT` accepts the stock integer formats (`8`, `8,4`, `4`, …) and, since this version, `fp8` and `nvfp4`. The float lanes come from the [MiaAI-Lab exllamav3 fork](https://github.com/MiaAI-Lab/exllamav3) and are carried here as ready-made patched files in `patches/kvcache-fp8-nvfp4-v1.4.4/` (plus the same change as a unified diff); `start.bat` / `start.sh` hot-patch the installed v1.4.4 engine with `tools/patch_kv.py` when you select one — it verifies each target file is stock v1.4.4 by hash, backs it up and swaps in the patched copy (Python + Triton only, no CUDA recompile, reversible with `patch_kv.py revert`). They need an Ada or Blackwell GPU (compute capability 8.9+, e.g. RTX 4060 Ti / 5060 Ti 16 GB) for the in-kernel FP8 conversions.

Two things to know before switching. First, the float formats are **not smaller**: `nvfp4` (E2M1 + one E4M3 scale per 16 values) is 4.5 bits/element exactly like the stock int4 cache (4-bit values + one fp16 scale per 32, after a Hadamard rotation), and `fp8` is 8 bits like int8. Second, the stock integer path is a strong baseline — its Hadamard rotation spreads the outlier channels that keys are full of, whereas `fp8` stores raw E4M3 with no scales at all. A CPU simulation of the four formats on Qwen-like key/value statistics (`tools/kv_format_sim.py`) puts int4 slightly *ahead* of nvfp4 and int8 well ahead of fp8 on attention-weight KL; the fork reports nvfp4 as generation-level lossless on its own models. The real answer is the GPU test: **`test_kv.bat`** runs the fork's kernel-parity checks, then loads the model once per format and reports KL against an fp16 cache on the model's own qbench transcripts, top-1 agreement, a 64k-token passkey retrieval, decode tok/s and VRAM (`--quick` for a short run; results in `kv_cache_tests.json`). Measured on an RTX 5090 (144 probes, KL vs an fp16 cache): int8 1.0e-4, `8,4` 5.4e-4, fp8 4.6e-4, int4 1.2e-3, nvfp4 1.5e-3 (nvfp4 also had the worst tail, 2.9e-2 max, and the only drop in reference accuracy); every format passed a 63k passkey at the same speed. So the stock integer path wins at both sizes and stays the default; for the 150k profile use `CACHE_QUANT=4` — its KL is ~300× smaller than the 2.0 bpw weight quantization's own, i.e. free. The fp8/nvfp4 lanes remain available as opt-in.

## Rebuild the quant

```bash
IN_DIR=/path/to/qwen3.8-27b-hf ./quantize.sh
```

`quantize.sh` passes `-mb 2 -vb 3` so MTP/vision stay at 2/3 bpw (those bits are not read from the recipe file). `cal_trace.safetensors` is the self-calibration trace.

## If you have more than 16 GB

On a **24 GB** card you can raise context to the native **262144** and `GPU_MEM_GB=22`. Do not do that on 16 GB.

## License

Apache-2.0 (inherited from the base model). Kit scripts: [MIT](LICENSE).
