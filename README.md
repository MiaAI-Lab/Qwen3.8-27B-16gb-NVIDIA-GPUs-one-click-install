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

**Windows 11:** double-click **`windows\START-HERE.bat`** once, then **`windows\start.bat`** whenever you want the model.

`windows\START-HERE.bat` opens a page in your browser and does the whole install: it shows what it found on the card, offers the model sizes that fit it, then installs and downloads with a progress bar and a live log. Nothing is asked in the console. When the download is finished it loads that model and hands the page over to the chat, so one double-click takes you from nothing to a working harness. `windows\START-HERE.bat --no-start` stops after the install instead — that is the one to use for fetching a second size without loading it.

`windows\start.bat` starts a model that is already here. If you have downloaded more than one size it lists them and asks which to load (Enter takes the one that ran last, and it starts on its own after 45 seconds so an unattended machine still comes up); then it loads the model and opens the [DeepSeek Harness](#the-deepseek-harness) at `http://127.0.0.1:3080/`. If nothing is installed yet it says so and offers to run setup for you, so double-clicking the wrong one is never a dead end.

Two files rather than one because they are two different questions: `windows\start.bat` never downloads anything, and `windows\START-HERE.bat --no-start` never loads anything. What `windows\START-HERE.bat` does *not* do is stop halfway with a page still promising a chat window — the install and the first start are one click, or the promise is a lie you find out about four minutes later.

The only thing you have to install first is **64-bit Python 3.11+** — and both scripts say so plainly if it is missing. Everything else Simplex installs into its own `.venv` folder. The harness additionally wants **Node** ([nodejs.org](https://nodejs.org/), LTS); without it the model still serves `/v1` and the launcher says what is missing. The engine arrives as a prebuilt wheel — the launcher installs torch, reads the version and CUDA line it actually got, and pulls the matching wheel from the engine's own GitHub release, so the first run is a ~100 MB download rather than a compile. **No CUDA Toolkit, no Visual Studio Build Tools, no Git.** Those are only needed in the cases where no wheel fits — a CUDA line or platform the engine has no build for (aarch64/GB10), or no route to github.com — where it falls back to building from source. See [Prebuilt wheels](#prebuilt-wheels-no-compiler-needed).

The download is resumable: closing the window, losing the connection or rebooting costs you nothing, because it picks up from the byte it stopped at. A model that was left half-downloaded is shown as such — in the menu, and in the list `windows\start.bat` offers — and running `windows\START-HERE.bat` again finishes it. Nothing offers to *start* a model until every one of its weight files is on the disk.

Later launches skip setup. Right before the model loads, `windows\start.bat` checks that about **14.7 GB of VRAM is actually free** (the whole recipe needs it) and, if not, lists the programs holding VRAM — browsers, games, Discord, other AI tools — and waits for you to close them (Enter re-checks, `c` continues anyway, `q` quits). Take that seriously on Windows: with too little free VRAM the driver pages the model into system RAM instead of failing, and it then runs many times slower.

While it runs, Simplex puts an icon in the notification area — right-click for **Open Simplex**, **Restart the model**, **Show the Simplex folder**, **View the log** and **Quit**. Every launch also writes a full transcript to `logs\`, so a crash that scrolls past is still readable afterwards. `.venv`, `models/`, `logs/` and `apps/` stay on your machine and are not part of the git tree.

To go back to the old console questions instead of the web page, set `SETUP=console` in `.env`.

**Linux:**

```bash
./linux/setup.sh              # once: venv + ExLlamaV3 v1.4.4 + the weights
./linux/start.sh              # then: pick a downloaded model and serve it
# http://localhost:8888/v1   and the harness on :3080
```

The same split as on Windows, and `.env` is created for you on the first run of
either. There is no setup page on Linux: `./linux/setup.sh` asks the profile
questions in the terminal (`tools/profiles.py`), builds the venv and downloads
the weights, then stops. `./linux/start.sh` lists the models that finished
downloading, asks which one (Enter is the last one used), and serves. No tray
icon and no shortcuts either — those are Windows. Everything after that is the
same, the harness included: once the model answers `/health`, `tools/dsh.py`
configures it and starts it, and prints the address to open. On a box with no
desktop session `webbrowser` has nothing to open, so the address is printed for
you to copy — take the whole thing, token and all.

### One command, both systems

The two files above are the double-click doors. Everything else — and every
verb on either system — is `simplex`:

```bash
./linux/simplex start                # load a model and serve it
./linux/simplex start --no-harness   # ...serving /v1 only
./linux/simplex start -b             # ...in the background, log in logs/
./linux/simplex status               # what is running, which model, which ports
./linux/simplex harness start        # attach the harness to a server already running
./linux/simplex stop --harness-only  # ...and detach it again, model still loaded
./linux/simplex stop                 # stop both
./linux/simplex restart              # stop, then start
./linux/simplex logs -f              # follow the launcher log
./linux/simplex models               # what is on the disk, and what is half-downloaded
./linux/simplex doctor               # check this machine before blaming the model
```

On Windows it is `windows\simplex.bat` with the same verbs, because it is the same
program: `tools/cli.py`. `windows\START-HERE.bat`/`linux/setup.sh`, `windows\start.bat`/`linux/start.sh` and
`windows\stop.bat`/`windows\stop.bat` still work exactly as before — `stop` and everything new
now run one implementation rather than one per system, which is what stops the
two drifting apart.

**With or without the harness.** `UI=` in `.env` is the standing answer
(`browser`, `server` or `no`); `--harness` / `--no-harness` overrides it for one
run, on `simplex start`, `linux/start.sh` and `windows\start.bat` alike. The harness is also a
verb of its own — `simplex harness start|stop|status|open|settings` — so it can
be attached to a model that is already loaded, or taken away without unloading
one. It configures itself from `/v1/models` either way, so what it offers is
what actually loaded.

`simplex doctor` is the first thing to run when something is wrong: it checks
the Python version, the venv and the engine version inside it, the driver and
the card, Node, both ports and who holds them, the `.env` values that have to be
valid, whether the weights are all there, and the free disk.

Requires **ExLlamaV3 v1.4.4** (quantized vision tower). PyPI skips 1.4.4 (`1.4.2` → `1.4.5`); the launchers install the git tag. Engine: [ExLlamaV3](https://github.com/turboderp-org/exllamav3).

## Profiles: the launcher picks quants for your GPU

Setup (`windows\START-HERE.bat` / `./linux/setup.sh`, or `PROFILE=ask` in `.env`) runs `tools/profiles.py`: it reads the card's VRAM with `nvidia-smi`, computes what fits under a budget of *VRAM − max(1.3 GB, 8 %)* using the measured weight sizes, KV cost per token (`8,4` 26 KB, `4` 18 KB, MTP draft cache +1/16), the vision tower (measured: 0.87 GB, or 0.17 GB for the 3-bit-quantised 2.0 bpw one) and 2.6 GB of runtime overhead (measured against real prefill peaks; it was 1.7, which under-predicted every quant by more than the safety margin covered), and offers the best-quality and longest-context options (plus a middle one when they are far apart). Enter takes the recommendation; if a model is already downloaded, "keep current" is the default so an unattended start never triggers a surprise download. The choice is written into `.env` (`MODEL_DIR`, `HF_TARGET_REPO`, `HF_REVISION`, `MODEL_ID`, `CONTEXT_SIZE`, `CACHE_QUANT`, `GPU_MEM_GB`, `VISION`) and everything downstream — server, Cherry's model entry and defaults — follows it.

| VRAM | offered (KV cache is always int4 — measured within 0.001 KL of fp16, and it has no hardware requirement: it runs on every supported GPU) |
| --- | --- |
| 12 GB | 2.0 bpw @ 33k, text-only — the floor, and the whole menu |
| 16 GB | **2.5 bpw @ 176k** with images · 3.5 bpw @ 78k text-only · 3.0 bpw @ 118k with images · 2.0 bpw @ 229k with images |
| 24 GB | **4.0 bpw @ 262k** with images · 5.0 bpw @ 176k with images · 6.0 bpw @ 82k text-only · everything below at 262k |
| 32 GB+ | **4.0 bpw @ 262k** with images · 5.0 bpw @ 176k · 6.0 bpw @ 82k — the two top rows are capped at what a prefill has survived |

Where a real prefill has been run at a stated budget, the menu offers what was
measured rather than what the formula computes. The context was grown on a
14.7 GB budget until a prefill failed (2026-09-06), and the formula had been
leaving a lot on the table:

| budget | quant | planner offered | measured, text | measured, images |
| --- | --- | --- | --- | --- |
| 14.7 GB | 3.5 | 0 | 77824 | 41984 |
| 14.7 GB | 3.0 | 69888 | 148480 | 117760 |
| 14.7 GB | 2.5 | 186368 | 212224 | 176128 |
| 22.1 GB | 6.0 | 8960 | 83712 | 57088 |
| 22.1 GB | 5.0 | 163072 | 204800 | 179712 |
| 22.1 GB | 4.0 | 262144 | 262144 | 262144 |

The flat 2.6 GB overhead is a bound over every quant, so on any one of them it is
slack — 3.5 bpw is the extreme case, priced out of a 16 GB card entirely by a formula
that the card then ran at 78k tokens, and 6.0 bpw is not far behind it at 9k against
82k. Above the budget it was measured under the formula takes over again; below it,
the measurement only ever lowers the answer.

Two rows are also *capped* at their measurement: nothing has ever prefilled past
204800 tokens on 5.0 bpw or 83712 on 6.0 bpw, at any budget, so neither plans past
it. (Their older ceilings — 183296 for 5.0, "nothing survived" for 6.0 — came from a
run with ~40 other processes on the card, and a clean run at a *tighter* budget beat
both, which is how you tell contention from a ceiling.)

Bold is what the launcher pre-selects: the best quality that still has real context
(≥ 128k), not the longest context. On a 16 GB card that is the 2.5 bpw row — 3.0 bpw
is the better model, but it fits 66k tokens there and only text-only, against a
measured 176k with images one rung down. The pick also keeps images where it can: on a 24 GB
card 5.0 bpw clears 128k only by dropping the vision tower, so the default steps one
rung down to 4.0 bpw, which holds native context with images. One rung, never more —
and answering "no images" puts 5.0 bpw back. `python tools/profiles.py --list --vram 16`
shows this table for any specific card without owning one.

Quants other than the 2.0 bpw baseline are pulled from turboderp's branches (`HF_REVISION`); their vision towers are unquantised (0.87 GB measured), which is why images are off on the tight profiles. The KL figures behind the menu's quality words are turboderp's (mean KL vs bf16): 2.0 → 0.35 *fair*, 2.5 → 0.30 *good*, 3.0 → 0.11 *better*, 3.5 → 0.08 *very good*, 4.0 → 0.05 *very good*, 5.0 → 0.014 *excellent*, 6.0 → 0.007 *near-lossless*. `python tools/profiles.py --list --vram 24` previews the menu for any card.

## Chat with the model

The server is **OpenAI-compatible**. Leave `windows\start.bat` / `linux/start.sh` running, then point a client at it. There is **no API key**; many apps still require a dummy value such as `local`.

| | |
| --- | --- |
| Base URL | `http://127.0.0.1:8888/v1` |
| API key | `local` (ignored) |
| Model | `qwen3.8-27b-exl3-2.0bpw` |

### The DeepSeek Harness

The kit serves the model. What you talk to is
**[DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness)** (`dsh`,
MIT) — DeepSeek's open-source agent harness, started as a second process once
the model is loaded and answering at **`http://127.0.0.1:3080/`**. `windows\start.bat`
and `linux/start.sh` open it for you; `http://127.0.0.1:8888/` is a small page saying
where everything is, which forwards there as soon as the harness answers.

None of it is vendored here. It is a Node application, so the launcher runs it
with `npx` and npm caches it after the first run — the one thing you have to
install is **Node** (the LTS build from [nodejs.org](https://nodejs.org/)). No
Node, no harness: the launcher says so, and `/v1` keeps serving every other
client regardless. `UI=no` in `.env` skips it entirely.

**It configures itself against whatever loaded.** Before starting the harness,
`tools/dsh.py` asks the running server on `/v1/models` what it actually is —
the model id, the context window, whether the vision tower fit, and which
reasoning levels this chat template accepts *and acts on* — and writes that as
a provider route into `.dsh/settings.yaml`, and names that route as the default
the picker opens on. So the model, its context meter, its image support and its
effort menu are right on the first launch with nothing typed into a form — and
a new chat starts on the model this PC just loaded rather than on dsh's own
default, which is a hosted DeepSeek model and is remembered per session once
it has been used. After a fallback (no room for the vision tower, say) the
harness is told what happened rather than what `.env` hoped for. Switching
quants rewrites the same file; the harness re-reads it per request, so it never
needs restarting.

Those two keys are yours the moment you edit them. The launcher keeps a copy of
what it last wrote beside the file and stops generating as soon as they differ,
so a hand-tuned route — a different effort vocabulary, other `compat` switches
— survives every restart. Delete the file to get a fresh one. Everything else
in it is read past and written back untouched: a second provider you added, and
the state dsh itself keeps there (it records the notices it has shown you, and
respaces the file on the way through), so neither one freezes the join. Every
field it accepts is in
[dsh's configuration catalog](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/config-catalog.md).

**The first address is not the plain one.** dsh authenticates the browser with
a token it mints fresh on every launch and prints once, as
`dsh web: http://127.0.0.1:3080/?token=...`. Opening *that* sets a cookie good
for thirty days and redirects to a clean `/`; arriving at the bare address
without it answers **"dsh web authentication required; reopen the URL printed
by dsh web"**. So the launcher reads the address off dsh's own output rather
than composing it from the port, and `http://127.0.0.1:8888/` forwards through
`/harness`, which knows the current one. If you need it by hand, it is in the
launcher window and in `logs\`.

That route only answers a browser on this computer, even with `HOST=0.0.0.0`:
the token is a session on an agent that runs commands here.

| | |
| --- | --- |
| Harness | `http://127.0.0.1:3080/` (`DSH_PORT` in `.env`) |
| Version | `DSH_VERSION` in `.env`, pinned; `latest` follows the newest |
| Its home | `.dsh/` next to `windows\start.bat` — settings, credentials, profiles, plugins |
| Run it alone | `python tools/dsh.py --open` (with the server already up) |
| Just the settings | `python tools/dsh.py --settings-only` |

The harness binds loopback only and **refuses to bind `0.0.0.0` at all**: its
agent runs commands on this PC and there is no login. To reach it from a phone,
put a proxy in front of it rather than opening the port — `tailscale serve --bg
3080`. Note that the shipped `HOST=0.0.0.0` already exposes `/v1` (the API, not
the harness) to whatever network you are on; set `HOST=127.0.0.1` if that
network is not yours.

Its workspace, approval policy, tools, MCP servers and plugins are all its own
— see [its documentation](https://deepseek-harness.github.io/deepseek-harness/).
Pick a workspace folder in it before the first message.

### Any other OpenAI client

`/v1` is a plain OpenAI endpoint and always has been, so nothing about the
harness is compulsory: llama.cpp-compatible front ends, Open WebUI, Cherry
Studio (below), a `curl`, an SDK, another machine on your network — all of them
work against the base URL above, at the same time as the harness does.

### Simplex, the UI this kit used to ship

Up to this version the kit served its own chat and agent UI in the server
process. That UI is now **[Simplex](../simplex)**, a standalone project: nothing
in it was specific to this model or this server, and it talks to any
OpenAI-compatible endpoint — including this one, at
`http://127.0.0.1:8888/v1`. Its conversations, projects and providers moved with
it, so an existing install picks up where it left off. Run its `windows\start.bat` or
`./linux/start.sh` beside this one and set `UI=no` here if you want it back in place
of the harness, or run both.

### Cherry Studio (optional, off by default)

The kit used to ship [Cherry Studio](https://github.com/CherryHQ/cherry-studio) as its chat app; the built-in UI above replaced it. It is still wired up for anyone who wants Cherry's assistants, knowledge bases and MCP servers — set `CHERRY_AUTOSTART=ask` (or `yes`) in `.env` and it works exactly as before:

* `windows\start.bat` downloads the pinned **portable** build (v2.0.10, ~285 MB, once) into `apps/cherry-studio/`. Nothing is installed system-wide; Cherry keeps its data in `apps/cherry-studio/data/`.
* The first time, Cherry opens and closes once by itself to create that data folder. The kit then writes its configuration straight into Cherry's store: provider **Simplex (local)** → `http://127.0.0.1:8888/v1`, key `local`, model `qwen3.8-27b-exl3-2.0bpw` (tool calling + image input on, ~200k context), set as the default chat model and on the default assistant, onboarding skipped. Usage analytics is switched off (Cherry → Settings → Privacy to change).
* After the **Ready** box: *"Open Cherry Studio and start chatting now? [Y/n]"*. Enter/`y` opens it (the portable build unpacks for ~10–20 s), `n` or no answer within 90 s leaves it closed. Keep the server window open while chatting; `windows\stop.bat` stops the server only.
* Changing `PORT` in `.env` re-points the provider on the next start. Open Cherry later without the prompt: `.venv\Scripts\python.exe tools\cherry.py open` (`status` shows what the kit thinks).
* `.env` knobs: `CHERRY_AUTOSTART=ask|yes|no` (`no` also skips the download), `CHERRY_VERSION` (pinned; the store layout is checked against v2.0.x), `CHERRY_EXE=<path>` to use a Cherry Studio you already installed — the kit then only sends Cherry's official import link (`cherrystudio://providers/api-keys`), you confirm the popup and add the model id under the new provider.
* **Web search and tools are on by default.** The default assistant gets `web_search` + `web_fetch` as function tools (Cherry's stock keyless search provider, Exa MCP at `mcp.exa.ai`; change it under Settings → Web Search), runs MCP in *auto* mode, and the kit installs Cherry's keyless builtin MCP servers `@cherry/fetch` and `@cherry/sequentialthinking`. The model decides when to call them; the server turns Qwen's XML tool calls into OpenAI `tool_calls` (below). Tune with `CHERRY_WEB_SEARCH=1|0` and `CHERRY_MCP_SERVERS=` (also `@cherry/python`, `@cherry/browser`; empty = none) in `.env` — re-applied on the next start when you change them. New assistants you create in Cherry start with Cherry's own defaults (web search off) unless you copy the default one. Cherry Studio is [AGPL-3.0](https://github.com/CherryHQ/cherry-studio/blob/main/LICENSE) (see its README for the commercial-use terms); the kit downloads the official release binary and does not redistribute it.

The harness starts on Linux too (`./linux/start.sh`, then open the address it prints — the bare `http://127.0.0.1:3080/` is refused until a token has been through it once). Manual setup for any other OpenAI client (Chatbox, Open WebUI, Continue, Cursor): add an **OpenAI-compatible** provider with base `http://127.0.0.1:8888/v1` (or host `http://127.0.0.1:8888` if the app appends `/v1` itself), API key `local`, model id `qwen3.8-27b-exl3-2.0bpw`. Any other OpenAI client works the same way (Chatbox, Open WebUI, Continue, Cursor custom endpoint). Open WebUI is stronger if you want a big tools/RAG UI and are fine running Docker.

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

Two routes, both of which end at the same first-run page.

| route | what you do |
| --- | --- |
| **Zip** | unzip anywhere, double-click `windows\START-HERE.bat`, then `windows\start.bat` |
| **Installer** | run `Simplex-x.y.z-setup.exe`, click Next |

The installer is per-user and needs no administrator rights: it goes into `%LOCALAPPDATA%\Programs\Simplex`, adds Start-menu and (optionally) desktop shortcuts and a sign-in entry, and appears in Add or Remove Programs. Uninstalling asks whether to keep the downloaded model, your settings and your conversations, and keeps them unless you say otherwise.

If you unzipped instead of installing, the first successful launch adds the Start-menu and desktop shortcuts for you (`SHORTCUTS=no` in `.env` to skip that).

### Prebuilt wheels: no compiler needed

Compiling the ExLlamaV3 CUDA kernels is the slowest and most fragile part of setup: it wants the CUDA Toolkit and Visual Studio Build Tools, several GB of downloads that have nothing to do with chatting to a model. Nobody has to do it, because the engine publishes wheels itself.

Simplex looks for one in this order:

1. **`wheels\`** next to `windows\start.bat` — what an installer drops in, or what you copy off a USB stick.
2. **The engine's own release** — `turboderp-org/exllamav3` attaches a wheel per (CUDA line × torch version × Python). This is the normal path and needs no configuration.
3. **`WHEEL_INDEX`** in `.env` — one or more `pip --find-links` targets (a GitHub Releases page, a file share, an internal index).
4. **PyPI**, which has `triton-windows` but not `exllamav3`.
5. **Compiling from source**, for the cases none of the above covers.

Step 2 resolves to one exact URL rather than pointing pip at the release page, and that distinction matters. The CUDA line and torch version live in the wheel's *local version* (`1.4.4+cu128.torch2.10.0`), which pip does not match against anything: given `--find-links` it filters on the Python and platform tags only, then takes the highest version string. A torch 2.10 environment would be handed the torch 2.11 build, and the failure arrives later as an undefined-symbol `ImportError` that reads like a corrupt install. So the launcher installs torch first, asks the venv what it actually got, and names the one wheel that fits.

This is also why the default PyTorch index is **cu128**: the engine builds for cu128 and cu132 only, so torch from any other line means no wheel exists and everyone compiles. cu128 covers Blackwell and needs driver 570+.

A wheel is only used when its Python, ABI and platform tags match the interpreter it is going into, so a `cp313` wheel can never land in a `cp312` environment. Check what would be picked:

```
.venv\Scripts\python.exe tools\wheels.py
```

That prints the venv's tags, the torch version and CUDA line found, and the wheel it would install. `wheels/README.md` covers the override cases — no network, an unbuilt CUDA line, aarch64/GB10 — and the recipe for building one yourself.

### Install it as an app

Whether the browser offers an **Install** button depends on the front end you use, not on this kit — the harness and Simplex each serve their own page. Installing only ever works on `localhost` or over HTTPS (a secure context); over a plain `http://` LAN address browsers refuse to register a service worker, so there the page stays an ordinary one.

## If you have more than 16 GB

On a **24 GB** card you can raise context to the native **262144** and `GPU_MEM_GB=22`. Do not do that on 16 GB.

## License

Apache-2.0 (inherited from the base model). Kit scripts: [MIT](LICENSE).
