# Handoff — Qwen3.8-27B EXL3 2.0bpw Windows kit

**Open this folder as the workspace on Windows:** `C:\Users\zurih\Documents\WINDOWS\qwen\`

Prior work was done from Linux Spark (`/home/zurih/NewModels/Qwen3.8-27B-EXL3-2.0bpw`). That tree has the latest launcher/server edits. The Windows copy may be behind. **Copy these two files onto Windows before debugging anything else:**

- `tools/win_start.py`
- `tools/serve_openai.py`

Also keep `start.bat` / `stop.bat` / `.env.example` / `README.md` in sync if they differ.

Prior chat: Cursor agent transcript `8321fb4e-9300-4d63-90b6-ca9954ed0703` (Windows Qwen kit).

## What this is

One-click local OpenAI-compatible server for **Qwen3.8-27B EXL3 2.0bpw** (`SC_2.00bpw_H3_V3`, turboderp) aimed at **16 GB NVIDIA**.

- Engine: **ExLlamaV3 v1.4.4** (vision_bits 3; PyPI skips 1.4.4, install git tag)
- Serve: `tools/serve_openai.py` via `start.bat` → `tools/win_start.py`
- Weights Hub: `Mia-AiLab/Qwen3.8-27B-EXL3-2.0bpw` → `models/Qwen3.8-27B-EXL3-2.0bpw/` (~9.7 GB, gitignored)
- Banner: **Simplex - one-click Qwen3.8-27B for Windows** (ASCII only; cmd turns em dashes into `ΓÇö`)

## Validated 16 GB recipe (do not “fix” these without measuring)

| Knob | Value |
| --- | --- |
| `GPU_MEM_GB` | `14.7` |
| `CONTEXT_SIZE` | `199936` (multiple of 256) |
| `CACHE_QUANT` | `8,4` |
| `DRAFT` | `mtp` |

Native **262144 does not boot** on 14.7 GB. Idle load ~12.35 GiB allocated / 13.37 GiB reserved. MTP is cheap; context/KV is what blows VRAM.
(That was with `CACHE_QUANT=8,4`, which costs 26 KB/token. On the stock int `4`
cache the 2026-09-03 bench reached the full 262144 on 2.0bpw at a measured 13.75
GiB peak - the cache format, not the context, was what did not fit.)

VRAM cap lives in `serve_openai.py` `_cap_process_vram()` (CUDA fraction + ExLlama `set_memory_fraction_*` patched so load stays capped).

## API

| | |
| --- | --- |
| Base | `http://127.0.0.1:8888/v1` |
| Key | `local` (ignored) |
| Model id | `qwen3.8-27b-exl3-2.0bpw` |

`GET /v1` is **404** (prefix only). Use `/health` (now reports `vision: true|false`), `/v1/models`, `POST /v1/chat/completions`. Defaults: temp 0.6, top-p 0.95, top-k 20, thinking on. Batch-1; extra requests queue.

**Images (added 2026-09-02, NOT yet run on the PC):** `serve_openai.py` `--vision auto|off` (from `.env` `VISION`) loads the vision component after the text model (`Model.from_config(config, component="vision")`), never fatal. `image_url` parts (data:/http(s) only, local paths refused) → `decode_image()` (Pillow, downscale to `--image_max_pixels`, from `.env` `IMAGE_MAX_PIXELS`) → `vm.get_image_embeddings(tokenizer, img)` under `gen_lock` → `hf_render_chat_template()` text, each `<|vision_start|><|image_pad|><|vision_end|>` replaced by the embedding's `text_alias` (its token_string already carries start/end, so do NOT use `hf_chat_template(embeddings=)`, which only swaps `<|image_pad|>` and would double the markers) → `tokenizer.encode(..., embeddings=)` → `Job(..., embeddings=)`. Checked in the v1.4.4 source: the MTP draft path passes `indexed_embeddings` through (only the external-draft path asserts against MM embeddings), so `DRAFT=mtp` + images is fine. VRAM: earlier load measured 12.35 GiB alloc / 13.37 reserved under the 14.7 cap; the tower measured 0.173 GiB on the 2.0bpw build (its own weights are 3-bit) and 0.87 GiB on the plain quants, plus transient activations; if not, lower CONTEXT_SIZE (180224). `win_start.py` installs `pillow` (`ensure_pillow()`) and passes the flags; `cherry.py` gives the model `image-recognition` + `input_modalities ["text","image"]` when VISION != off (re-applied when it changes).

**Chat app:** the kit now serves its **own web UI** from the model server (`tools/chatui.py` + `tools/webui*`, see the section below). `UI=browser` in `.env` opens `http://127.0.0.1:PORT/` after Ready. Cherry Studio stays in the tree but is **off by default** (`CHERRY_AUTOSTART=no`): `tools/cherry.py`, driven by `win_start.py`. Manual fallback for any client: OpenAI-compatible provider, base `http://127.0.0.1:8888/v1`, key `local`, model id above. Enable tools/MCP on the assistant for function calls.

## Cherry Studio integration (added 2026-09-02, NOT yet run on the PC)

- `win_start.py` → before the model load: `cherry.prepare()`; after the server starts, a thread polls `/health` and then asks **"Open Cherry Studio and start chatting now? [Y/n]"** (Enter = yes, 90 s timeout = stays closed). `.env`: `CHERRY_AUTOSTART=ask|yes|no`, `CHERRY_VERSION=2.0.10`, `CHERRY_EXE=` (use an existing install → deep link only).
- `tools/cherry.py`: downloads `Cherry-Studio-2.0.10-win-x64-portable.exe` from GitHub releases into `apps/cherry-studio/` (gitignored, ~285 MB). Portable build keeps its userData in `apps/cherry-studio/data/` (electron-builder `PORTABLE_EXECUTABLE_DIR/data`), DB at `data/Data/cherrystudio.sqlite`.
- First run: `first_run_init()` launches Cherry, waits for `app_state` key `seedRunner:bootstrapCompleted` + an assistant row, +8 s, then `taskkill /F /T` on the child of the NSIS stub (so the stub cleans its temp dir). Then `configure_db()` writes with stdlib sqlite3:
  - `user_provider` `mia-qwen-local` (custom, `preset_provider_id` NULL, `endpoint_configs={"openai-chat-completions":{"baseUrl":".../v1"}}`, `api_keys=[{id,key:"local",isEnabled}]`, enabled, first `order_key` via a fractional-indexing port)
  - `user_model` `mia-qwen-local::qwen3.8-27b-exl3-2.0bpw` (custom: name/capabilities `["function-call"]`/supports_streaming required by a CHECK; `context_window`=CONTEXT_SIZE)
  - `preference` `chat.default_model_id`, `feature.quick_assistant.model_id`, `feature.translate.model_id` → that id; `app.onboarding.provider_setup.status="completed"`; `app.privacy.data_collection.enabled=false` (avoids the privacy-policy dialog and telemetry)
  - `assistant.model_id` → ours where it was NULL or `cherryai::qwen`
  - **tools (added same day):** `assistant.settings` JSON gets `enableWebSearch: true` + `mcpMode: "auto"`; rows in `mcp_server` for `@cherry/fetch` and `@cherry/sequentialthinking` (`type='inMemory'`, `install_source='builtin'`, `is_trusted=1`, active — same shape as Cherry's Install button) linked in `assistant_mcp_server`. Web search needs no key: default keyword provider `exa-mcp` (`WEB_SEARCH_PROVIDER_PRESET_MAP`, requiresApiKey false) is "ready" out of the box; `resolveWebToolRoutes` gives a `function-call` model the client `web_search`/`web_fetch` tools. Knobs `CHERRY_WEB_SEARCH`, `CHERRY_MCP_SERVERS` (allowed: fetch, sequentialthinking, python, browser); removing a server from the list does not deactivate it.
  - marker `apps/cherry-studio/kit-config.json` (incl. `tools`); later runs only re-point the base URL when PORT changes or re-apply tools when the CHERRY_* tool lines change.
- Schema facts were taken from the cherry-studio `main` tree at v2.0.10 (`src/main/data/db/schemas/*`, `seeding/*`, `preferenceSchemas.ts`). Cherry extracts inline `<think>` for openai-chat-completions itself (`reasoningExtraction.ts`) — do not add the `reasoning` capability. Not added: REASONING/vision capabilities.
- Verified in the sandbox against a DB built from upstream `migrations/sqlite-drizzle` (FK check clean, idempotent, port change re-points). **Not verified on Windows yet**: the real portable first run, the kill/relaunch dance, and a live chat. If Cherry shows a Chinese UI, set language in its settings (`app.language` was left at system default).
- Deep link (for `CHERRY_EXE` or debugging): `cherry.py link` prints `cherrystudio://providers/api-keys?v=1&data=<b64>`; Cherry pops a confirm and creates the provider+key, models still need "Manage".

**Tools:** server accepts OpenAI `tools`; model emits Qwen XML; server parses to `tool_calls`; client must reply `role: tool`. **No built-in internet.** Web search only if Cherry/MCP actually fetches.


## Built-in web UI (added 2026-09-03, NOT yet run on the PC)

Replaces Cherry as the default chat app. Same process as the server, no extra
download, no third-party schema to track.

- `serve_openai.py --ui on|off` (default on) calls `mount_ui()`, which imports
  `tools/chatui.py` and attaches the UI routes to the same aiohttp app; a
  failure there prints one line and the model server keeps serving. `/v1` is
  untouched - the UI is just another OpenAI client, over `127.0.0.1:PORT/v1`.
- `win_start.py`: `ui_mode(cfg)` reads `UI=browser|server|no`; a thread waits
  for `/health` and opens the browser. `cherry_mode()` now defaults to `no`.
- Files:
  - `tools/chatui.py` - two adapters over the core: `mount(app, ui)` for
    aiohttp, and a stdlib `http.server` dev server (`--mock` serves a scripted
    model so the UI can be worked on with no GPU: `python tools/chatui.py
    --mock --open`).
  - `tools/webui_app.py` - framework-agnostic core: routing, static files,
    sessions (`sessions/*.json`, OpenAI-shaped messages so UI and model never
    drift), the approval hand-shake, `/ui/browse` folder picker.
  - `tools/webui_agent.py` - `ModelClient` (streaming OpenAI over urllib) and
    `run_turn()`, a **synchronous generator** of UI events. Sync on purpose:
    the same generator drives both adapters and the tests.
  - `tools/webui_tools.py` - tool registry, schemas, risk classes
    (safe/write/exec), `Workspace` sandbox, keyless web search + HTML-to-text.
  - `tools/webui_mock.py`, `tools/test_webui.py` - scripted model, checks.
  - `tools/webui/{index.html,app.js,style.css}` - the page. No framework, no
    CDN, no web fonts (it must work offline); own ~90-line Markdown renderer;
    `localStorage` only for sampling settings and the theme choice.
    Design pass 2026-09-03: token-based palette (edit the two :root blocks to
    re-skin), inline SVG sprite instead of emoji glyphs, sliding mode toggle,
    message entrance/typing/shimmer/caret motion with a
    `prefers-reduced-motion` escape, code blocks with language + copy button,
    per-answer copy row, suggestion chips, sessions grouped by day, toasts,
    theme toggle, scroll-to-newest, Ctrl/Cmd+K for a new chat, and a mobile
    layout (drawer sidebar, shorter placeholder, no keyboard hints).
    **Two traps worth remembering:** every component sets its own `display`,
    so `[hidden] { display: none !important; }` is load-bearing (without it
    Send and Stop show at once); and an approval must render *inside* its tool
    card, or the result appears above the question that produced it.
- **Modes.** Chat = `web_search`, `web_fetch`. Agent = plus `list_dir`,
  `read_file`, `find_files`, `search_text`, `write_file`, `edit_file`,
  `run_python`, `run_command`. Switching mode in the browser starts a new chat
  so one transcript never mixes toolsets.
- **Approvals.** `write`/`exec` tools yield an `approval` event; the app layer
  registers the pending request *before* the event is written to the wire (key
  = the tool-call id), so a fast click cannot 404. `POST /ui/approve`
  {id, allow|always|deny} releases a `threading.Event`; "always" is
  per-session, in memory. `APPROVAL_TIMEOUT` (default 900 s) denies.
- **Network.** `ChatUI.allow_lan` (`UI_LAN`, default 0): every UI route is
  403 for a non-loopback peer, since `HOST=0.0.0.0` would otherwise hand the
  whole LAN a shell through Agent mode. Adapters pass `request.remote` /
  `client_address[0]`; `/v1` is untouched.
- **Sandbox.** Every path goes through `Workspace.resolve()`: resolved, then
  refused unless the root is a parent. Tested against `..`, absolute paths and
  writes.
- **tok/s (added 2026-09-03).** `serve_openai.py` now honours
  `stream_options.include_usage`: the streaming path carries `ptoks/otoks`
  through the worker's done payload and writes one final chunk with `usage`
  before `[DONE]` (non-stream already had it). `ModelClient` asks for it,
  `run_turn` sums *model* time only (`gen_seconds`, tool time excluded), and
  the `done` event carries `usage`, `seconds`, `tok_s`. The UI shows that exact
  figure per answer and in the hint; the live pill in the top bar is a
  different source - deltas of `/health`'s `completion_tokens_total`, polled at
  1 s while streaming and 5 s idle (`scheduleHealth`; the cadence must be bumped
  from `setStreaming`, or a short turn ends before the first busy poll). The dev
  server answers `/health` itself under `--mock` (`ui.fake_health`), and
  `CHATUI_MOCK_DELAY` slows the fake stream so the live readout can be seen.
- **Native folder picker (added 2026-09-03).** `tools/webui_picker.py`:
  Windows runs PowerShell `-STA` with a WinForms `FolderBrowserDialog` (script
  on stdin, start path passed in `WEBUI_PICK_START` so nothing is interpolated
  into a command line, topmost owner form so the dialog lands in front); other
  platforms run tkinter's `askdirectory`. Always a child process, one at a time
  (`_lock`), 300 s timeout, `PickerUnavailable` on any failure.
  `POST /ui/pick_folder` is loopback-only (409 for other devices, since the
  dialog opens on the server's screen) and 409 when `FOLDER_PICKER=browser`;
  the UI falls back to the built-in browser on any refusal. **Not yet run on
  Windows** - only the failure branch has been exercised here.
- **Agentic upgrades borrowed from DeepSeek Harness (added 2026-09-03).**
  Read their `docs/tool-catalog.md` before changing tool descriptions - the
  wording there is the reference. What landed:
  * `ToolContext` (workspace, cfg, `observed` files, `plan`, `jobs`, `ask`)
    replaces the bare Workspace argument and is **kept per session** in
    `ChatUI._contexts`, so read-before-edit records and background jobs
    survive between turns.
  * read-before-mutate: `_guard_write` + the `edit_file` check, message and
    remedy modelled on their `FS_NOT_OBSERVED` (`edit requires reading "x"
    first` / `read the file, then retry`).
  * `ToolError(message, hint)` and `.render()` - every failure carries the
    recovery instruction; the loop returns `error:` + `hint:` to the model.
  * `update_plan` (their `todo_write`): whole list per call, one `in_progress`,
    statuses pending|in_progress|completed. Emits a `plan` event; stored on the
    session as `plan` and rendered as the progress strip (`#plan-bar`).
  * `ask_user` (their `ask_user_question`): question + header + options.
    Same hand-shake as approvals - the loop yields `question`, the app
    registers the pending record before it goes out, `POST /ui/answer` sets the
    text. `QUESTION_TIMEOUT` (default 1800 s) gives up.
  * `run_command`: `workdir`, `description` (shown on the card), `[exit code:
    N]`, tail-kept output with the whole thing spilled to `.simplex/`, and
    `run_in_background` + `job_output` / `job_kill`.
  * `read_file` is line-numbered with `offset`/`limit` (2000 default) and is
    what records observation; `search_text` groups by file and caps at 250.
  * `UNTRUSTED` banner on web results; both system prompts say external text
    is data, not instructions.
  * `.env`: `AGENT_ASK` toggles ask_user.
- **Model switching (added 2026-09-03).** `tools/webui_models.py`:
  `installed()` scans `models/*` for weights, matches the folder name against
  `profiles.QUANTS` for quality/size, and pre-plans each one's context so the
  UI can grey out a quant this card cannot hold; `settings_for()` recomputes
  CONTEXT_SIZE / GPU_MEM_GB / VISION with profiles' own `budget_gib`,
  `max_ctx`, `nice_ctx` (cache pinned to int4); `apply()` refuses anything
  outside `models/` and writes .env through `profiles.write_env`.
  `POST /ui/switch_model` is loopback-only, refuses while a turn is streaming,
  and then calls `ui.on_restart()` - installed by `chatui.mount`, which
  `os._exit(87)`s after 0.7 s so the response reaches the browser first.
  **The launcher is what makes this work:** `win_start.server_command(cfg)` was
  extracted so the args can be rebuilt from a re-read .env, and main() now
  loops - exit 87 means re-read, re-run the VRAM pre-flight, relaunch in the
  same console window (`SIMPLEX_SUPERVISED=1` in the child env is what makes
  the UI offer the switch at all). `start.sh` does the same by re-exec'ing
  itself. Without a launcher the route answers 409 and the UI says so.
  Untested on the PC: the actual restart cycle.
- **UX round (added 2026-09-03).** `POST /ui/rewind` truncates the session at
  the last user message and returns its content: Retry resends it, Edit puts it
  back in the composer (the plan and the read-before-edit record are kept - they
  describe the workspace, not the transcript). Text files dropped into the
  composer are read client-side and pasted into the prompt as fenced blocks
  (120k char cap); images still go to the vision tower, and a tall user bubble
  clips with a "show the whole message". The context meter is the last turn's
  `prompt_tokens` over `context_length` (amber 75%, red 90%). Sidebar rows gained
  rename (`POST /ui/sessions/<id>` with a title, which already existed) and
  Markdown export (client-side Blob download); the filter box is client-side over
  titles. `notifyIfAway()` flags the tab title always and fires a Notification
  only when the `notify` setting was turned on, which is also where permission is
  requested.
- **.env knobs:** `UI`, `UI_TITLE`, `FOLDER_PICKER`, `AGENT_WORKSPACE`, `WEB_TOOLS`,
  `AGENT_EXEC`, `AGENT_ASK`, `AGENT_MAX_STEPS`, `SEARCH_PROVIDER`, `SEARXNG_URL`,
  `TAVILY_API_KEY`, `TEMPERATURE`/`TOP_P`/`TOP_K`/`MAX_TOKENS`.
- **Web search is keyless DuckDuckGo (lite endpoint, scraped)** with `ddgs`
  used when installed, and SearXNG / Tavily preferred when configured. The
  scrape is the weak point: expect it to break eventually and say so in the
  error, which it does.
- **Verified** (sandbox, no GPU, no network): tool loop over the mock model,
  approvals allow/deny with real files written, cancel, session save/restore/
  delete, sandbox escapes refused, mode gating, plus Chromium runs of both
  modes - streaming, thinking, Markdown, tool cards, approval buttons, session
  restore, workspace picker, settings, dark and light. **Not yet run on the
  PC:** the aiohttp `mount()` path against the real model (only the stdlib
  adapter was exercised), and image attachments through the vision tower.
- Next: `start.bat`, then open `http://127.0.0.1:8888/`. If the UI does not
  appear, the server prints `!! built-in chat UI disabled: ...` right after
  Ready. `python tools/chatui.py --mock --open` isolates UI bugs from the
  model.

## Providers (added 2026-09-03)

`tools/webui_providers.py` + `providers.json` (gitignored, holds keys).

- A provider is `{id, name, base_url, api_key, models[], default_model}`.
  `public()` is the only thing the browser ever sees - `api_key` becomes
  `has_key` + `key_hint`; an empty key on save means "keep the stored one"
  (`_clean(row, existing)`). `local` is reserved for the kit's own server.
- `probe()` GETs `<base>/v1/models` with the key and returns the ids; the
  **user** chose this URL, so unlike `web_fetch` it may point at localhost or
  the LAN - that is the point of pointing at another local runtime.
- `resolve(root, pid, model, local)` decides where a turn goes.
  `ChatUI._client_for` caches a `ModelClient` per (base, model, key) and
  returns `self.client` for local - which is what the dev server replaces with
  the mock, so do not bypass it.
- The chat body carries `provider`/`model`; the session stores them, so a
  reopened conversation goes back to the endpoint that answered it, and the
  `session` event tells the UI which one that is.
- Routes: `GET/POST /ui/providers`, `POST /ui/providers/test`,
  `DELETE /ui/providers/<id>`; `GET /ui/models` now also returns `remote`.
- UI: Settings -> "Providers and endpoints" (list, add/edit with Test, remove);
  the Model picker groups "This computer" (restart) and "Providers" (instant);
  the sidebar shows a provider badge and hides the context meter for a remote
  model, whose window we do not know.
- Note: a remote model in Agent mode drives **local** tools. Approvals are what
  keeps that honest - the same guarantee as the local model, just worth
  remembering when the endpoint is someone else's.

## VRAM benchmark (tools/bench_vram.py, added 2026-09-03 - RUN on the 5090 2026-09-03)

The `QUANTS` table in `profiles.py` now carries **measured** weights and vision
sizes for 2.0 / 3.0 / 3.5 / 4.0 / 5.0 / 6.0 (2.5 is still an estimate - its
download is corrupt, `embed_tokens.weight` missing). See "Results" below. This
measures them on the real card:

- Every load happens in a **fresh child process** (`--child`, spec on stdin,
  JSON on stdout) - freeing an ExLlamaV3 model in-process leaves fragmentation
  that would corrupt the next number.
- Per quant: probe A at 32768 tokens (also loads the vision tower and diffs it),
  probe B at 65536 (the difference gives **measured KB/token**, replacing the
  assumed 18), then probe C at the largest planned context with a **real prefill
  of 90% of it**, stepping down 15% and 30% if it does not survive. Records
  `torch` allocated/reserved *and* the driver's per-process figure (what Task
  Manager shows and what actually has to fit), plus prefill and decode tok/s.
- `bench_vram.json` is appended after each quant (resumable, `--redo` to
  re-measure), `bench_vram.md` holds the table and the `QUANTS` rows to paste
  back into `profiles.py`.
- `--delete-after` removes each quant's weights once measured (keeping 2.0), so
  the whole sweep needs ~25 GB free instead of ~110 GB.
- `--sweep` additionally measures the **overhead at several context sizes**
  (`--sweep-points`, default 32k/64k/128k/192k/256k) and least-squares fits
  `overhead(ctx) = base + per_token * ctx`. One context per quant cannot separate
  the context-dependent part from quant-to-quant variation; this can.
- A dry run writes to `bench_vram.dryrun.json/.md`, never the real files.
- The child now applies `serve_openai._cap_process_vram(budget)` exactly as the
  server does. Without it `-gs` was only an autosplit hint and the child could
  use the whole physical card, so a "verified" context was not one the server
  could necessarily reach.

### Results (RTX 5090, 31.8 GB, driver 610.74, budget 29.3 GiB, cache `4`, MTP)

| bpw | weights GiB | vision GiB | KV KB/tok | verified context | peak GiB |
| --- | --- | --- | --- | --- | --- |
| 6.0 | 18.935 | 0.874 | 19.39 | none - see below | - |
| 5.0 | 16.123 | 0.877 | 18.85 | 183296 | 22.51 |
| 4.0 | 13.283 | 0.873 | 19.04 | 262144 | 21.29 |
| 3.5 | 11.864 | 0.872 | 18.98 | 262144 | 19.95 |
| 3.0 | 10.422 | 0.870 | 19.81 | 262144 | 18.09 |
| 2.5 | - | - | - | download corrupt | - |
| 2.0 | 7.082 | 0.173 | 19.07 | 262144 | 13.75 |

**KV_KB was right.** Measured rates are 18.85-19.81 KB/token *with MTP loaded*;
`profiles.py` applies the 17/16 factor separately, so the base rate is ~18.06 and
`KV_KB["4"] = 18` stands. An earlier reading of ~47 KB/token was a bug in the
bench itself (it subtracted `vision_gib` from a probe that never included it -
`kv_rate_kb()` now carries the explanation).

**OVERHEAD_GIB 1.7 -> 2.6.** Against real peaks the old figure under-predicted
every row (worst 0.73 GiB at 3.5bpw/262144), which is more than `MARGIN_GIB`
covers - a profile that "fit" could still run out. 2.6 bounds all five measured
rows. The per-token term (`OVERHEAD_KB_PER_TOKEN`, currently 0) waits on
`--sweep`.

**6.0bpw and the verified ceilings.** 6.0bpw loaded fine at both probe sizes but
no context survived the prefill, and 5.0bpw stopped at 183296. Both failures came
with the driver reporting **0 bytes free** while the process itself held only
~22.3-22.5 GiB of a 31.8 GiB card, and ~40 other PIDs holding the rest (their
per-process figures print as a garbled "17179869184.00 GiB" on this driver, but
the count is real). So these are contention with other GPU apps, not hardware
ceilings: `verified_ctx` is a **known-good floor recorded from a contended run**,
and a rerun on an idle card should raise it. Rerun with other GPU apps closed.
- Not yet run: it cannot be exercised anywhere but on the GPU box, so the first
  real run is also its first test. `--dry-run` exercises the plumbing only.

## Remote access (added 2026-09-03)

`UI_HOSTS` (comma separated) lists Host header values the UI answers to besides
localhost. It is deliberately *not* `UI_LAN`: a proxy such as `tailscale serve`
connects from loopback, so `_is_local` still passes and only the name needs
trusting - the port stays off the LAN and Tailscale does the identity. Tested
three ways: the named host passes, an unlisted name is refused, and a genuinely
remote peer is still refused even when it sends the right name.

## Conversation list: menu, pinning, content search (added 2026-09-03)

- `POST /ui/sessions/<id>` now takes `pinned` as well as `title`; rows carry
  `pinned` and sort `(not pinned, -updated)`.
- `GET /ui/sessions?q=` searches **titles and message text**. `_searchable()`
  flattens a conversation into one lowercase blob (skipping image data URLs,
  capped at `SEARCH_BLOB_CHARS`), `_index()` caches it against the file's mtime
  so a keystroke does not re-parse every session, and `_snippet()` returns the
  match in context. Rows gain `snippet` only when the title itself did not
  match.
- UI: a real context menu (`#menu-pop`, right-click or the button at the end of
  a row) with Rename / Pin / Save as Markdown / Delete; the row keeps only a pin
  marker. Search is now server-side and debounced, and the matched words are
  bolded inside the snippet. Groups are "Pinned", then the day labels, or
  "Results" while searching.

## Background turns (added 2026-09-03, supersedes the disconnect fix below)

DSH keeps working when its web UI is closed; so does this now. **The review fix
that cancelled a turn on disconnect was deliberately reversed** - only Stop
ends a turn.

- `Turn` (webui_app.py) is the unit of work: an append-only event log, a
  `threading.Condition`, `done`/`finished`. `chat()` no longer returns the
  generator - it starts a daemon thread that pumps the generator into a `Turn`
  and returns `Stream(turn.follow(0))`. HTTP responses are *views*; several can
  follow one turn (desktop and phone at once).
- `POST /ui/attach {session_id, from}` replays from an index and then follows
  live. `GET /ui/sessions` gained `running` per row and a `live` map;
  `_gc_turns()` keeps a finished turn replayable for `TURN_TTL` (900 s).
- A second turn in a busy conversation is **409** - they would share one
  `ToolContext`. The UI turns that into "watching it instead" and attaches.
- Both adapters now let a dropped connection be just that. `_drop_session`
  cancels and forgets the turn, so deleting a conversation still stops it.
- `APPROVAL_TIMEOUT` default raised 900 -> 1800 s: an approval should still be
  waiting when someone comes back to the tab.
- Client: `consumeStream()` and `finishTurn()` are shared by sending and
  reattaching; `detach()` (abort the fetch, leave the work) replaced `stop()`
  in newChat/openSession; boot looks for a live turn and opens it;
  `markResolved()` neutralises approval/question cards that were replayed after
  they had already been answered.

## Code review 2026-09-03 - fixed, and still open

Three reviews (security, concurrency, frontend/launcher). **Fixed and covered by
`tools/test_webui.py`:**

- **Origin/Host guard** (`ChatUI._origin_ok`). Loopback-only was not a boundary
  in a browser: DNS rebinding hands a hostile page the same peer address, and a
  cross-site POST needs no read access to do damage (it could have driven
  `/ui/chat` in agent mode and approved its own `run_command`). Now a non-local
  `Host`, a cross-site `Sec-Fetch-Site`, or a mutating request without either
  `Sec-Fetch-Site: same-origin` or the `X-Simplex-UI` marker is 403. The UI
  sends the marker on every request (`UI_HEADERS` in app.js); scripts must too.
  `headers=None` (in-process calls) skips the check.
- **web_fetch SSRF**: private/loopback/link-local/reserved addresses refused
  (`_public_host`), so the model cannot read `127.0.0.1:8888/ui/sessions` or the
  LAN. `WEB_FETCH_PRIVATE=1` opts back in.
- **`javascript:`/`data:` links** in model output rendered as inert text.
- **"Always allow"**: checked *before* the card is drawn (`pre_approved`), so an
  auto-approved call no longer shows a stale prompt or leaks a pending record;
  the grant is dropped when the workspace changes, since consent was for that
  folder.
- **Stop**: approval/question waits poll the cancel flag instead of sitting for
  15 minutes; `_active` is a counter, so a model switch cannot fire during a
  second concurrent turn. (Disconnect *used* to cancel too - reversed the same
  day, see "Background turns" below.)
- **Session writes**: per-session lock + unique temp name (a shared `<sid>.tmp`
  could publish a half-written file), and turns now **append** to the stored
  transcript instead of rewriting it from the model-facing copy - which was
  deleting every earlier turn's `reasoning_content` and clobbering renames.
- **Context meter**: prompt tokens are the max over steps, not the sum, so a
  5-step agent turn no longer reports ~100% used.
- **UI pump gets its own ThreadPoolExecutor** - it shared asyncio's default one
  with `generate_full`, which could deadlock both.
- **Jobs**: process-group kill (`_kill_tree`), stdout closed, ids from a
  counter; `DELETE /ui/sessions/<id>` now stops jobs and drops the context.
- **Model switch** refuses a folder with no weight files (an interrupted
  download would have re-downloaded the *previous* repo into it).
- Frontend: unterminated code fences render as code while streaming (they were
  reflowed prose for the whole stream); one render per animation frame instead
  of per token (67 KB answer was ~7 s of JS); rewind re-renders from the server
  instead of removing "the last two bubbles" (wrong once tools are in the
  transcript); leaving a chat stops its stream; `.env` sampling defaults are
  actually applied; filter debounced; malformed SSE frame skipped instead of
  killing the reply; `@@CB` placeholder collision; context meter reset per
  conversation; light-theme muted text raised to AA; modal is a real dialog and
  the thread is no longer an aria-live region.

**Known open (ranked, none of them silent-corruption class):**

1. A **failed model switch** has no rollback: `.env` already points at the new
   quant, so if `server_command` dies the launcher exits and every later
   `start.bat` fails until `.env` is hand-edited. Wants: keep the old settings,
   restore them if the child exits non-zero before Ready.
2. **Stop does not free the GPU.** Cancelling closes the HTTP connection but
   `generate_full` keeps generating to `max_tokens` while holding `gen_lock`.
   Needs a cancel flag threaded into the generator loop in `serve_openai.py`.
3. **Two concurrent turns in one session** share a `ToolContext` (the second
   turn's `ctx.workspace`/`ctx.ask` win). Serialise per session, or give each
   turn its own view.
4. A **restored** conversation still renders one assistant bubble per assistant
   message (live merges them), splits reasoning, puts the action row above the
   tool cards, and shows `ask_user` as a raw tool card.
5. `vram_preflight` can block a model switch on a console prompt nobody is
   watching; `HF_TOKEN` is not re-read after a switch; the Ready box prints the
   chat URL before `mount_ui` has had a chance to fail.
6. Markdown: nested lists flatten, a wrapped list item breaks the list.
7. Tall *image* messages are not clipped (height is measured before decode).

## One-click packaging round (added 2026-09-03, NOT yet run on the PC)

The user asked for the Windows experience to be as simple as it can be. Seven
things were built. All of them are new files plus edits to `win_start.py`,
`serve_openai.py`, `webui_app.py`, `start.bat` and `.env.example`.

### 1. Prebuilt wheels — `tools/wheels.py`, `wheels/`

Compiling the engine was the step most likely to fail and the reason the kit
demanded the CUDA Toolkit and VS Build Tools. `install_environment()` now tries,
in order: an exact `.whl` in `wheels/` → `WHEEL_INDEX` (`pip --find-links`) →
PyPI (for `triton-windows`) → source build. `wheel_matches()` compares Python,
ABI and platform tags against **the venv's** interpreter (asked with a
subprocess, never guessed from the launcher's own Python), so a cp313 wheel can
never be installed into cp312. `abi3` is treated as a floor, not an equality.
`source_build_blockers()` turns "no compiler" into a sentence with a fix in it.
`wheels/README.md` documents building one.

### 2. First run in the browser — `tools/setup_core.py`, `tools/setup_web.py`, `tools/webui/setup.{html,js}`

`setup_core` is the pipeline with no console attached: `probe()`, `options_for()`
(the same `profiles.plan()` numbers the console menu used), `apply_choice()`,
`install_environment()`, `download_weights()`, `needs_setup()`. `setup_web` is a
stdlib HTTP server on the *same port the chat UI will use*, with an append-only
event log and `follow()` over a `threading.Condition` — the same pattern as
`Turn` in `webui_app.py`, so a reload or a second tab replays rather than
guesses. Phases: `probe → choose → install → done | error | cancelled`.

Guards match the chat UI: loopback bind, `Host` check, `X-Simplex-UI: 1` on
every POST.

`win_start.main()` calls it when `needs_setup()` says so or on `start.bat setup`
/ `profile`. The outcome is a tuple: only `"unavailable"` (the page could not
open at all) falls back to the console questions — `"cancelled"` and `"error"`
exit with a message, because re-asking what the user just answered on the page
is worse than stopping.

`SETUP=console` in `.env` keeps the old flow.

### 3. Resumable download — `tools/downloader.py`

Standard library only, so it runs **before** the venv exists, which is what lets
the weights come down while pip is still working. Lists the repo with the HF
tree API, streams each file to `<name>.part` with a `<name>.part.json` sidecar
recording size+oid; a `.part` whose sidecar does not match the file on the
server is discarded rather than appended to. Resumes with `Range`, handles a
server that ignores `Range`, verifies size and (for LFS files) sha256, deletes a
file that fails its checksum. `download_model()` in `win_start.py` now uses it
too, so the console path gets a percentage and an ETA instead of a still cursor.

Byte accounting on resume is the subtle part: `_fetch` tracks `counted` so a
retry that resumes the same prefix cannot count it twice. There is a test for it.

### 4. Tray icon — `tools/tray.py`

Pure ctypes Win32 (`Shell_NotifyIconW` + a hidden window + a message loop on its
own thread). No pystray, no extra dependency. Menu: Open / Restart the model /
Show the folder / View the log / Quit. Re-adds itself on `TaskbarCreated` (an
Explorer restart). Balloon when the model finishes loading. `Tray.available()`
is False off Windows and `start()` returns False rather than raising, so the
launcher calls it unconditionally. `TRAY=no` disables it.

`Runtime` in `win_start.py` is what the menu acts on; the run loop checks
`rt.quit` / `rt.restart` after the child exits.

### 5. Shortcuts — `tools/shortcuts.py`

Start-menu and desktop `.lnk` via `WScript.Shell` from a PowerShell script fed
on **stdin** (nothing interpolated into a command line). Created once after the
first good launch, remembered in `.simplex/shortcuts.json`. `SHORTCUTS=no` to
skip, `yes` to enforce. `autostart()` uses a scheduled task with `/RL LIMITED`
(no UAC at sign-in) — deliberately not a service, which cannot reach the GPU
the way a user session does.

### 6. Installer and manifests — `packaging/`

`simplex.iss` (Inno Setup 6), per-user into `%LOCALAPPDATA%\Programs\Simplex`
with `PrivilegesRequired=lowest`, because the kit writes into its own folder
forever (venv, 10 GB of weights, sessions) and Program Files would mean either
UAC on every launch or a program that cannot write to itself. Excludes
everything the kit creates. Uninstall asks before deleting the model and the
conversations. `build.ps1` builds the installer and the zip and prints SHA-256s.
winget (3 YAMLs) and Scoop (1 JSON) manifests are templates with `CHANGEME`
where a real publisher/repo has to go; `packaging/README.md` explains signing
and why there is deliberately no single-file `Simplex.exe`.

### 7. Installable web app — `tools/webui/{manifest.webmanifest,sw.js,icon-*.png}`

Served from the **root** (`PWA_FILES` in `webui_app.py`): a service worker can
only control paths below its own URL. The worker caches the shell only —
`/`, `style.css`, `app.js`, the icons — network-first, and never touches
`/ui/*`, `/v1`, `/health` or the event streams, because a cached answer there
would be a stale conversation or a stalled stream. Registration is guarded by
`isSecureContext`, so over a plain-http LAN or Tailscale address it simply does
not register and the UI stays a normal page. An **Install** button in the topbar
appears only when `beforeinstallprompt` fires.

Icons are generated by `tools/make_icons.py` (Pillow, run once, results
committed) — PNG 192/512/maskable/apple-touch plus `tools/simplex.ico` for the
tray, the shortcuts and the installer.

### 8. Console honesty — `tools/logbook.py`, `serve_openai.py`, `start.bat`

* **The Ready box used to lie.** It printed before `mount_ui()` ran and before
  the port was bound, so it could announce a chat address that did not exist and
  then print "built-in chat UI disabled" underneath. `main()` now builds the app,
  mounts the UI (which returns a bool), starts an `AppRunner`/`TCPSite` — that is
  where the bind actually happens — and only then prints, with the chat line
  reflecting what really mounted. A port clash is now a sentence, not a traceback.
  **This path could not be exercised in the container (no aiohttp): it needs one
  real run on the PC.**
* `run_app` was replaced by an explicit runner, so Ctrl+C on Windows is handled
  by short `asyncio.sleep(0.5)` ticks rather than aiohttp's own loop.
* `logbook.Logbook` tees stdout/stderr into `logs/simplex-<stamp>.log` (ANSI
  stripped in the file, kept on the console), keeps the last 10, and the child
  server's output is piped through the launcher so it lands there too — which is
  what makes the tray's "View the log" worth having.
* Any exception that reaches the top of `win_start` now prints one sentence about
  what happened and one about what to do (`logbook.explain()`), with the traceback
  going to the log. `die()` names the log file.
* `start.bat` prefers the kit's own `.venv\Scripts\python.exe`, explains a
  missing Python in four numbered steps, and pauses on any non-zero exit.

### Tests

`tools/test_webui.py` grew ~60 checks: wheel tag matching and selection, the
downloader against a fake HF server (full download, resume with `Range`, stale
`.part` discarded, no re-fetch, bad checksum deleted, cancel), `setup_core`
(steps, menu, `.env` writing, bad option refused), `setup_web` (guards, the
whole state machine to `done`, event numbering, path traversal, `/health` 503),
`logbook`, `shortcuts`, the PWA files and routes, and an AST-ish check that
Ready comes after the mount and the bind. `tools/setup_mock.py` fakes the GPU,
the installer and the download so the page can be driven without either.

### Still to verify on the PC

* One real `start.bat` run: the new Ready path (aiohttp), the tray icon, the
  shortcuts, and a real HF download through `tools/downloader.py`.
* The tray's ctypes code has never executed — it is guarded everywhere, but
  "the icon appears and the menu works" is unproven.
* `start.sh` (Linux/Spark) was **not** given the browser setup; it still uses the
  console path. That is fine but is now an inconsistency worth knowing about.

## Code review of the packaging round, and the fixes (2026-09-03)

Three reviewers (correctness/concurrency, security, Windows/integration) over
everything added above. ~32 raw findings; these were fixed in one pass, each with
a regression test in `tools/test_webui.py`. Suite is now **281 checks, ALL PASS**,
idempotent across runs.

### Security

* **Path traversal in the downloader.** `target = self.dest / f.path` where
  `f.path` came straight from the HF tree JSON. `../../x` escaped, and on Windows
  `C:/Users/Public/x.bat` discarded `dest` entirely - a Startup-folder write, i.e.
  code execution, from a malicious or typosquatted repo. Fixed with
  `safe_relpath()` (rejects absolute, drive letters, backslashes, `.`/`..`/empty
  segments, control chars, names Windows cannot represent) applied in
  `list_files`, plus `_contained()` re-checked at both write sites.
* **HF token forwarded across redirects.** `urlopen` strips only content headers,
  so `Authorization` followed every 302 - and HF redirects each file to a CDN on
  another host, so this fired on the happy path. Fixed with `_SafeRedirects`,
  which drops the header when the hostname changes and refuses non-http schemes.
  All requests now go through a module-level `_OPENER`.
* **`.env` injection.** `set_token` did `.strip()`, which does not remove interior
  newlines, and `.env` is parsed line by line - `"hf_x\nWHEEL_INDEX=http://attacker/"`
  became a second setting that made pip install wheels from an attacker's index.
  Fixed in two places: a token regex in `setup_web.set_token`, and
  `profiles._env_value()` stripping CR/LF/NUL/control chars and anything after a
  `#` for **every** caller - which also covers `PROFILE_GPU`, whose value comes
  from `nvidia-smi`.
* **No peer check on the setup server.** It matched `webui_app`'s Host and
  `X-Simplex-UI` guards but not its `_is_local(remote)` one, so `curl -H 'Host:
  127.0.0.1' -H 'X-Simplex-UI: 1'` from the LAN could rewrite `.env` and start
  pip. Latent (the bind is loopback) but now closed with `_peer_ok()`.
* **`shortcuts._q` escaped one quote character; PowerShell has five.** U+2018,
  U+2019, U+201A and U+201B also delimit a verbatim string, and these literals
  carry the user's account name and install path - `O\u2019Brien` was injection
  into a script run with `-ExecutionPolicy Bypass`.

### Correctness

* **The setup page froze at 4000 events.** I copied `Turn.follow`'s cursor-as-list-
  index pattern but added `del self.events[:-MAX_LOG]`, which the original does
  not have. Once trimming started the list stopped growing, every event got the
  same `i`, and the live log, checklist and progress bar died - on exactly the
  slow source-build install this was for. Fixed with a `_dropped` counter so `i`
  is absolute and `follow()` maps a cursor to a position; a cursor pointing at
  trimmed events resumes at the oldest still held.
* **The error screen's buttons were dead.** `finished` was a one-way latch and
  `run()` shut the server down 1.5 s after a failure. `set_phase` now clears it on
  any non-terminal phase, and `run()` holds the page open for `RETRY_GRACE`
  (10 min) after an error, printing a console line saying so. Cancelled still
  exits promptly. Also found and fixed while testing this: `#go` disables itself
  on click and was never re-enabled, so coming back to the menu left the only
  button on the page dead.
* **Three download-resume defects.** `counted` was only updated on a *successful*
  `_stream`, so a mid-file failure made the retry count the same prefix again -
  the bar leapt, speed read "8 GB/s", and it pinned at 100% with ten minutes to
  go. It is now always read back off the disk. A `.part` that was exactly complete
  (crash between the last read and the rename) was discarded and re-fetched - now
  kept and verified. And `_open` turned every `HTTPError` into a `DownloadError`
  that the retry arm could not catch, so one CDN 503 ended setup; errors now carry
  a `retryable` flag.
* **A failed download left `progress.state` non-terminal**, so `setup_core`'s
  cancel watcher polled forever. `run()` now guarantees a terminal state before
  any exception leaves.

### Windows

* **The child's `\r` progress bar became thousands of console lines.** Piping with
  `text=True` puts the pipe in universal-newline mode where a bare CR ends a line.
  `_pump_child` now reads bytes and decodes incrementally, so the bar redraws in
  place; `logbook.for_file()` collapses CR redraws for the file copy only.
  `PYTHONIOENCODING`/`PYTHONUTF8` are set on the child so a pipe does not drop it
  to the ANSI code page.
* **A dead `.venv` was an unrecoverable loop.** `start.bat` preferred
  `.venv\Scripts\python.exe` with no liveness check; after a base-Python upgrade
  it dies instantly, `win_start.py` never runs, and the code that would rebuild
  the venv is unreachable. `start.bat` now probes it, `install_environment`
  rebuilds a venv that cannot run, and `_system_python()` refuses to pick an
  interpreter inside `.venv` (the launcher puts it on PATH).
* **The pump thread was never joined**, so `proc.wait()` could return before the
  child's last words - including the new "could not listen on port" message -
  reached the console or the log. Now joined, with a timeout, on every path.
* **A second Ctrl+C orphaned the server.** `except Exception` does not catch
  `KeyboardInterrupt`, so `terminate()` was skipped and a child kept the GPU and
  port 8888. Now `except BaseException`.
* **`_port_free` could not detect a busy port on Windows** (SO_REUSEADDR has
  bind-over-listener semantics there), so the friendly message was unreachable.
  Now a connect probe first, then `SO_EXCLUSIVEADDRUSE` on Windows.
* **Shortcuts were created before the server had ever run**, so a kit that died on
  the model load still left an icon; and they overrode the desktop-icon choice an
  installer user had made. Now created from the `/health` ready callback, skipped
  when `looks_installed()` sees an `unins*.exe`, and never overwriting an existing
  `.lnk`.
* **Tray**: handles were truncated (`GetModuleHandleW` had no `restype` - it worked
  only because Register and Create agreed on the same wrong value); `notify()`
  parked `uFlags` at `NIF_INFO`, so a concurrent `set_tip` or an Explorer restart
  could register a blank, unclickable icon - all `_nid` access is now under a lock;
  `stop()` did not wait, leaving a ghost icon - it now waits for `NIM_DELETE`; a
  real `RegisterClassW` failure is no longer mistaken for "already registered";
  `rt.url` is refreshed when a model switch changes `PORT`.
* **`build.ps1 -Version` was ignored for the installer** (`#define` overrode `/D`)
  - now `#ifndef`. `ArchitecturesAllowed=x64compatible` needs Inno 6.3+, so it is
  `x64`. `ExtraDiskSpaceRequired` declares the real 10 GB. Uninstall now reads
  `MODEL_DIR` out of `.env` before deleting it, so weights outside `{app}` are
  found. `.github` and `bench_vram.json` added to the excludes.
* **`logbook.start()` twice nested a Tee**, restoring the inner one on stop and
  leaking a file handle that on Windows keeps the log locked - which then made
  `_prune` abort its whole sweep on the first `PermissionError`. The guard is now
  module-global, and prune skips a locked file instead of giving up.

Still unverified: everything Windows-only still needs one real run (the tray, the
aiohttp Ready path, shortcuts, a real HF download). One reviewer claimed `stop.bat`
is missing - it is not, it just is not in the container's copy of the tree.

## Windows status (user’s PC)

Setup got through: venv, pip, torch cu130, ExLlama compile, **triton-windows**, Hub download. Then load started (`Loading (LS) ~1%`) with Triton `remark:` spam.

**Not confirmed in this session:** a full Windows boot to `Ready` + a real Cherry chat. Next session should verify that.

## Bugs already fixed (must be in the Windows copies)

1. `python -m pip` not `pip.exe` (Windows cannot self-upgrade pip).
2. Ignore unusable `EXL3_REPO` (Spark path `/home/zurih/NewModels/exllamav3-1.4.4-aarch64`). Engine: `git+https://github.com/turboderp-org/exllamav3.git@v1.4.4`.
3. **`ensure_triton()`** installs `triton-windows`. Without it: `cannot import name '_dsa_attn_split_kernel'` from `dsa_triton`. `ensure_triton` and `find_cuda_home` must stay separate functions (`cfg` was undefined when they were accidentally merged).
4. ASCII banners only.
5. **Ready must not print before load.** `win_start.py` now says “loading… do not connect yet”. Ready box is printed by `serve_openai.py` **after** `build_model()`.
6. Triton compiler remarks filtered in `serve_openai.py` `_quiet_triton()` (stderr pipe; one line `compiling Triton kernels...` instead of hundreds of `remark:` lines).

## `.env` on Windows

First `start.bat` copies `.env.example` → `.env`. **Comment out / delete `EXL3_REPO=`** if it still points at a Linux path. Do not copy Spark `.env` as-is (`EXL3_REPO=/home/zurih/...`).

Needs: 64-bit Python 3.11+, Git, NVIDIA driver, CUDA Toolkit (`nvcc`; user had `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0`), VS Build Tools C++ workload.

**VRAM pre-flight (added 2026-09-02):** `win_start.vram_preflight()` runs right before the server launch: `nvidia-smi` used/free/total; if free < GPU_MEM_GB + 0.3 GiB it lists VRAM holders via the Windows perf counter `\GPU Process Memory(*)\Dedicated Usage` (what Task Manager shows; nvidia-smi on Windows omits graphics apps) and prompts Enter/c/q (120 s auto-continue for unattended runs). Note the 12.35/13.37 GiB load numbers came from the Spark under the cap, not from a 16 GB Windows card - the first Windows run is still the real test; on a short card set GPU_MEM_GB=14.3 / CONTEXT_SIZE=180224. NVIDIA Control Panel -> CUDA Sysmem Fallback Policy -> "Prefer No Sysmem Fallback" makes a short budget fail fast instead of crawling.

**fp8 / nvfp4 KV (added 2026-09-02, NOT yet run on the PC):** `patches/kvcache-fp8-nvfp4-v1.4.4.patch` = the MiaAI-Lab fork's KV lanes (its base is v1.4.2) re-based onto v1.4.4: `cache/fp8.py`, `cache/nvfp4.py` (new), `cache/__init__.py`, `attention_fn/common.py` (k/v scale fields), `attention_fn/dispatch.py` (fp8/nvfp4 fast path), `attention_fn/triton_paged.py` (online-dequant kernels), `model_init.py` (`-cq fp8|nvfp4`; the MTP draft cache stays on the stock int 8,4 path, unlike the fork which keeps it fp16). Verified: the fork's attention files are byte-identical between upstream 1.4.2 and 1.4.4, `git apply --check` passes on a clean v1.4.4 checkout and on a site-packages-style tree. `tools/patch_kv.py apply|revert|status` patches the venv in place by copying the complete patched files from `patches/kvcache-fp8-nvfp4-v1.4.4/` after a CR-normalised SHA-256 check against stock (manifest.json; backup in `exllamav3/.kvpatch-backup/`). **Do not use `git apply` for this**: the first Windows run did, and because the kit lives inside a git work tree git silently ignored every path and reported success (`already applied`, nothing installed). The Windows checkout is CRLF, so hashes are compared with CR stripped and patched files are written CRLF to match; `win_start.py` `ensure_kv_patch()` / `start.sh` call it when `CACHE_QUANT` is fp8/nvfp4, and validate the value otherwise. `HF_REVISION` knob added to both launchers (branch download).
- CPU format simulation `tools/kv_format_sim.py` (numpy; re-implements q_cache_kernels.cuh int path incl. H32 rotation + midpoint grid, and the fork's E4M3 / E2M1 quantizers): on Qwen-like K/V, int4 beat nvfp4 (attn-weight KL 2.2e-2 vs 1.3e-1, top-1 96.3% vs 88.7%) and int8 beat fp8 by a lot (7e-5 vs 5.8e-2) - fp8 has no scales, so its ~2.6% multiplicative error lands on the big key channels. On pure Gaussian data int4 ~= nvfp4. Synthetic, so indicative only.
- **GPU results (RTX 5090, sm_120, 2026-09-02, 144 teacher-forced probes @16k ctx, KL vs fp16 cache):** `8`: mean 1.0e-4 / p95 4.9e-4 / max 1.5e-3, top-1 100%. `8,4`: 5.4e-4 / 2.8e-3 / 6.6e-3, 100%. `fp8`: 4.6e-4 / 3.0e-3 / 9.3e-3, 100%. `4`: 1.2e-3 / 6.3e-3 / 1.1e-2, 98.6%. `nvfp4`: 1.45e-3 / 7.2e-3 / 2.9e-2, 97.2%, ref-acc 88.9% (all others 90.3%). All formats retrieved a 63k-token passkey; decode 69-73 tok/s for every format (no MTP in the test); VRAM identical per bit-width. **Verdict: stock int wins** - int `4` beats nvfp4 (20% lower mean KL, 2.5x lower tail), fp8 lands between int8 and int8,4 with a worse tail and no size advantage. Every KV format's KL is ~300x smaller than the 2.0 bpw weight quant's own KL (0.35), so the 150k profile with `CACHE_QUANT=4` is essentially free. fp8/nvfp4 stay opt-in; no reason to default to them. The patch and tests are kept for future engines. NB: the dev box is a 32 GB RTX 5090 - the 16 GB recipe is for the kit's users, the dev box can run 3.0-4.0 bpw at full context.
- GPU tests to run on the PC: `test_kv.bat` -> `tools/kvtests/{fp8,nvfp4}cache_test.py` (fork's kernel parity, synthetic) then `tools/kv_cache_tests.py` (one load per format: KL vs fp16 at 16k on qbench cut points, top-1, ref-acc, 64k passkey, tok/s, VRAM; writes `kv_cache_tests.json`). Needs cc 8.9+ for the Triton fp8 conversions. Expect ~20-40 min; `--quick` ~10. Decision rule: lowest KL that passes the passkey; int `4` needs no patch.

**GPU profiles (added 2026-09-02, NOT yet run on the PC):** `tools/profiles.py` - nvidia-smi VRAM -> budget (total - max(1.3, 8%)) -> planner over the quant table (gpu_gib **measured on the 5090 2026-09-03**: 2.0 7.082 / 3.0 10.422 / 3.5 11.864 / 4.0 13.283 / 5.0 16.123 / 6.0 18.935; 2.5 still the 8.3 estimate - corrupt download. Vision measured 0.87, or 0.173 for the 3-bit-quantised 2.0 tower), need = weights + KV*17/16 + vision + overhead (2.6 measured, was 1.7), 0.4 GiB safety. Rows also carry `measured` and `verified_ctx`; the menu marks anything unverified and the recommendation only ever points at a row that is both. Options: quality (highest bpw with >=128k), balanced, context (262k); KV cache always int `4` (user decision 2026-09-02: auto-select, never shown; int4 measured KL 0.0012 vs fp16). Menu is a fixed-width table (no cache column). Writes PROFILE/MODEL_DIR/HF_TARGET_REPO/HF_REVISION/MODEL_ID/CONTEXT_SIZE/CACHE_QUANT/GPU_MEM_GB/VISION into .env in place. Triggers: no PROFILE line, PROFILE=ask, `start.bat profile` (start.bat now passes %*), `./start.sh profile`. "Keep current" (option 0) is the timeout default when MODEL_DIR has weights. **The existing Windows .env has no PROFILE line -> the next start.bat will show the menu once** (Enter = keep current). `serve_openai.py --model_id` (default: model folder name lower-cased) replaces the hard-coded MODEL_ID; `cherry.kit_model_id(cfg)` derives the same id and `configure_db` retargets Cherry's default model + assistants when the id changes (tested). Non-2.0 branches are turboderp's plain quants: head_bits 6, vision bf16, no SC trace - fine on v1.4.4. Weight sizes for 2.5-6.0 are estimates: the first load of each is the check (VRAM pre-flight + Ready box).
- **GPU capability gating:** `profiles.detect_gpu()` reads name/memory/compute_cap/driver_version; `gpu_support()`: cc < 7.5 -> unsupported (no kernels in cu128/cu130 wheels), 7.5-7.9 -> "slow" warning, < 8.9 -> fp8/nvfp4 unavailable note, driver < 580 -> cu128 (>= 570) or cu126 wheels via `win_start.torch_index_for_driver()` (TORCH_INDEX_URL still wins). `win_start` dies on cc < 7.5 and on CACHE_QUANT=fp8|nvfp4 with cc < 8.9. The stock integer KV formats (8 / 8,4 / 4) have no generation requirement (plain CUDA integer kernels, verified in q_cache_kernels.cuh: only a < 7.5 tanh fallback and an >= 8.0 fast path in routing.cu). `python tools/profiles.py --list --vram 16 --cc 7.5` simulates a card.

## How to run

1. Double-click `start.bat`. Later runs skip install.
2. Wait for the **Ready** box. First kernel compile is slow; later runs use Triton’s cache.
3. Answer `y` to the Cherry Studio prompt (or `CHERRY_AUTOSTART=yes`). `stop.bat` stops the server only; Cherry stays open (closing its window hides it to the tray by default).

## Constraints / don’ts

- Do not raise context to 262144 on 16 GB.
- `CACHE_QUANT`: numeric on the stock engine; `fp8`/`nvfp4` only after `tools/patch_kv.py apply` (the launchers do it). Unpatched engine + fp8/nvfp4 aborts the cache loader.
- `cal_trace.safetensors` is for **rebuild only**, not serving.
- Spark-only: `install_engine.sh`, `patches/`, `EXL3_REPO` aarch64 checkout. Windows does not use those.
- `CPU_CACHE_GB` is accepted but inert.
- Host in example is `0.0.0.0` (LAN, no auth). Fine for home; not for the public internet.

## Likely next work on Windows

1. Confirm the two Python files above are present, then `start.bat`.
2. Confirm Ready appears **after** load, remarks are one line, Cherry can chat.
3. If remarks still flood: they may be writing stdout; extend the filter.
4. If load OOM: check `GPU_MEM_GB=14.7` and that the VRAM cap still pins after ExLlama autosplit.
5. **Run the Cherry integration end to end** (download → init run → prompt → chat with tools). Check `.venv\Scripts\python.exe tools\cherry.py status` and `apps\cherry-studio\data\logs\` if it misbehaves.
6. Optional: Cherry MCP web search (client-side; server stays local).

## Key files

```
start.bat, stop.bat
tools/win_start.py      # Windows bootstrap, run loop, tray/shortcut wiring
tools/setup_core.py     # first-run pipeline with no console attached
tools/setup_web.py      # the first-run page's server (stdlib, same port as the UI)
tools/setup_mock.py     # fake GPU/installer/download for driving that page
tools/downloader.py     # resumable HF download, stdlib only, runs before the venv
tools/wheels.py         # prebuilt-wheel discovery and tag matching
tools/logbook.py        # tee stdout/stderr to logs/, plain-English error text
tools/tray.py           # Win32 notification-area icon (ctypes)
tools/shortcuts.py      # Start-menu/desktop .lnk, start-at-login task
tools/make_icons.py     # regenerates the PNG/ICO icon set (needs Pillow)
tools/webui/setup.html  # the first-run page
tools/webui/sw.js       # service worker: shell only, never the API
packaging/              # Inno Setup script, build.ps1, winget + Scoop manifests
wheels/                 # drop .whl here to skip compiling the engine
tools/serve_openai.py   # OpenAI server + VRAM cap + Triton quiet + Ready box
tools/profiles.py       # GPU-aware quant/context profile picker (writes .env)
tools/cherry.py         # Cherry Studio: download, first-run init, sqlite config, open
tools/patch_kv.py       # hot-patch fp8/nvfp4 KV lanes into the venv's exllamav3 1.4.4
tools/kv_cache_tests.py # GPU: KL/passkey/speed per cache format (test_kv.bat)
tools/kv_format_sim.py  # CPU: format-level error simulation (numpy)
tools/kvtests/          # fork's Triton kernel parity tests
patches/kvcache-fp8-nvfp4-v1.4.4.patch
.env.example            # 16 GB defaults + CHERRY_* knobs
models/                 # gitignored weights
apps/cherry-studio/     # gitignored: portable exe, data/, kit-config.json
```

Do not treat the Linux Spark workspace as the Windows source of truth for `.venv`, `.env`, or `models/`. Sync kit scripts; keep machine-local dirs on Windows.
