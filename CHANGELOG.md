# Changelog

Notable changes to this kit. Newest first.

## Unreleased

### Fixed
- Console setup (`SETUP=console`) installs the prebuilt engine wheel before
  compiling, like the browser setup; fixes a failed compile on machines whose only
  CUDA toolkit is newer than torch's CUDA line (#3).
- `linux/start.sh` no longer crashes with an arithmetic error under a non-English
  locale: `free` is read by row position under `LC_ALL=C` (#4).
- `CUDA_HOME` is detected (PATH, `/usr/local/cuda`, `/opt/cuda`, versioned
  installs) instead of assumed to be `/usr/local/cuda` (#5).
- Source builds stop early, with the `CC=gcc-13 CXX=g++-13` hint, when the host
  compiler is GCC 14+ and CUDA is 12.x; `SKIP_COMPILER_CHECK=1` disables it (#6).
- A model folder missing `tokenizer.json` is no longer reported complete, so an
  interrupted download is resumed instead of failing at start (#8).
- `max_tokens` is clamped to what the KV cache can admit, so a client's large
  `max_tokens` on a long prompt no longer fails with "Job requires N pages (only M
  available)"; a prompt that cannot fit is a clear error (#11).
- Requests that omit `max_tokens` now default to 32768 instead of 1024, which cut
  off long summaries with `finish_reason=length` (#14).
- A `#` inside a quoted `.env` value is kept by the Windows reader, the profile
  writer and the harness launcher, matching `linux/start.sh`. Thanks to Sasha
  Mitchell (PR #12).
- An error raised by the engine while generating (for example out of memory in a
  long prefill) is no longer answered with HTTP 200, `finish_reason=length` and an
  empty reply. A non-streaming request now gets HTTP 500 with the engine's message,
  and the failed job's references are released (`gc.collect()`,
  `torch.cuda.empty_cache()`).

### Added
- `DRAFT=dflash2` on 16 GB cards, Linux only, with the kit's 2.5 bpw quant: context
  90112, `GPU_MEM_GB=15.4`, 4-bit KV cache, and `EXL3_QC_STAGING=0` in the server's
  environment (set by `linux/start.sh`). Measured on one RTX 5060 Ti; any other
  quant, and a 16 GB card under Windows, is refused. The 32 GB row is unchanged.
  Numbers and limits are in the README ("DRAFT=dflash2").
- Per-request stats: one ` == stats` line per request, live ` .. ` progress lines
  (`PROGRESS_EVERY`), and `last_request` in `/health`. Thanks to Ivan Ribeiro Rocha
  (PR #7).
- `tests/` (stdlib `unittest` and one bash script) for the `.env` readers, the
  token budget, the downloader's completeness check and the Linux pre-flight
  helpers.

### Changed
- **ExLlamaV3 1.6.0 is now the default engine for every `DRAFT` value**
  (`mtp`, `none`, `dflash2`), replacing 1.4.4. All values share one `.venv`.
  - `tools/wheels.py`: the 1.6.0 release table is the default; the 1.4.4 table is
    kept as legacy for `wheels.py --engine-version 1.4.4`. The find-links route
    always pins `exllamav3==<engine>`.
  - `linux/start.sh`, `tools/win_start.py`, `tools/setup_core.py`: pin v1.6.0 and
    check for exactly that version.
  - `transformers` is always installed. 1.6.0's chat template imports it and the
    engine no longer pulls it in; without it every chat request returned HTTP 500.
  - `DRAFT=dflash2` keeps its 32 GB gate and measured settings, but no longer
    builds a separate `.venv-dflash2`.
  - Windows model switch only re-applies the card gate; the quant check and the
    drafter fetch happen once, in `server_command`.
- README benchmark tables are labelled against the previous default (1.4.4 + MTP);
  the depth table shows 1.6.0 + MTP as the current default. dflash2 needs about
  2.4 GiB more VRAM than the default.

### Upgrading
Delete the old `.venv` folder (and `.venv-dflash2` if present) and start again. The
first run reinstalls the engine and `transformers`. Models in `models/` are kept.
Do not set `EXL3_REPO` to work around the version error: it skips the prebuilt wheel
and forces a source compile.

### Not validated
- `DRAFT=dflash2` on 16 GB was measured on one RTX 5060 Ti on Linux. Its MTP
  comparison comes from a separate server start earlier the same day, there is no
  MTP quality run at that context, and the 500 response above was exercised in an
  earlier build (a 68013-token prompt), not in the final configuration. In an
  alternating check, 2 of 15 dflash2 code runs fell into a repetition loop at
  temperature 0; the cause is not known.
- The 12 / 16 / 24 GB profiles were tuned on 1.4.4 and have not been re-measured on
  1.6.0.
- The Windows launcher has not been run on 1.6.0.
- `transformers` is installed unpinned.

## 2026-10-09

### Added
- Opt-in `DRAFT=dflash2` (ExLlamaV3 1.6.0 + DFlash2 drafter) for 32 GB cards,
  in its own `.venv-dflash2` (PR #15).

## 2026-09-10

### Fixed
- Harness: `DSH_PORT` in `.env` was fatal to the current `dsh`; renamed and spawned
  from `.dsh` (PR #2).
