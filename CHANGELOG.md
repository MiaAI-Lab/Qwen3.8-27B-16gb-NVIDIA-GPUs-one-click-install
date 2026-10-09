# Changelog

Notable changes to this kit. Newest first.

## Unreleased

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
