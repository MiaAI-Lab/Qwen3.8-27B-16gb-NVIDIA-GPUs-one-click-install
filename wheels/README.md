# Prebuilt wheels

**You probably do not need to put anything here.** The engine's own GitHub
release publishes a wheel for every (CUDA line x torch version x Python)
combination it supports, and the launcher installs the one matching this
machine's venv. A first run is a download, not a compile.

This folder is the override, for the cases where that does not apply:

- no network, or a network that cannot reach github.com - copy a wheel here
  off a USB stick;
- a CUDA line the engine has no build for (it builds cu128 and cu132);
- a platform it has no build for, such as aarch64 / GB10;
- a wheel you built yourself, which always wins over the release.

## What to put here

A wheel is only used when its tags match the Python that Simplex is installing
into - so a `cp313` wheel is ignored by a `cp312` environment, and nothing can
be installed into the wrong interpreter by accident. Check what would be used:

    .venv\Scripts\python.exe tools\wheels.py

That prints the venv's tags, the torch version and CUDA line it found, the
release wheel it would pull, and anything already sitting in this folder.

| package          | why |
|------------------|-----|
| `exllamav3`      | the engine. Building it is the slow, fragile step. |
| `triton-windows` | GPU kernels the engine imports. Normally comes from PyPI. |

## Building the engine wheel yourself

On a machine that *does* have the CUDA Toolkit and the Visual Studio C++
workload, once per Python version:

    py -3.13 -m venv build-env
    build-env\Scripts\python -m pip install -U pip wheel setuptools ninja
    build-env\Scripts\python -m pip install torch --extra-index-url https://download.pytorch.org/whl/cu128
    build-env\Scripts\python -m pip wheel --no-build-isolation --no-deps ^
        git+https://github.com/turboderp-org/exllamav3.git@v1.4.4 -w wheels

The wheel is tied to the torch build the line above installed, so build one per
CUDA line you intend to support.

## Hosting them instead

Set `WHEEL_INDEX` in `.env` to somewhere pip can reach with `--find-links`:
a GitHub Releases page, a file share, an internal index. Several are allowed,
separated by spaces or commas.

Order of preference: this folder, then the engine's release, then
`WHEEL_INDEX`, then a source build. A USB stick beats the network.
