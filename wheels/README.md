# Prebuilt wheels

Drop `.whl` files here and Simplex installs them instead of compiling.

This is the difference between a first run that needs the NVIDIA CUDA Toolkit
and Visual Studio Build Tools (several GB, twenty minutes, and the step most
likely to fail) and one that is a plain download.

## What to put here

A wheel is only used when its tags match the Python that Simplex is installing
into - so a `cp313` wheel is ignored by a `cp312` environment, and nothing can
be installed into the wrong interpreter by accident. Check what would be used:

    .venv\Scripts\python.exe tools\wheels.py --package exllamav3

Useful wheels:

| package          | why |
|------------------|-----|
| `exllamav3`      | the engine. Building it is the slow, fragile step. |
| `triton-windows` | GPU kernels the engine imports. Normally comes from PyPI. |

## Building the engine wheel yourself

On a machine that *does* have the CUDA Toolkit and the Visual Studio C++
workload, once per Python version:

    py -3.13 -m venv build-env
    build-env\Scripts\python -m pip install -U pip wheel setuptools ninja
    build-env\Scripts\python -m pip install torch --extra-index-url https://download.pytorch.org/whl/cu130
    build-env\Scripts\python -m pip wheel --no-build-isolation --no-deps ^
        git+https://github.com/turboderp-org/exllamav3.git@v1.4.4 -w wheels

The wheel is tied to the CUDA line the PyTorch above came from, so build one
per CUDA line you intend to support (cu126 / cu128 / cu130) and name the folder
accordingly if you keep several.

## Hosting them instead

Set `WHEEL_INDEX` in `.env` to somewhere pip can reach with `--find-links`:
a GitHub Releases page, a file share, an internal index. Several are allowed,
separated by spaces or commas. The `wheels\` folder still wins when it has a
match, so a USB stick beats the network.
