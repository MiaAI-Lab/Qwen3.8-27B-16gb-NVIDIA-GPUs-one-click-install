@echo off
setlocal
title Simplex - VRAM benchmark
cd /d "%~dp0"

REM A venv whose base Python was upgraded or removed keeps a python.exe that
REM exists but dies instantly - "exist" alone would trust it and then every
REM run below would fail with a cryptic traceback. Probe that it actually runs.
set "VENV_OK="
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -c "pass" >nul 2>nul
  if not errorlevel 1 set "VENV_OK=1"
)

if not defined VENV_OK (
  echo.
  echo   ERROR: .venv is missing or broken. Run start.bat once first, so the
  echo   engine is installed - then run bench.bat again.
  echo.
  pause
  exit /b 1
)

echo.
echo   +------------------------------------------------------------+
echo   ^|  Simplex - measuring VRAM and context per quant            ^|
echo   ^|  Each quant: 2 sizing loads, then a full prefill stress     ^|
echo   ^|  Results append to bench_vram.json (safe to stop / resume)  ^|
echo   +------------------------------------------------------------+
echo.

if "%~1"=="" (
  echo   No quant given - running the whole sweep, smallest first.
  echo   Ctrl+C stops it; run this again to carry on where it left off.
  echo.
  .venv\Scripts\python.exe -u tools\bench_vram.py --all %*
) else (
  if /i "%~1"=="list" (
    .venv\Scripts\python.exe -u tools\bench_vram.py --list
  ) else (
    if /i "%~1"=="report" (
      .venv\Scripts\python.exe -u tools\bench_vram.py --report
    ) else (
      .venv\Scripts\python.exe -u tools\bench_vram.py --quant %*
    )
  )
)

echo.
echo   Done. Tables: bench_vram.md   Raw numbers: bench_vram.json
echo.
pause
