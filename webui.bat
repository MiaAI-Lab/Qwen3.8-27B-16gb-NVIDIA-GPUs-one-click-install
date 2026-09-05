@echo off
setlocal
title Simplex - web UI (no model)
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
  echo   engine is installed - then run webui.bat again.
  echo.
  pause
  exit /b 1
)

echo.
echo   +------------------------------------------------------------+
echo   ^|  Simplex - the chat UI on its own, no model loaded         ^|
echo   ^|  Nothing touches the GPU. Use this to browse settings,     ^|
echo   ^|  past sessions, or chat through a remote provider while    ^|
echo   ^|  the local server (or the bench) is busy - or is not       ^|
echo   ^|  running at all.                                           ^|
echo   ^|  Already running? This replaces that copy, so you always   ^|
echo   ^|  get the current code. Ctrl+C stops it.                    ^|
echo   +------------------------------------------------------------+
echo.

.venv\Scripts\python.exe -u tools\chatui.py --open %*

echo.
echo   Stopped.
echo.
pause
