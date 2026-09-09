@echo off
REM ---------------------------------------------------------------------------
REM Find a Python to run the kit with, and leave it in PY for the caller.
REM
REM Called by start.bat and START-HERE.bat. It deliberately has no setlocal: the
REM whole point is to set a variable in the script that called it. Returns 1
REM when there is no Python at all, having said what to do about it.
REM ---------------------------------------------------------------------------

REM Prefer the kit's own environment - but only if it still runs. A venv whose
REM base Python was upgraded or removed keeps a python.exe that dies instantly,
REM and trusting it means win_start.py never runs, so the very code that would
REM rebuild the venv can never be reached.
set "PY="
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -c "pass" >nul 2>nul
  if not errorlevel 1 (
    set "PY=.venv\Scripts\python.exe"
  ) else (
    echo   The kit's Python environment is broken - rebuilding it.
    echo.
  )
)

if not defined PY (
  where py >nul 2>nul && set "PY=py -3"
)
if not defined PY (
  where python >nul 2>nul && set "PY=python"
)

if not defined PY (
  echo.
  echo   Simplex needs Python and cannot find it.
  echo.
  echo   1. Open https://www.python.org/downloads/
  echo   2. Install Python 3.11 or newer ^(64-bit^)
  echo   3. Tick "Add python.exe to PATH" in the installer
  echo   4. Run this file again
  echo.
  echo   Nothing else has to be installed by hand - Simplex sets up the rest itself.
  echo.
  pause
  exit /b 1
)
exit /b 0
