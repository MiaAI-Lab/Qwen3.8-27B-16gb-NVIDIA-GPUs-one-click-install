@echo off
setlocal EnableExtensions
title Simplex
cd /d "%~dp0"
cls
color 0F

REM ---------------------------------------------------------------------------
REM Simplex launcher. Everything a person has to decide is asked on a web page
REM (see tools\setup_web.py); this file only has to find a Python and start it.
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

%PY% tools\win_start.py %*
set "ERR=%ERRORLEVEL%"

if not "%ERR%"=="0" (
  echo.
  echo   Simplex exited with code %ERR%.
  echo   The reason is above, and the full log is in the logs folder next to this file.
  echo.
  echo   Press any key to close this window.
  pause >nul
)
exit /b %ERR%
