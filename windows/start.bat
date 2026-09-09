@echo off
setlocal EnableExtensions
title Mia's one-click Qwen3.8-27B
REM The kit lives one level up; everything below is relative to it.
cd /d "%~dp0.."
cls
color 0F

REM ---------------------------------------------------------------------------
REM Start a model that is already on this PC. If several sizes have been
REM downloaded it asks which one - Enter takes the one that ran last - then
REM loads it and opens the DeepSeek Harness when it is ready.
REM
REM Nothing is downloaded here. If the kit has never been set up, or no model
REM finished downloading, this offers to run setup (the same thing START-HERE.bat
REM does) rather than leaving you at a dead end.
REM
REM   start.bat            start a model
REM   start.bat setup      go straight to setup (same as START-HERE.bat)
REM ---------------------------------------------------------------------------

call "%~dp0find_python.bat"
if errorlevel 1 exit /b 1

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
