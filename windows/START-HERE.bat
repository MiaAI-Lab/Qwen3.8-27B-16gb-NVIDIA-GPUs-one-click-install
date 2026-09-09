@echo off
setlocal EnableExtensions
title Mia's one-click Qwen3.8-27B - setup
REM The kit lives one level up; everything below is relative to it.
cd /d "%~dp0.."
cls
color 0F

REM ---------------------------------------------------------------------------
REM One click: check the graphics card, choose a model size, build the kit's own
REM Python environment, install the engine, download the weights - and then load
REM what it installed, because that is what the page it opened says will happen
REM ("this page turns into the chat window on its own") and where the harness
REM comes from. This window stays open as the server afterwards.
REM
REM   START-HERE.bat              install, then start what was installed
REM   START-HERE.bat --no-start   install only - a second size fetched, nothing loaded
REM
REM Everything a person has to decide is asked on a web page (tools\setup_web.py);
REM this file only has to find a Python and start it.
REM ---------------------------------------------------------------------------

call "%~dp0find_python.bat"
if errorlevel 1 exit /b 1

%PY% tools\win_start.py setup %*
set "ERR=%ERRORLEVEL%"

if not "%ERR%"=="0" (
  echo.
  echo   Setup exited with code %ERR%.
  echo   The reason is above, and the full log is in the logs folder next to this file.
  echo.
  echo   Press any key to close this window.
  pause >nul
)
exit /b %ERR%
