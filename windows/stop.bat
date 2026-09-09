@echo off
setlocal EnableExtensions
REM The kit lives one level up; everything below is relative to it.
cd /d "%~dp0.."

REM Stop the server started by start.bat, and the harness with it. The work is
REM in tools\cli.py (`simplex stop`) - the same code the Linux stop runs, so
REM the two cannot drift apart again.
REM
REM   stop.bat                 stop both
REM   stop.bat --harness-only  leave the model loaded
REM   stop.bat --server-only   leave the harness running

call "%~dp0find_python.bat"
if errorlevel 1 exit /b 1

%PY% tools\cli.py stop %*
exit /b %ERRORLEVEL%
