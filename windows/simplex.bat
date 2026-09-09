@echo off
setlocal EnableExtensions
REM The kit lives one level up; everything below is relative to it.
cd /d "%~dp0.."

REM One command for the whole kit:  simplex start ^| stop ^| status ^| doctor ...
REM Everything it does lives in tools\cli.py, so Windows and Linux behave alike.

call "%~dp0find_python.bat"
if errorlevel 1 exit /b 1

%PY% tools\cli.py %*
exit /b %ERRORLEVEL%
