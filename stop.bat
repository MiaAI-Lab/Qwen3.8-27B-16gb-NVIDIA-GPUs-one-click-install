@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "PORT=8888"
if exist .env (
  for /f "usebackq tokens=1,* delims==" %%A in (`findstr /b /i "PORT=" .env`) do (
    set "VAL=%%B"
  )
)
if defined VAL (
  for /f "tokens=1" %%P in ("%VAL%") do set "PORT=%%P"
)

powershell -NoProfile -Command ^
  "$p=%PORT%; $c=Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1; " ^
  "if (-not $c) { Write-Host ('No server listening on port ' + $p + '.'); exit 0 }; " ^
  "$proc=Get-Process -Id $c.OwningProcess -ErrorAction SilentlyContinue; " ^
  "if (-not $proc) { Write-Host ('Port ' + $p + ' is open but the process is gone.'); exit 0 }; " ^
  "$cmd=($proc.Path + ' ' + $proc.StartInfo.Arguments); " ^
  "if ($proc.ProcessName -notmatch 'python') { Write-Host ('Port ' + $p + ' is held by ' + $proc.ProcessName + ' - refusing to kill it.'); exit 1 }; " ^
  "Write-Host ('Stopping server (pid ' + $proc.Id + ') on port ' + $p + '...'); " ^
  "Stop-Process -Id $proc.Id -ErrorAction SilentlyContinue; " ^
  "Start-Sleep -Seconds 1; " ^
  "if (Get-Process -Id $proc.Id -ErrorAction SilentlyContinue) { Stop-Process -Id $proc.Id -Force }; " ^
  "Write-Host 'Stopped.'"
exit /b %ERRORLEVEL%
