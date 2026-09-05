@echo off
setlocal EnableExtensions
title KV-cache format tests
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Run start.bat once first ^(it creates .venv and downloads the model^).
  pause & exit /b 1
)
echo.
echo  KV-cache format tests: fp16 / int8 / int8,4 / fp8 / int4 / nvfp4
echo  Needs the GPU to itself - close the server ^(stop.bat^) and other GPU apps first.
echo  Full run loads the model ~10 times ^(20-40 min^). Add --quick for a short run.
echo.
.venv\Scripts\python.exe tools\patch_kv.py apply || goto :fail
echo.
echo  [1/2] kernel parity ^(synthetic tensors through the Triton fp8 / nvfp4 kernels^)
.venv\Scripts\python.exe tools\kvtests\fp8cache_test.py || goto :fail
.venv\Scripts\python.exe tools\kvtests\nvfp4cache_test.py || goto :fail
echo.
echo  [2/2] model-level: KL vs fp16, passkey, decode speed, VRAM
.venv\Scripts\python.exe tools\kv_cache_tests.py %* || goto :fail
echo.
echo  Done. Results: kv_cache_tests.json
pause
exit /b 0
:fail
echo.
echo  A test failed ^(exit code %ERRORLEVEL%^). See the output above.
pause
exit /b 1
