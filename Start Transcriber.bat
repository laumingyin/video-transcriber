@echo off
setlocal
cd /d "%~dp0"
set "PY=.venv\Scripts\python.exe"

if not exist "%PY%" (
  echo First-time setup: creating a Python environment for the app.
  echo This downloads about 1 GB and takes a few minutes. It only happens once.
  echo.
  py -3.13 -m venv .venv 2>nul || python -m venv .venv
  if not exist "%PY%" (
    echo Could not create the environment. Is Python installed? Get it from https://www.python.org
    pause
    exit /b 1
  )
)

echo Checking packages...
"%PY%" -m pip install --quiet --disable-pip-version-check -r requirements.txt
"%PY%" -c "import openvoice" 2>nul || "%PY%" -m pip install --quiet --disable-pip-version-check --no-deps "https://github.com/myshell-ai/OpenVoice/archive/refs/heads/main.zip"

echo.
echo Keep this window open while you use the app. Close it to stop.
echo.
"%PY%" server.py
pause
