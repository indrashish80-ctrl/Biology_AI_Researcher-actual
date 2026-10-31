@echo off
REM BioSynthetica - double-click this file to start (Windows)
cd /d %~dp0
where python >nul 2>nul || (echo Python is not installed. Install Python 3.10+ from python.org and tick "Add Python to PATH". & pause & exit /b 1)
REM a half-made environment (an interrupted first run) is rebuilt instead of being reused
if exist .venv if not exist .venv\Scripts\python.exe (echo The program environment is incomplete - rebuilding it... & rmdir /s /q .venv)
if not exist .venv (echo First run: creating environment... & python -m venv .venv)
call .venv\Scripts\activate
REM Fast start: packages are installed only when requirements.txt has changed since the last successful install
python -c "import hashlib,pathlib,sys;h=hashlib.sha256(pathlib.Path('requirements.txt').read_bytes()).hexdigest();p=pathlib.Path('.venv/req.sha256');sys.exit(0 if p.exists() and p.read_text().strip()==h else 1)"
if errorlevel 1 (
  echo Installing packages - the first run takes a few minutes. Do not close this window.
  python -m pip install -q --upgrade pip > install_log.txt 2>&1
  python -m pip install --prefer-binary -r requirements.txt >> install_log.txt 2>&1
  if errorlevel 1 goto install_failed
  python -m pip install --prefer-binary -r requirements-lab.txt >> install_log.txt 2>&1
  if errorlevel 1 echo The Simulator package (Tellurium) could not be installed - everything else works. See install_log.txt.
  python -c "import hashlib,pathlib;pathlib.Path('.venv/req.sha256').write_text(hashlib.sha256(pathlib.Path('requirements.txt').read_bytes()).hexdigest())"
  echo Packages installed.
) else (
  echo Packages already installed - skipping.
)
python check_setup.py
if errorlevel 1 (echo. & echo Fix the problems listed above, then double-click run.bat again. & pause & exit /b 1)
start "" http://127.0.0.1:8000
python server.py
pause
exit /b 0

:install_failed
echo.
echo Package installation failed. The last lines of the log:
echo ----------------------------------------------------------------------
powershell -NoProfile -Command "Get-Content -Path 'install_log.txt' -Tail 25" 2>nul
echo ----------------------------------------------------------------------
findstr /i /c:"Long Path" /c:"path too long" /c:"No such file or directory" install_log.txt >nul 2>nul && (
  echo This folder path is too long for Windows. Move the BioSynthetica folder to a short path such as C:\BioSynthetica
  echo and double-click run.bat again.
)
echo The full log is saved as install_log.txt in this folder - send it if you need help.
pause
exit /b 1
