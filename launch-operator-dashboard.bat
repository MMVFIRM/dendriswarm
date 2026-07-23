@echo off
setlocal
if not defined PYTHON_BIN set PYTHON_BIN=python
if not defined DENDRISWARM_VENV set DENDRISWARM_VENV=.venv
if not exist "%DENDRISWARM_VENV%\Scripts\python.exe" %PYTHON_BIN% -m venv "%DENDRISWARM_VENV%"
"%DENDRISWARM_VENV%\Scripts\python.exe" -m pip install --upgrade pip
"%DENDRISWARM_VENV%\Scripts\python.exe" -m pip install -e ".[coordinator]"
"%DENDRISWARM_VENV%\Scripts\dendriswarm.exe" app %*
endlocal
