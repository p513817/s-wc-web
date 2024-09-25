@echo off
REM Check if Python 3.10 or higher is installed
for /f "tokens=2 delims==" %%i in ('python -c "import sys; print(sys.version_info >= (3,10))"') do set python_version=%%i
IF "%python_version%" == "False" (
    echo Python 3.10 or higher is not installed or not found in the system PATH.
    echo Please install Python 3.10 or higher and try again.
    exit /B 1
)

REM Check if .env virtual environment exists
IF EXIST .env (
    echo Virtual environment ".env" already exists.
) ELSE (
    REM Create virtual environment
    echo Creating virtual environment...
    python -m venv .env
)

REM Activate the virtual environment
echo Activating virtual environment...
call .\.env\Scripts\activate

REM Install required Python modules
echo Installing required Python modules...
python -m pip install -r .\requirements.txt

echo Setup is complete. Virtual environment and modules are ready.
