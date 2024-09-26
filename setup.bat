@echo off
REM Check if Python 3.10 or higher is installed
for /f "tokens=2 delims==" %%i in ('python -c "import sys; print(sys.version_info >= (3,10))"') do set python_version=%%i
IF "%python_version%" == "False" (
    echo Python 3.10 or higher is not installed or not found in the system PATH.
    echo Please install Python 3.10 or higher and try again.
    exit /B 1
)


REM Install required Python modules
echo Installing required Python modules...
python -m pip install -r .\requirements.txt

echo Setup is complete. Virtual environment and modules are ready.
