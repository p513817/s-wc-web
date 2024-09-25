@echo off
set SWC_LIB=swc
set SWC_HOME=runtime.py

set swc_exec=%SWC_LIB%\%SWC_HOME%

powershell -Command "Start-Process python -ArgumentList '-m streamlit run %swc_exec%' -Verb RunAs"

if %errorlevel% equ 0 (
    echo Command executed successfully
) else (
    echo Error: Command failed with return code %errorlevel%
    pause
)

