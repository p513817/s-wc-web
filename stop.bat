@echo off
:: 使用 taskkill 終止 python 進程
taskkill /F /IM python3.10.exe /T

:: 顯示已關閉的結果
if %errorlevel% equ 0 (
    echo Python process terminated successfully
) else (
    echo Failed to terminate Python process
)

pause
