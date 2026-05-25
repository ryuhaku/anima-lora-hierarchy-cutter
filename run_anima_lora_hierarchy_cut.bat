@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo Anima LoRA hierarchy cutter
echo ===========================
echo Drop one or more .safetensors LoRA files onto this .bat file.
echo Generated files will be created in the same folder as the source LoRA.
echo.

where py >nul 2>nul
if %errorlevel%==0 (
    py -3 "%~dp0anima_lora_hierarchy_cut.py" %*
) else (
    python "%~dp0anima_lora_hierarchy_cut.py" %*
)

echo.
echo Finished. Press any key to close this window.
pause >nul
