@echo off
setlocal

cd /d "C:\Users\david\Desktop\VOXCPM"

call "C:\Users\david\miniconda3\Scripts\activate.bat" voxcpm
if errorlevel 1 (
    echo Failed to activate conda environment "voxcpm".
    pause
    exit /b 1
)

python "C:\Users\david\Desktop\VOXCPM\voxcpm_chunk_studio.py"
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo VoxCPM Chunk Studio exited with code %EXIT_CODE%.
    pause
)

exit /b %EXIT_CODE%
