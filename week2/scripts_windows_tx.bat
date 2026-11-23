@echo off
rem Windows SENDER.  Usage:  scripts_windows_tx.bat [B] [options]
rem   scripts_windows_tx.bat               -> B=4, pick a photo from images\
rem   scripts_windows_tx.bat 8 --tiles 2   -> B=8, 64x64
rem Defaults --baud 100 --gray (robust to room echo). The receiver must use the same B / options.
setlocal
cd /d %~dp0
rem ---- first argument = B (2/4/8, default 4), the rest are passed to python ----
set B=4
if "%~1"=="2" (set B=2& shift)
if "%~1"=="4" (set B=4& shift)
if "%~1"=="8" (set B=8& shift)
set REST=
:args
if "%~1"=="" goto argsdone
set REST=%REST% %1
shift
goto args
:argsdone
rem ---- find conda and activate eece101 ----
if /i "%CONDA_DEFAULT_ENV%"=="eece101" goto run
for %%d in ("%USERPROFILE%\miniforge3" "%USERPROFILE%\miniconda3" "%USERPROFILE%\anaconda3" "%LOCALAPPDATA%\miniforge3" "C:\miniforge3" "%ProgramData%\miniforge3" "%ProgramData%\miniconda3") do (
    if exist "%%~d\Scripts\activate.bat" (
        call "%%~d\Scripts\activate.bat" eece101
        goto run
    )
)
call conda activate eece101 2>nul
:run
python -c "import sounddevice" 2>nul
if errorlevel 1 (
    echo [error] conda env 'eece101' not found. See README: conda env create -f environment.yml
    pause
    exit /b 1
)
python tx.py --B %B% --baud 100 --gray%REST%
pause
