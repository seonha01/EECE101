@echo off
rem Windows RECEIVER.  Usage:  scripts_windows_rx.bat [B] [options]
rem   scripts_windows_rx.bat                               -> B=4, show the received image
rem   scripts_windows_rx.bat 4 --tiles 2   -> 64x64. The sent photo is found in images\ automatically
rem Start this FIRST, then the sender. Recording time is computed from the options.
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
python rx.py --B %B% --baud 100 --gray%REST%
pause
