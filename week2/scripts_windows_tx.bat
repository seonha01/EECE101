@echo off
rem Windows sender: edit B / TILES / IMAGE at the top of tx.py, then double-click this file
rem defaults: --baud 100 --gray (robust to room echo). Override e.g. --baud 200
rem (or run in the conda prompt:  scripts_windows_tx.bat --B 8)
cd /d %~dp0
call conda activate eece101 2>nul
python tx.py --baud 100 --gray %*
pause
