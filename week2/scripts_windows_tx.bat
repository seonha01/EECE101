@echo off
rem Windows sender: edit B / IMAGE at the top of tx.py, then double-click this file
rem (or run in the conda prompt:  scripts_windows_tx.bat --B 8)
cd /d %~dp0
call conda activate eece101 2>nul
python tx.py %*
pause
