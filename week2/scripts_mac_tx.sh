#!/bin/bash
# Mac SENDER.  Usage:  bash scripts_mac_tx.sh [B] [options]
#   bash scripts_mac_tx.sh               -> B=4, pick a photo from images/
#   bash scripts_mac_tx.sh 8 --tiles 2   -> B=8, 64x64
# Defaults --baud 100 --gray (robust to room echo). The receiver must use the same B / options.
# ---- first argument = B (2/4/8, default 4), the rest are passed to python ----
B=4
case "$1" in 2|4|8) B=$1; shift ;; esac
cd "$(dirname "$0")"
# ---- find conda and activate eece101 ----
if [ "$CONDA_DEFAULT_ENV" != "eece101" ]; then
  if ! command -v conda >/dev/null 2>&1; then
    for d in "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/opt/anaconda3" \
             "/opt/homebrew/Caskroom/miniforge/base" "/opt/miniconda3" "/usr/local/miniforge3"; do
      [ -f "$d/etc/profile.d/conda.sh" ] && . "$d/etc/profile.d/conda.sh" && break
    done
  else
    eval "$(conda shell.bash hook)"
  fi
  conda activate eece101 2>/dev/null
fi
if ! python -c "import sounddevice" 2>/dev/null; then
  echo "[error] conda env 'eece101' not found. See README: conda env create -f environment.yml"
  exit 1
fi
python tx.py --B "$B" --baud 100 --gray "$@"
