#!/bin/bash
# Mac RECEIVER.  Usage:  bash scripts_mac_rx.sh [B] [options]
#   bash scripts_mac_rx.sh                               -> B=4, show the received image
#   bash scripts_mac_rx.sh 4 --tiles 2   -> 64x64. The sent photo is found in images/ automatically
# Start this FIRST, then the sender. Recording time is computed from the options.
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
python rx.py --B "$B" --baud 100 --gray "$@"
