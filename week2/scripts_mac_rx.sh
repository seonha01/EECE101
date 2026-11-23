#!/bin/bash
# Mac receiver. Use the same B as the sender (2 / 4 / 8).
#   bash scripts_mac_rx.sh      -> B=4
#   bash scripts_mac_rx.sh 8    -> B=8
B=${1:-4}
case $B in
  2) REC=25 ;;    # 1/24, ~9 s
  4) REC=30 ;;    # 1/12, ~18 s
  8) REC=45 ;;    # 1/6,  ~35 s
  *) echo "B must be 2, 4 or 8"; exit 1 ;;
esac
cd "$(dirname "$0")"
eval "$(conda shell.bash hook)" 2>/dev/null && conda activate eece101
echo "B=$B, recording ${REC} s -> start the sender after '녹음 시작'"
python rx.py --B "$B" --record "$REC" --save_wav "rx_B$B.wav" --fig "results/rx_B$B.png"
