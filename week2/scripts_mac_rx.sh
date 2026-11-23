#!/bin/bash
# Mac receiver. Use the same B (and options) as the sender.
#   bash scripts_mac_rx.sh                      -> B=4
#   bash scripts_mac_rx.sh 8                    -> B=8
#   bash scripts_mac_rx.sh 8 --baud 200         -> faster (default is --baud 100 --gray, same as the sender script)
#   bash scripts_mac_rx.sh 4 --tiles 2 --image Lenna.png  -> 64x64, compare with the original
B=${1:-4}
shift
case $B in
  2) DUR10=92  ;;   # 1/24, 9.2 s at 200 baud
  4) DUR10=175 ;;   # 1/12, 17.5 s
  8) DUR10=340 ;;   # 1/6,  34.0 s
  *) echo "B must be 2, 4 or 8"; exit 1 ;;
esac
BAUD=100   # default: slow + Gray code (robust to room echo). Override with --baud 200
TILES=1
ARGS=("$@")
for ((i = 0; i < ${#ARGS[@]}; i++)); do
  [ "${ARGS[$i]}" = "--baud" ] && BAUD=${ARGS[$((i + 1))]%.*}
  [ "${ARGS[$i]}" = "--tiles" ] && TILES=${ARGS[$((i + 1))]}
done
REC=$(( (DUR10 * TILES * TILES * 200 / BAUD + 9) / 10 + 12 ))   # sound length + 12 s margin
TAG="B$B"; [ "$TILES" != 1 ] && TAG="${TAG}_tiles$TILES"; [ "$BAUD" != 100 ] && TAG="${TAG}_baud$BAUD"
cd "$(dirname "$0")"
eval "$(conda shell.bash hook)" 2>/dev/null && conda activate eece101
echo "B=$B, tiles=$TILES, baud=$BAUD, recording ${REC} s -> start the sender after '녹음 시작'"
python rx.py --B "$B" --record "$REC" --save_wav "rx_$TAG.wav" --fig "results/rx_$TAG.png" --baud 100 --gray "$@"
