#!/usr/bin/env bash
# OPTIONAL: train POLYGUESS (Poly-1 loss) on the mini subset from scratch (requires a GPU).
set -euo pipefail
cd "$(dirname "$0")/.."
python main.py --mode train --config configs/mini/train_naz_mini.ini
echo
echo "Done. Checkpoint: checkpoints/mini_train_output/"
