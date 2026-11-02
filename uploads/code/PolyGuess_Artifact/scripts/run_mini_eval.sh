#!/usr/bin/env bash
# Minimal end-to-end evaluation (Scenario #5 Naz k=1-4 mini subset) using the provided LoRA checkpoint.
set -euo pipefail
cd "$(dirname "$0")/.."
python main.py --mode eval --config configs/mini/eval_naz_mini.ini
echo
echo "Done. Results: results/mini_eval/input_output.jsonl, log.txt, eval_config.json"
