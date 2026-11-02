#!/usr/bin/env bash
# Download the exact Qwen2.5-0.5B base model used with the provided LoRA checkpoint.
set -euo pipefail
MODEL_DIR="${MODEL_DIR:-model/Qwen0.5B}"
huggingface-cli download Qwen/Qwen2.5-0.5B \
  --revision 060db6499f32faf8b98477b0a26969ef7d8b9987 \
  --local-dir "${MODEL_DIR}"
echo "Base model downloaded to ${MODEL_DIR}."
echo "Expected SHA-256 for model.safetensors: 88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342"
