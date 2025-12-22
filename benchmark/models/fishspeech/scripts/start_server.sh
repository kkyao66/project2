#!/bin/bash
set -euo pipefail
cd /users/leli/tts_clean/fish-speech

source scripts/fishspeech_env.sh

LOG="$HOME/tmp/fishspeech_server_${SLURM_JOB_ID:-manual}.log"
mkdir -p "$HOME/tmp"

nohup "$PYTHON_BIN" -u tools/api_server.py --mode tts --device cuda --half \
  --llama-checkpoint-path "$LLAMA_DIR" \
  --decoder-checkpoint-path "$DECODER_CKPT" \
  --listen 127.0.0.1:7860 --workers 1 \
  > "$LOG" 2>&1 &

echo "[OK] server starting, log=$LOG"
