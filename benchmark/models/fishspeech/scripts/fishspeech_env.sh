#!/bin/bash
set -euo pipefail

export LLAMA_DIR=/iopsstor/scratch/cscs/leli/ml_project2/checkpoints/fish-speech-1.4/openaudio-s1-mini
export DECODER_CKPT=$LLAMA_DIR/codec.pth

export JOBLIB_TEMP_FOLDER=$HOME/tmp/joblib
mkdir -p "$JOBLIB_TEMP_FOLDER"

export NO_PROXY="127.0.0.1,localhost"
export no_proxy="$NO_PROXY"

# Force python from fishspeech_clean (avoid env mixups)
export PYTHON_BIN=/users/leli/miniconda3_x86/envs/fishspeech_clean/bin/python
