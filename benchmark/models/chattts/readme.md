# ChatTTS Bring-up & Benchmark Guide (HPC / Bristen)

This document describes how to **bring up ChatTTS**, run a **minimal synthesis test**, and use it in a **benchmark-oriented TTS pipeline** on an HPC cluster (e.g., CSCS Bristen).

This guide is **CLI/Python-only** (no WebUI, no Docker) and is written to be **copy-paste runnable**.

---

## 1) Environment Assumptions

- OS: Linux (HPC login / compute node)
- Python: `>= 3.10`
- A Conda environment already exists (env name may differ)
- GPU/CUDA available **if** running on a GPU node
- Internet access is available for Hugging Face downloads

Activate your environment (example):

```bash
source ~/.bashrc
conda activate <your_env>
````

---

## 2) Repository Setup

Clone ChatTTS:

```bash
mkdir -p ~/tts_bringup
cd ~/tts_bringup
git clone https://github.com/2noise/ChatTTS.git
cd ChatTTS
```
---

## 3) Install Dependencies (Minimal, Tested)

Install Python dependencies:

```bash
pip install -U pip
pip install -r requirements.txt
```

Additional packages commonly required in practice:

```bash
pip install soundfile numpy
```

Notes:

* `ffmpeg` warnings can be ignored if you only write WAV files via `soundfile`.
* `nemo_text_processing` / `WeTextProcessing` warnings are typically non-fatal for basic English synthesis.

---

## 4) Hugging Face Access (Model Download)

ChatTTS supports automatic model download via Hugging Face.

Check whether you are authenticated:

```bash
hf auth whoami
```

If not logged in:

```bash
hf auth login
```

No manual checkpoint handling is required for the basic bring-up path.

---

## 5) Minimal Bring-up Test (CLI)

Run a single-sentence synthesis test using the provided example CLI:

```bash
python examples/cmd/run.py \
  --source hf \
  "This is a short test sentence for ChatTTS bring up."
```

Expected behavior:

* Model initializes successfully
* Audio is generated
* An output audio file is written to the working directory (exact filename/format depends on the example script)

Non-fatal warnings you may see (typically safe to ignore for basic English):

```text
Package nemo_text_processing not found
WeTextProcessing not found
ffmpeg not found
```

---

## 6) Benchmark-Friendly WAV Output (Python API)

For benchmarking, we recommend forcing **WAV (PCM)** output at a fixed sample rate.
The most reliable method is to use the Python API + `soundfile`.

Create a script `run_chattts_benchmark.py`:

```bash
cat > run_chattts_benchmark.py << 'PY'
import ChatTTS
import torch
import soundfile as sf

OUT_WAV = "chattts_demo_en.wav"
TEXT = "This is a short test sentence for ChatTTS benchmark."

SR = 24000
device = "cuda" if torch.cuda.is_available() else "cpu"

chat = ChatTTS.ChatTTS()
chat.load(source="hf", device=device)

wavs = chat.infer([TEXT])

sf.write(OUT_WAV, wavs[0], SR)
print(f"[OK] Saved {OUT_WAV} (sr={SR})")
PY
```

Run it:

```bash
python run_chattts_benchmark.py
ls -lh chattts_demo_en.wav
```

---

## 7) Recommended HPC Usage (Bristen / Slurm)

On Bristen (or similar Slurm clusters), run on a GPU node:

```bash
srun -A <project_account> --gres=gpu:1 --mem=<memory> --time=<walltime> --pty bash
```

Then:

```bash
source ~/.bashrc
conda activate <your_env>

nvidia-smi
python -c "import torch; print('cuda?', torch.cuda.is_available())"
```

Run the bring-up commands from Sections 5–6 inside this allocation.


