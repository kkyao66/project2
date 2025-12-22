# FishSpeech — Batch Generation & Benchmark Guide (CSCS Bristen / HPC)

This document describes the **end-to-end, reproducible FishSpeech pipeline** used on **CSCS Bristen** to generate **multi-language** and **multi-voice** speech audio from **JSONL text inputs**, and how it links to the repository’s **scripts**, **configs**, and **server entrypoints**.

It covers:

- JSONL input → start FishSpeech API server → register reference voices → batch generation
- Resume / skip logic (checkpointing by existing WAVs)
- Output directory structure and artifacts
- Common pitfalls encountered (proxy, msgpack, reference registration)
- How the following repo components fit together:
  - `scripts/fishspeech_env.sh`
  - `scripts/start_server.sh`
  - `configs/benchmark_fishspeech_60_{en,de,fr,ja,zh}_v1.json`
  - `tools/api_server.py`
  - `requirements.txt`

> This guide is written for **HPC / CLI usage** and assumes **no WebUI** and **no Docker**.

---

## 0) What this pipeline does

Given a JSONL file of texts, FishSpeech generates WAV audio organized by:

- **language**: `en/fr/de/ja/zh` (or any supported lang tag you use)
- **voice**: `voice01..voice05` (each mapped to a FishSpeech `reference_id`)
- **index**: `000000.wav ...`

For each language run, the pipeline also writes:

- `metadata_<lang>.jsonl` — one line per generated sample (paths, text, voice, etc.)
- `failed_<lang>.jsonl` — failed samples for retry/debug (often empty if stable)

### Two-stage architecture (Server–Client)

FishSpeech runs as a **local HTTP service** plus a **batch client**:

1. **Server (GPU node):** `tools/api_server.py` loads checkpoints and serves HTTP on `127.0.0.1:7860`
2. **Batch client (same GPU node):** `scripts/batch_generate_fishspeech.py` reads JSONL, calls the local server, and saves WAVs + manifests

**Critical constraint:** the server and the batch client must run on the **same compute node**, because the client calls `http://127.0.0.1:7860`.

---

## 1) Repository layout and linkage (scripts / configs / entrypoints)

### 1.1 Key files and roles

- `requirements.txt`  
  Python dependencies required by the FishSpeech server and batch generation scripts.

- `scripts/fishspeech_env.sh`  
  Centralized runtime environment variables (checkpoint paths, python executable, proxy/no_proxy).  
  Both server and batch should source this script to ensure consistent settings.

- `scripts/start_server.sh`  
  Canonical server launcher. Typically:
  - `cd` into repo root
  - `source scripts/fishspeech_env.sh`
  - launches `tools/api_server.py` via `nohup` (or equivalent)
  - writes server logs to a known location

- `tools/api_server.py`  
  The actual FishSpeech HTTP server entrypoint. It loads checkpoints and listens on `127.0.0.1:7860`.

- `configs/benchmark_fishspeech_60_<lang>_v1.json`  
  Benchmark configuration files for the “60 sentences” benchmark, one per language (`en/de/fr/ja/zh`).  
  These configs are the **single source of truth** for benchmark runs (texts, voices, audio settings, inference options, output layout).

### 1.2 Invocation chain (high-level)

```text
requirements.txt
   ↓ (install deps)
scripts/fishspeech_env.sh
   ↓ (export ckpt/proxy/python vars)
scripts/start_server.sh
   ↓ (launch)
tools/api_server.py  (serving on 127.0.0.1:7860)
   ↓ (client requests; same node)
batch generation / benchmark runner
   ↓ (driven by)
configs/benchmark_fishspeech_60_<lang>_v1.json
   ↓
OUTBASE/audio/<lang>/<voice>/<id>.wav + metadata_<lang>.jsonl
````

---

## 2) Prerequisites

* SSH access to Bristen
* Slurm access to request a GPU allocation
* A working conda environment (name may differ)
* Internet access (for first-time model download / HF cache population, if applicable)

---

## 3) Install dependencies (`requirements.txt`)

From repo root:

```bash
pip install -U pip
pip install -r requirements.txt
```

If your environment is conda-based, activate it first:

```bash
source ~/.bashrc
conda activate <your_fishspeech_env>
```

---

## 4) End-to-end reproduction steps (from 0 to audio)

### 4.1 Allocate a GPU node (required)

On Bristen:

```bash
ssh bristen
srun -A <project_account> --gres=gpu:1 --mem=<memory> --time=<walltime> --pty bash

hostname
nvidia-smi | head
```

Run both the server and batch generation inside this same `srun` session.

### 4.2 Activate environment and enter the repo

```bash
source ~/.bashrc
conda activate <your_fishspeech_env>

cd /users/<USER>/tts_clean/fish-speech
source scripts/fishspeech_env.sh
```

Sanity checks:

```bash
which python
python -c "import torch; print('cuda?', torch.cuda.is_available())"
```

### 4.3 Start the API server and verify the port

Start server:

```bash
bash scripts/start_server.sh
```

Wait until the port is listening:

```bash
for i in {1..120}; do
  ss -ltn | grep -q ':7860' && break
  sleep 1
done
ss -ltn | grep ':7860' || { echo "[FATAL] 7860 not listening"; exit 1; }
```

Server logs are typically written by `scripts/start_server.sh` (commonly under `$HOME/tmp/` or an output logs directory).

---

## 5) Reference voice management (voice01..voice05)

FishSpeech voice control is done via `reference_id`. The batch client sends requests like:

```json
{"text": "...", "format": "wav", "reference_id": "voice01"}
```

So the server must have `voice01..voice05` registered.

### 5.1 Proxy pitfall: `curl` may return HTML (proxy interception)

If `curl http://127.0.0.1:7860/...` returns HTML, your request is going through a proxy.

Fix:

```bash
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="$NO_PROXY"
```

Always call the local server with:

```bash
curl --noproxy '*' -i http://127.0.0.1:7860/v1/references/list | head
```

### 5.2 `/v1/references/list` returns msgpack (not JSON)

The list endpoint may return:

```text
content-type: application/msgpack
```

Decode with Python:

```bash
python - <<'PY'
import requests, msgpack
r = requests.get(
    "http://127.0.0.1:7860/v1/references/list",
    proxies={"http": None, "https": None}
)
print(msgpack.unpackb(r.content, raw=False))
PY
```

### 5.3 Register references: field name is `audio` (not `file`)

To add a reference, the correct multipart field is `audio=@...wav`:

```bash
REFDIR=/path/to/reference_wavs
PROMPT_TEXT="This is the reference voice prompt."

curl --noproxy '*' -sS -X POST "http://127.0.0.1:7860/v1/references/add" \
  -F "id=voice01" \
  -F "text=$PROMPT_TEXT" \
  -F "audio=@$REFDIR/voice01.wav"
```

Batch-add five voices:

```bash
REFDIR=/path/to/reference_wavs
PROMPT_TEXT="This is the reference voice prompt."

for id in voice01 voice02 voice03 voice04 voice05; do
  wav="$REFDIR/${id}.wav"
  echo "[ADD] $id <- $wav"
  curl --noproxy '*' -sS -X POST "http://127.0.0.1:7860/v1/references/add" \
    -F "id=$id" \
    -F "text=$PROMPT_TEXT" \
    -F "audio=@$wav" \
    > /tmp/add_${id}.resp
done
```

### 5.4 Reference state inconsistency (“already exists” but list is empty)

If `/v1/references/add` reports “already exists” but `/v1/references/list` shows none, the safest recovery is:

1. stop server
2. backup and rebuild repo-local `references/`
3. restart server
4. re-add voices

Example:

```bash
# stop server (example; adapt to your start_server.sh implementation)
PID=$(ss -ltnp | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' | head -n 1)
kill "$PID"
sleep 2

cd /users/<USER>/tts_clean/fish-speech
[ -d references ] && mv references references_backup_$(date +%F_%H%M%S)
mkdir -p references

bash scripts/start_server.sh
for i in {1..120}; do ss -ltn | grep -q ':7860' && break; sleep 1; done

# re-add voice01..05 (see Section 5.3)
```

---

## 6) Batch generation (per language)

### 6.1 Why OUTBASE matters (skip/resume behavior)

`batch_generate_fishspeech.py` typically **skips** generation if the target WAV file already exists.

This is useful for resume, but it also means:

* If you change reference voices and reuse the same OUTBASE, old WAVs will be skipped.
* After updating references, use a new OUTBASE (recommended), or delete previous audio.

### 6.2 Run one language directly (example: zh)

```bash
JSONL=/path/to/tts_inputs_zh.jsonl
OUTBASE=/path/to/fishspeech_batch_out_zh_run

mkdir -p "$OUTBASE/logs"

nohup python -u scripts/batch_generate_fishspeech.py \
  --lang zh \
  --jsonl "$JSONL" \
  --outdir "$OUTBASE" \
  > "$OUTBASE/logs/zh_run.log" 2>&1 &

echo "[OK] log: $OUTBASE/logs/zh_run.log"
```

### 6.3 Monitor progress

```bash
watch -n 60 "echo WAVS=\$(find $OUTBASE/audio/zh -name '*.wav' 2>/dev/null | wc -l); tail -n 5 $OUTBASE/logs/zh_run.log"
```

---

## 7) Run the “60 sentences benchmark” from configs (recommended)

This repository provides benchmark configs:

* `configs/benchmark_fishspeech_60_en_v1.json`
* `configs/benchmark_fishspeech_60_de_v1.json`
* `configs/benchmark_fishspeech_60_fr_v1.json`
* `configs/benchmark_fishspeech_60_ja_v1.json`
* `configs/benchmark_fishspeech_60_zh_v1.json`

These configs are intended to drive a consistent benchmark run per language.

### 7.1 Typical usage pattern

If your project uses a unified benchmark runner script, run:

```bash
python <your_benchmark_runner>.py --config configs/benchmark_fishspeech_60_en_v1.json
```

Repeat for other languages by swapping the config file.

> Note: the exact runner entrypoint depends on your repo (e.g., `run_benchmark_generate.py`).
> Replace `<your_benchmark_runner>.py` with the actual script used in your codebase.

### 7.2 Expected outputs

Outputs should follow the benchmark output convention, e.g.:

```text
<OUTBASE>/
  audio/
    <lang>/voice01/*.wav
    <lang>/voice02/*.wav
    ...
  metadata_<lang>.jsonl
  failed_<lang>.jsonl
  logs/
    <lang>_*.log
```

---

## 8) Resume / checkpointing (how it works)

The batch script implements resume by checking the expected output file path:

* If a WAV already exists (and passes a basic validity check), it is skipped
* Re-running the same command fills in missing indices

This is why you can safely re-run the same command after interruptions.

---

## 9) Completion checks (recommended)

### 9.1 Total count

```bash
find "$OUTBASE/audio/zh" -type f -name '*.wav' | wc -l
```

### 9.2 Per-voice distribution

```bash
for v in voice01 voice02 voice03 voice04 voice05; do
  printf "zh %s: " "$v"
  find "$OUTBASE/audio/zh/$v" -type f -name '*.wav' 2>/dev/null | wc -l
done
```

### 9.3 Detect empty or suspiciously small WAV files (optional)

```bash
find "$OUTBASE/audio/zh" -type f -name '*.wav' -size 0 -print | head
find "$OUTBASE/audio/zh" -type f -name '*.wav' -size -10k -print | head
```

---

## 10) Common issues (what we actually hit)

### 10.1 `curl` returns HTML instead of API output

Cause: proxy interception of localhost requests. Fix: always use:

```bash
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="$NO_PROXY"
curl --noproxy '*' ...
```

### 10.2 `/v1/references/list` is not JSON

It returns msgpack (`application/msgpack`). Decode with Python + `msgpack` (see Section 5.2).

### 10.3 `/v1/references/add` returns 422

Usually the multipart field is wrong. Use `audio=@...wav` (not `file=@...`).

### 10.4 `tools/api_client.py` fails due to `pyaudio`

On HPC, `pyaudio` is often missing. Prefer:

* `curl --noproxy '*'` for reference endpoints, and
* Python `requests + msgpack` for decoding list responses

---

## 11) Upload reference WAVs to Bristen (example)

Stage your local reference wavs as:

```text
voice01.wav
voice02.wav
voice03.wav
voice04.wav
voice05.wav
```

Then copy to Bristen:

```bash
rsync -avP /local/path/to/reference_wavs/ bristen:/path/to/reference_wavs_on_bristen/
```

---

## 12) Export artifacts (metadata / audio) back to local (optional)

Example: pull metadata JSONLs back to local:

```bash
rsync -avP bristen:$OUTBASE/metadata_*.jsonl /local/path/metadata/
```

For large audio trees, consider transferring per language or per voice.

---

## 13) Minimal “one-screen” summary

1. `ssh bristen` → `srun ... --gres=gpu:1 ... --pty bash`
2. `conda activate <env>` → `cd fish-speech` → `source scripts/fishspeech_env.sh`
3. `bash scripts/start_server.sh` and confirm `:7860` is listening
4. Register `voice01..voice05` via `/v1/references/add` (`audio=@...wav`, `--noproxy '*'`)
5. Run `python scripts/batch_generate_fishspeech.py --lang <lang> --jsonl ... --outdir ...`
6. Re-run to resume; use a new OUTBASE when you change reference voices

```

