# FishSpeech — Batch Generation & Benchmark Guide (CSCS Bristen / HPC)

This document describes the **end-to-end, reproducible FishSpeech pipeline** we used on **CSCS Bristen** to generate **multi-language** and **multi-voice** speech audio from **JSONL text inputs**.

It covers the full invocation chain:

- JSONL input → start FishSpeech API server → register reference voices → batch generation
- Resume / skip logic (checkpointing by existing WAVs)
- Output directory structure and artifacts
- Common pitfalls we encountered (proxy, msgpack, reference registration)

> This guide is written for **HPC / CLI usage** and assumes you do **not** use WebUI or Docker.

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

## 1) Repository paths and key files (as used on Bristen)

### 1.1 FishSpeech repository

Example path (your deployment may differ):

/users/<USER>/tts_clean/fish-speech
Key scripts:

scripts/fishspeech_env.sh
Sets environment variables (checkpoints, python path, proxy/no_proxy, etc.)

scripts/start_server.sh
Starts the FishSpeech API server (usually via nohup) and writes a server log

scripts/batch_generate_fishspeech.py
Batch generation client that calls the local server and writes audio + metadata

(Optional) scripts/run_all_langs_nohup.sh
Helper wrapper to run multiple languages

1.2 Input JSONL (TTS inputs)
We used one JSONL per language:

.../tts_inputs_en.jsonl
.../tts_inputs_fr.jsonl
.../tts_inputs_de.jsonl
.../tts_inputs_ja.jsonl
.../tts_inputs_zh.jsonl
Each line is one sample. The batch script typically supports multiple candidate text keys
(e.g., text, instruction, prompt, sentence, transcript).

1.3 Output directory (OUTBASE)
We write outputs to a single OUTBASE directory, for example:

OUTBASE=/path/to/fishspeech_batch_out
Typical structure:

fishspeech_batch_out/
  audio/
    en/voice01/000000.wav ...
    en/voice02/...
    fr/voice01/...
    ...
  logs/
    server_*.log
    <lang>_*.log
  metadata_en.jsonl
  failed_en.jsonl
  metadata_fr.jsonl
  failed_fr.jsonl
  ...
2) End-to-end reproduction steps (from 0 to audio)
2.1 Allocate a GPU node (required)
On Bristen:

ssh bristen
srun -A <project_account> --gres=gpu:1 --mem=<memory> --time=<walltime> --pty bash

hostname
nvidia-smi | head
Run both the server and batch generation inside this same srun session.

2.2 Activate environment and enter the repo
source ~/.bashrc
conda activate <your_fishspeech_env>

cd /users/<USER>/tts_clean/fish-speech
source scripts/fishspeech_env.sh
Recommended sanity check:

which python
python -c "import torch; print('cuda?', torch.cuda.is_available())"
2.3 Start the API server and verify the port
Start server (tested method):

bash scripts/start_server.sh
Wait until the port is listening:

for i in {1..120}; do
  ss -ltn | grep -q ':7860' && break
  sleep 1
done
ss -ltn | grep ':7860' || { echo "[FATAL] 7860 not listening"; exit 1; }
Server logs are typically written by the start script (commonly under $HOME/tmp/ or $OUTBASE/logs/).

3) Reference voice management (voice01..voice05)
FishSpeech voice control is done via reference_id.
The batch client sends requests like:

{"text": "...", "format": "wav", "reference_id": "voice01"}
So the server must have voice01..voice05 registered.

3.1 Proxy pitfall: curl may return HTML (proxy interception)
If curl http://127.0.0.1:7860/... returns HTML, your request is going through a proxy.

Fix:

export NO_PROXY="127.0.0.1,localhost"
export no_proxy="$NO_PROXY"
Always call the local server with:

curl --noproxy '*' -i http://127.0.0.1:7860/v1/references/list | head
3.2 /v1/references/list returns msgpack (not JSON)
The list endpoint may return:

content-type: application/msgpack

Decode with Python:

python - <<'PY'
import requests, msgpack
r = requests.get(
    "http://127.0.0.1:7860/v1/references/list",
    proxies={"http": None, "https": None}
)
print(msgpack.unpackb(r.content, raw=False))
PY
3.3 Register references: field name is audio (not file)
To add a reference, the correct multipart field is audio=@...wav:

REFDIR=/path/to/reference_wavs
PROMPT_TEXT="This is the reference voice prompt."

curl --noproxy '*' -sS -X POST "http://127.0.0.1:7860/v1/references/add" \
  -F "id=voice01" \
  -F "text=$PROMPT_TEXT" \
  -F "audio=@$REFDIR/voice01.wav"
Batch-add five voices:

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

4) Batch generation (per language)
4.1 Why OUTBASE matters (skip/resume behavior)
batch_generate_fishspeech.py typically skips generation if the target WAV file already exists.

This is useful for resume, but it also means:

If you change reference voices and reuse the same OUTBASE, old WAVs will be skipped.

After updating references, use a new OUTBASE (recommended), or delete previous audio.

4.2 Run one language (example: zh)
JSONL=/path/to/tts_inputs_zh.jsonl
OUTBASE=/path/to/fishspeech_batch_out_zh_run

mkdir -p "$OUTBASE/logs"

nohup python -u scripts/batch_generate_fishspeech.py \
  --lang zh \
  --jsonl "$JSONL" \
  --outdir "$OUTBASE" \
  > "$OUTBASE/logs/zh_run.log" 2>&1 &

echo "[OK] log: $OUTBASE/logs/zh_run.log"
4.3 Monitor progress

watch -n 60 "echo WAVS=\$(find $OUTBASE/audio/zh -name '*.wav' 2>/dev/null | wc -l); tail -n 5 $OUTBASE/logs/zh_run.log"
5) Resume / checkpointing (how it works)
The batch script implements resume by checking the expected output file path:

If a WAV already exists (and passes a basic validity check), it is skipped

Re-running the same command fills in missing indices

This is why you can safely re-run the same command after interruptions.

6) Completion checks (recommended)
6.1 Total count
find "$OUTBASE/audio/zh" -type f -name '*.wav' | wc -l
6.2 Per-voice distribution
for v in voice01 voice02 voice03 voice04 voice05; do
  printf "zh %s: " "$v"
  find "$OUTBASE/audio/zh/$v" -type f -name '*.wav' 2>/dev/null | wc -l
done
6.3 Detect empty or suspiciously small WAV files (optional)
find "$OUTBASE/audio/zh" -type f -name '*.wav' -size 0 -print | head
find "$OUTBASE/audio/zh" -type f -name '*.wav' -size -10k -print | head
7) Common issues (what we actually hit)
7.1 curl returns HTML instead of API output
Cause: proxy interception of localhost requests.
Fix: always use:

export NO_PROXY="127.0.0.1,localhost"
export no_proxy="$NO_PROXY"
curl --noproxy '*' ...
7.2 /v1/references/list is not JSON
It returns msgpack (application/msgpack). Decode with Python + msgpack (see Section 3.2).

7.3 /v1/references/add returns 422
Usually the multipart field is wrong. Use audio=@...wav (not file=@...).

7.4 tools/api_client.py fails due to pyaudio
On HPC, pyaudio is often missing. Prefer:

curl --noproxy '*' for reference endpoints, and

Python requests + msgpack for decoding list responses

8) Upload reference WAVs to Bristen (example)
Stage your local reference wavs as:

voice01.wav
voice02.wav
voice03.wav
voice04.wav
voice05.wav
Then copy to Bristen:

rsync -avP /local/path/to/reference_wavs/ bristen:/path/to/reference_wavs_on_bristen/
9) Export artifacts (metadata / audio) back to local (optional)
Example: pull metadata JSONLs back to local:

rsync -avP bristen:$OUTBASE/metadata_*.jsonl /local/path/metadata/
For large audio trees, consider transferring per language or per voice.
