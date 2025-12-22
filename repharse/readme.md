# Rephrase CLEVR (First N Samples) on Bristen — Reproducible Recipe

This document describes how to reproduce the CLEVR rephrasing pipeline on **CSCS Alps (Bristen)**. The goal is to:

1. Load the **CLEVR subset** from the Hugging Face dataset **`mvp-lab/LLaVA-OneVision-1.5-Instruct-Data`**  
2. Stream and dump the first **N** examples to JSONL  
3. Rephrase **user turns only** into a more natural, spoken style while preserving meaning

---

## 0) Assumptions

You have:

- SSH access to **CSCS bristen**
- A working conda environment (name may differ) with:
  - `python>=3.10`
  - `datasets`, `transformers`, `torch`, `tqdm`

---

## 1) Login and create a working directory

On your local machine:

ssh bristen
On bristen:

mkdir -p ~/proj/rephrase_clevr
cd ~/proj/rephrase_clevr
pwd
Expected path pattern:

/users/<YOUR_USER>/proj/rephrase_clevr
## 2) (Optional) Start an interactive GPU shell
GPU is strongly recommended for a 7B-class model. If you only run the dataset dump step (Section 4), GPU is not required.

srun -A <project_account> --gres=gpu:1 --mem=<memory> --time=<walltime> --pty bash
Notes:

Use values appropriate for your project policy.

If you are not inside an srun GPU session, torch.cuda.is_available() will typically be False.

## 3) Activate the conda environment
Inside the node (login or GPU session):

source ~/.bashrc
conda activate <your_env>
If your shell does not auto-initialize conda, explicitly load it (x86 example):

export PATH=/users/$USER/miniconda3_x86/bin:$PATH
source /users/$USER/miniconda3_x86/etc/profile.d/conda.sh
conda activate <your_env>
Sanity checks:

python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "import transformers; print('transformers OK')"
python -c "import datasets; print('datasets OK')"
python -c "import tqdm; print('tqdm OK')"
Expected:

CUDA: True if running inside a GPU srun session

all imports succeed

## 4) Dataset source and dump (stream first N samples)
4.1 Dataset source (where “CLEVR” comes from)
The CLEVR examples used here are not a standalone dataset. They are loaded from the Hugging Face dataset:

mvp-lab/LLaVA-OneVision-1.5-Instruct-Data

This dataset provides multiple subsets/configurations. In this pipeline, we select the CLEVR subset by passing config_name="CLEVR" to datasets.load_dataset, i.e.:

ds = load_dataset(
    "mvp-lab/LLaVA-OneVision-1.5-Instruct-Data",
    "CLEVR",
    split="train",
    streaming=True
)
We then stream the train split and take the first N samples (default N=1000).

4.2 Dump script (streaming)
Create the script:

cat > dump_clevr_1000.py << 'PY'
import json
from datasets import load_dataset
from tqdm import tqdm

DATASET = "mvp-lab/LLaVA-OneVision-1.5-Instruct-Data"
CONFIG  = "CLEVR"
SPLIT   = "train"
N       = 1000
OUT     = "clevr_first1000_raw.jsonl"

ds = load_dataset(DATASET, CONFIG, split=SPLIT, streaming=True)

with open(OUT, "w", encoding="utf-8") as f:
    for i, ex in enumerate(tqdm(ds, total=N)):
        if i >= N:
            break
        out_ex = {
            "id": ex.get("id"),
            "data_source": ex.get("data_source"),
            "conversations": ex["conversations"],
        }
        f.write(json.dumps(out_ex, ensure_ascii=False) + "\n")

print(f"[OK] wrote {N} samples -> {OUT}")
PY
Run it:

python dump_clevr_1000.py
Verify:

wc -l clevr_first1000_raw.jsonl
head -n 1 clevr_first1000_raw.jsonl
Expected:

line count equals 1000

each line is a JSON object containing id, data_source, conversations

## 5) Rephrase user turns only (Transformers)
This step loads an instruction-tuned LLM and rewrites only user turns to be more natural and spoken, without changing meaning. Assistant turns are kept unchanged.

5.1 Rephrasing rules enforced
Do not answer the question

Do not add new facts

Output only the rewritten user utterance (no quotes, no explanations)

If a turn starts with <image>\n, keep <image> unchanged and rewrite only the remaining text

Context is built from previous turns only (to avoid leakage)

5.2 Create the rephrasing script
rephrase_clevr_1000.py

5.3 Run rephrasing
python rephrase_clevr_1000.py \
  --infile clevr_first1000_raw.jsonl \
  --outfile clevr_first1000_rephrased.jsonl

## 6) Verify outputs

Line count:

wc -l clevr_first1000_rephrased.jsonl


Compare first sample (raw vs rephrased):

head -n 1 clevr_first1000_raw.jsonl
echo "----"
head -n 1 clevr_first1000_rephrased.jsonl


Check conversation turn distribution:

python - <<'PY'
import json
from collections import Counter

c = Counter()
with open("clevr_first1000_rephrased.jsonl","r",encoding="utf-8") as f:
    for line in f:
        ex = json.loads(line)
        c[len(ex["conversations"])] += 1
print("turn_count_distribution:", dict(sorted(c.items())))
PY


Expected (for this workflow):

turn_count_distribution: {2: 1000}

## 7) Outputs

This pipeline produces:

dump_clevr_1000.py

clevr_first1000_raw.jsonl

rephrase_clevr_1000.py

clevr_first1000_rephrased.jsonl

## 8) Parameters you may customize (optional)

Dump size: edit N in dump_clevr_1000.py

Rephrase model: --model <hf_model_name>

Generation length: --max_new_tokens <int>

Input/output paths: --infile, --outfile

Example:

python rephrase_clevr_1000.py \
  --model <your_model> \
  --max_new_tokens 96 \
  --infile clevr_first1000_raw.jsonl \
  --outfile clevr_first1000_rephrased.jsonl




