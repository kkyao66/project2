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
