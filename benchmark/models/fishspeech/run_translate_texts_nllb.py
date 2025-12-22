import argparse, json, os
from pathlib import Path

def read_jsonl(p):
    out=[]
    with open(p,"r",encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                out.append(json.loads(line))
    return out

def write_jsonl(p, rows):
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    with open(p,"w",encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# NLLB language codes (English source -> target)
NLLB_LANG = {
    "en": "eng_Latn",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "ar": "arb_Arab",
    "es": "spa_Latn",
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out_dir", default="texts")
    ap.add_argument("--langs", required=True)  # e.g. zh,ja,ko,fr,de,ar,es
    ap.add_argument("--model", default="facebook/nllb-200-distilled-600M")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")  # good for A100
    args = ap.parse_args()

    rows = read_jsonl(args.input)
    for r in rows:
        if "sid" not in r or "text" not in r:
            raise SystemExit("Input must be JSONL with {'sid', 'text'} per line.")

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    for l in langs:
        if l not in NLLB_LANG:
            raise SystemExit(f"Unsupported lang '{l}'. Supported: {list(NLLB_LANG.keys())}")

    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(args.dtype, None)

    tok = AutoTokenizer.from_pretrained(args.model)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map="auto" if args.device == "cuda" else None,
    )

    tok.src_lang = NLLB_LANG["en"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for lang in langs:
        tgt = NLLB_LANG[lang]
        forced_bos = tok.convert_tokens_to_ids(tgt)

        outputs = []
        for i in range(0, len(rows), args.batch_size):
            batch = rows[i:i+args.batch_size]
            texts = [b["text"] for b in batch]
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True).to(mdl.device)

            with torch.no_grad():
                gen = mdl.generate(
                    **enc,
                    forced_bos_token_id=forced_bos,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )

            dec = tok.batch_decode(gen, skip_special_tokens=True)
            for b, t in zip(batch, dec):
                outputs.append({"sid": int(b["sid"]), "text": t.strip()})

        outputs.sort(key=lambda x: x["sid"])
        out_path = out_dir / f"{lang}_{len(rows)}.jsonl"
        write_jsonl(str(out_path), outputs)
        print(f"[OK] wrote {out_path}")

if __name__ == "__main__":
    main()