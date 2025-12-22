import argparse, json, re
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

SYSTEM_PROMPT = (
    "You are a rewriting assistant.\n"
    "Task: rewrite ONLY the user's utterance to be more natural and spoken (verbal-friendly), "
    "while preserving the original meaning.\n"
    "Rules:\n"
    "- Do NOT answer the question.\n"
    "- Do NOT add new facts.\n"
    "- Output ONLY the rewritten user utterance (no explanations, no quotes).\n"
    "- If the utterance starts with '<image>\\n', do not rewrite '<image>' itself.\n"
)

def split_image_prefix(s: str):
    """If the text starts with the multimodal placeholder '<image>\\n', keep it as a prefix and rewrite only the rest."""
    prefix = ""
    if s.startswith("<image>\n"):
        prefix = "<image>\n"
        s = s[len(prefix):]
    return prefix, s

def build_context(prev_turns: List[Dict]) -> str:
    """
    Build conversation context from previous turns ONLY (to avoid leaking future information).
    This includes both USER and ASSISTANT messages, which helps the rewriting model understand context.
    """
    lines = []
    for t in prev_turns:
        role = t.get("role", "").upper()
        content = t.get("content", "").strip()
        lines.append(f"{role}: {content}")
    return "\n".join(lines)

def clean_generation(s: str) -> str:
    """Post-process model output: remove common prefixes/quotes and keep a single clean line."""
    s = s.strip()
    # Remove common output prefixes the model may add.
    s = re.sub(r'^(REWRITE|Rewritten|USER|USER_UTTERANCE|OUTPUT)\s*:\s*', '', s, flags=re.I).strip()
    # Strip paired quotes.
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    # Keep only the first line to avoid multi-line explanations.
    s = s.splitlines()[0].strip()
    return s

@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default="clevr_first1000_raw.jsonl")
    ap.add_argument("--outfile", default="clevr_first1000_rephrased.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    model.eval()

    with open(args.infile, "r", encoding="utf-8") as fin, open(args.outfile, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, total=1000):
            ex = json.loads(line)
            conv = ex["conversations"]
            new_conv = []

            for idx, turn in enumerate(conv):
                # Keep assistant turns unchanged; only rewrite user turns.
                if turn.get("role") != "user":
                    new_conv.append(turn)
                    continue

                raw = turn.get("content", "")
                prefix, user_text = split_image_prefix(raw)

                # Provide context from all previous turns (including assistant turns).
                # IMPORTANT: only use turns before the current user turn to avoid leaking future info.
                context = build_context(conv[:idx])

                prompt = (
                    f"{SYSTEM_PROMPT}\n"
                    f"CONTEXT:\n{context}\n\n"
                    f"USER_UTTERANCE_TO_REWRITE:\n{user_text}\n\n"
                    f"REWRITE:"
                )

                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                )
                gen = tokenizer.decode(out_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
                rewritten = clean_generation(gen)

                new_conv.append({"role": "user", "content": prefix + rewritten})

            out_ex = {
                "id": ex.get("id"),
                "data_source": ex.get("data_source"),
                "conversations": new_conv,
            }
            fout.write(json.dumps(out_ex, ensure_ascii=False) + "\n")

    print(f"[OK] wrote -> {args.outfile}")

if __name__ == "__main__":
    main()
