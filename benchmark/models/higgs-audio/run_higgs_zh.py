# run_generation_from_jsonl.py
import json
import subprocess
import os

# load jsonl
texts = []
with open("zh_60.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            texts.append(json.loads(line)["text"])

voices = [
    "voice1-zh",
    "voice2-zh",
    "voice3-zh",
    "voice4-zh",
    "voice5-zh",
]

for voice in voices:
    out_dir = f"outputs/chinese/{voice}"
    os.makedirs(out_dir, exist_ok=True)

    for i, text in enumerate(texts):
        print(f"Generating voice={voice}, text={i}")

        subprocess.run(
            [
                "python3",
                "examples/generation.py",
                "--transcript", text,
                "--ref_audio", voice,
                "--temperature", "0.3",
                "--out_path", f"{out_dir}/text_{i}.wav",
            ],
            check=True,
        )

print("Done")
