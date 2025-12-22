import torchaudio as ta
import torch
from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
import os
import json

data = []
with open("ja_60.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            data.append(json.loads(line))

texts = []
for d in data:
    texts.append(d['text'])

    
def gen_audios(texts, lang, audio_path, voices, output_path):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print(f"Using device: {device}")
    os.makedirs(f'{output_path}', exist_ok=True)
    multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    for voice in voices:
        voice_path = f'{audio_path}/{voice}'
        out_dir = f"{output_path}/{voice}" 
        os.makedirs(out_dir, exist_ok=True)
        for i in range(len(texts)):
            text = texts[i]
            wav = multilingual_model.generate(text, language_id=lang, audio_prompt_path=voice_path)
            path = f'{out_dir}/text_{i}.wav'
    
            ta.save(path, wav, multilingual_model.sr)

voices = ["voice_1.wav", "voice_2.wav", "voice_3.wav", "voice_4.wav", "voice_5.wav"]
gen_audios(texts, "ja", "voices_ja", voices, "outputs/japanese")