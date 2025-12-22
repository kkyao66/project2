import torchaudio as ta
import torch
from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
import os
import json
import argparse
from pathlib import Path

# sub_dir/sub_dir/voice_paths.txt
# voice_path in voice_paths might look like this: "asset/voice_0.wav"
def get_voices(voice_path):
    path = Path(voice_path)
    voices = path.read_text(encoding="utf-8").splitlines()
    return voices

# sub_dir/sub_dir/texts.txt
def get_texts(text_path):
    path = Path(text_path)
    texts = path.read_text(encoding="utf-8").splitlines()
    return texts

def gen_audios(texts, lang, voices, output_path):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print(f"Using device: {device}")
    os.makedirs(f'{output_path}', exist_ok=True)
    multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    for voice_n, voice in enumerate(voices):
        out_dir = f"{output_path}/voice_{voice_n}" 
        os.makedirs(out_dir, exist_ok=True)
        for i in range(len(texts)):
            text = texts[i]
            wav = multilingual_model.generate(text, language_id=lang, audio_prompt_path=voice)
            path = f'{out_dir}/text_{i}.wav'
    
            ta.save(path, wav, multilingual_model.sr)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--texts",
        required=True,
        help="Path to text file (one sentence per line)"
    )

    parser.add_argument(
        "--voices",
        required=True,
        help="Path to audio list file (one wav path per line)"
    )

    parser.add_argument(
        "--lang",
        required=True,
        help="Language code (choose from 'en', 'ja', 'de', 'fr', 'zh')"
    )
    
    #"outputs/english/final"
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Path for output"
    )

    args = parser.parse_args()
    texts = get_texts(args.texts)
    voices = get_voices(args.voices)
    output_dir = args.output_dir
    lang = args.lang
    
    gen_audios(texts, lang, voices, output_dir)

if __name__ == "__main__":
    main()
