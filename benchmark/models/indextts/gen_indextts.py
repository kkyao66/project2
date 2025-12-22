import os
import argparse
from pathlib import Path
from indextts.infer_v2 import IndexTTS2

def gen_audio(tts, texts, voices, output_dir):
    emo_ref = None
    for voice_n, voice in enumerate(voices):
        os.makedirs(f'{output_dir}/voice_{voice_n}', exist_ok=True)
    
        for i in range(len(texts)):
            text = texts[i]
            path = f'{output_dir}/voice_{voice_n}/text_{i}.wav'
            tts.infer(spk_audio_prompt=voice, text=text, output_path=path, emo_audio_prompt=emo_ref, verbose=True)


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

    tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_fp16=False, use_cuda_kernel=True, use_deepspeed=False)
    gen_audio(tts, texts, voices, output_dir)