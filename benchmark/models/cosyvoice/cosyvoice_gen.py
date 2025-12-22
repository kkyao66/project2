import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import AutoModel
import torchaudio
import torch
import os
import json
import pyopenjtalk
from pathlib import Path
import argparse

# Combine audios into 1, since inference_instruct2 and inference_zero_shot can generate more than 2 audios
def one_audio(gen, sr, out_path):
    chunks = []
    for j in gen:
        wav = j["tts_speech"]
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        chunks.append(wav)

    full = torch.cat(chunks, dim=1)
    torchaudio.save(out_path, full.cpu(), sr)

# generate audio
# make output directories for each voice (specified_output_path/voice_num)
# if tone_instruct is true, run audio generation for each prompt
# otherwise, prompt is one-to-one transcription of refrence audio (voice)
def gen_audio(cosyvoice, texts, audios, prompts, output_path, tone_instruct=False):
    if not tone_instruct:
        assert len(audios) == len(prompts), "audios and prompts must be equal length"
        for audio_n, audio in enumerate(audios):
            prompt = prompts[audio_n]
            os.makedirs(f'{output_path}/voice_{audio_n}', exist_ok=True)

            for text_n in range(len(texts)):
                gen = cosyvoice.inference_zero_shot(texts[text_n], prompt, audio)
                one_audio(gen, cosyvoice.sample_rate, f"{output_path}/voice_{audio_n}/text_{text_n}.wav")
    else:
        for i, prompt in enumerate(prompts):
            for audio_n, audio in enumerate(audios):
                os.makedirs(f'{output_path}/prompt_{i}/voice_{audio_n}', exist_ok=True)
                for text_n in range(len(texts)):
                    gen = cosyvoice.inference_instruct2(texts[text_n], prompt, audio, stream=False)
                    one_audio(gen, cosyvoice.sample_rate, f"{output_path}/prompt_{i}/voice_{audio_n}/text_{text_n}.wav")

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

# sub_dir/sub_dir/prompts.txt
def get_prompts(prompt_path):
    path = Path(prompt_path)
    prompts = path.read_text(encoding="utf-8").splitlines()
    return prompts

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
        "--prompts",
        required=True,
        help="Path to prompt list file (one prompt per line)"
    )

    parser.add_argument(
        "--instruct",
        action="store_true",
        help="True if using tone instruction (default False)"
    )

    #"outputs/english/final"
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Path for output"
    )

    args = parser.parse_args()
    texts = get_texts(args.texts)
    audios = get_voices(args.voices)
    prompts = get_prompts(args.prompts)
    use_instruct = args.instruct
    output_dir = args.output_dir

    cosyvoice = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B')
    gen_audio(cosyvoice, texts, audios, prompts, output_dir, tone_instruct=use_instruct)

if __name__ == "__main__":
    main()

