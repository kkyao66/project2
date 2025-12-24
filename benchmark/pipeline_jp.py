import os
import csv
import sys
import pandas as pd
import json

import utmosv2
import whisper
from sentence_transformers import SentenceTransformer

def sentence_similarity(model, embeddings_orig, text):
    embeddings_trans = model.encode([text])

    similarities = model.similarity(embeddings_orig, embeddings_trans).tolist()
    similarity = max(s[0] for s in similarities)

    return similarity

def avg(l):
    return sum(l) / len(l)


def pipeline(tts_models, sub_dirs, texts_orig):
    if (len(tts_models) != len(sub_dirs)):
        return
    trans_model = whisper.load_model("turbo")
    similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
    scoring_model = utmosv2.create_model(pretrained=True)

    embeddings_orig = similarity_model.encode(texts_orig)

    files = []
    similarities = []
    scores = []
    tts_ = []
    
    for i, tts in enumerate(tts_models):
        for sub_dir in sub_dirs[i]:
            dir_str = tts + '/' + sub_dir
            directory = os.fsencode(dir_str)
    
            for file in sorted(os.listdir(directory)):
                filename = os.fsdecode(file)
                if filename.endswith(".wav"):
                    path = os.path.join(dir_str, filename)
                    tts_.append(tts)
                    files.append(path)
                    # transcribe
                    result = trans_model.transcribe(path)
                    transcript = result["text"]
                    # similarity
                    similarity = sentence_similarity(similarity_model, embeddings_orig, transcript)
                    similarities.append(similarity)
                    # scoring
                    mos = scoring_model.predict(input_path=path, num_workers=0)
                    scores.append(mos)

    df = pd.DataFrame({'TTS Model': tts_, 'File': files, 'Similarity': similarities, 'Score': scores})
    df.to_csv('table_jp.csv')

    return df


data = []
with open("ja_60.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            data.append(json.loads(line))

texts_orig = []
for d in data:
    texts_orig.append(d['text'])

#in current working directory I have directories with the name of the models, and within these directories there are subdirectories, as specified in sub_dirs. For instance, I have 60 audio files in chattervoice/output/english or CosyVoice/outputs/english/voice_0
tts_models =["chatterbox", "CosyVoice", "higgs-audio", "index-tts", "fishspeech"]
sub_dirs = [["outputs/japanese/voice_1.wav","outputs/japanese/voice_2.wav","outputs/japanese/voice_3.wav","outputs/japanese/voice_4.wav","outputs/japanese/voice_5.wav"], ["outputs/japanese/voice_0", "outputs/japanese/voice_1", "outputs/japanese/voice_2", "outputs/japanese/voice_3", "outputs/japanese/voice_4"], ["outputs/japanese/voice_1_jp", "outputs/japanese/voice_2_jp", "outputs/japanese/voice_3_jp", "outputs/japanese/voice_4_jp", "outputs/japanese/voice_5_jp"], ["outputs/japanese/voice_0", "outputs/japanese/voice_1", "outputs/japanese/voice_2", "outputs/japanese/voice_3", "outputs/japanese/voice_4"],["outputs/japanese/voice_0", "outputs/japanese/voice_1", "outputs/japanese/voice_2", "outputs/japanese/voice_3", "outputs/japanese/voice_4"]]

pipeline(tts_models, sub_dirs, texts_orig)
