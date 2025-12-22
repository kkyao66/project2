# IndexTTS Batch Audio Generator

This script generates speech audio using **IndexTTS**, supporting:

- **Zero-shot voice cloning**

It takes:
- a list of texts
- a list of reference voice WAV files

and generates WAV outputs in a structured directory layout.



## Requirements
detailed setup on https://github.com/index-tts/index-tts


## File Format
1. 'voices.txt': Each line is a path to a reference WAV file (one voice per line):
	```
    assets/voice_0.wav
	assets/voice_1_jp.wav
    ```
	
	These reference WAV files define the speaker identity used for voice cloning.
2. 'texts.txt': Each line is a text to be spoken:
    ```
	"Why did you do that to me! I didn't want you to say that to me, I am very offended!"
	"Why is my life so miserable, I want to just live in peace... Someone please help..."
    ```
## Usage
### Zero-shot voice cloning
```bash
python gen_indextts.py \
  --texts path/to/sample_text.txt \
  --voices path/to/sample_voices.txt \
  --output_dir outputs/dir 
  ```

## Output Structure 
there will be num_voices * num_texts outputs
```
outputs/dir/
├── voice_0/
│   ├── text_0.wav
│   └── text_1.wav
└── voice_1/
    ├── text_0.wav
    └── text_1.wav

```

## Note
- When using English text, make sure that the reference audio is in English, when in Chinese, make sure that the reference audio is in Chinese.
