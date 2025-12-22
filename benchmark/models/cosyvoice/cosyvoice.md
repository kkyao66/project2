# CosyVoice3 Batch Audio Generator

This script generates speech audio using **CosyVoice3**, supporting:

- **Zero-shot voice cloning**
- **Instruction-based voice cloning (adjustable tone and emotion)**

It takes:
- a list of texts
- a list of reference voice WAV files
- a list of prompts (either transcripts or tone instructions)

and generates WAV outputs in a structured directory layout.



## Requirements
detailed setup on https://github.com/FunAudioLLM/CosyVoice


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
3. 'prompts.txt': Each line is a prompt used for audio generation
	1. if --instruct is not passed, zero shot requires transcription of reference audio, so each prompt must be the transcription corresponding to the reference audio in the same line
		```
        You are a helpful assistant.<|endofprompt|>The New York daily news, saturday stumper. Saturday, April 19th, 2014. Constructed by Doug Peterson, edited by Stanley Newman
        ```
	2. if --instruct is passed, prompt is used to give an instruction on the tone of the generated audio.
        ```
		You are a helpful assistant. Please read the sentence in a depressing tone.<|endofprompt|>
        ```
        <|endofprompt|> must be put at the end of each prompt
## Usage
### Instruction-based generation (tone and emotion control)

```bash
python cosyvoice_gen.py \
  --texts path/to/text.txt \
  --voices path/to/voices.txt \
  --prompts path/to/instruction_prompt.txt \
  --instruct \
  --output_dir outputs/dir 
  ```

### Zero-shot voice cloning
```bash
python cosyvoice_gen.py \
  --texts path/to/sample_text.txt \
  --voices path/to/sample_voices.txt \
  --prompts path/to/transcribed_prompt.txt \
  --output_dir outputs/dir 
  ```

## Output Structure 
### with --instruct
there will be num_instructions * num_voices * num_texts outputs
```
outputs/dir/
├── prompt_0/
│   ├── voice_0/
│   │   ├── text_0.wav
│   │   └── text_1.wav
│   └── voice_1/
│       ├── text_0.wav
│       └── text_1.wav
└── prompt_1/
    ├── voice_0/
    │   ├── text_0.wav
    │   └── text_1.wav
    └── voice_1/
        ├── text_0.wav
        └── text_1.wav
```
### without --instruct
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
