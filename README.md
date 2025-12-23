# Towards Train-Ready Image+Audio→Text Instruction Data  
## Reproducible TTS Benchmarking, Query Rewriting, and Triplet Construction

This repository builds **train-ready Image+Audio→Text instruction data** by extending an existing **image–text instruction corpus** into a multimodal format with **spoken user prompts**. The pipeline combines (i) **reproducible multilingual TTS benchmarking with benchmark-time model/voice selection**, (ii) **query rewriting into colloquial spoken-style prompts**, and (iii) **large-scale synthesis plus triplet packaging** to produce standardized **training triplets, manifests/metadata, and benchmark tables** suitable for downstream mid-training (e.g., Apertus 1.5).

---

## 1) Pipeline at a Glance

```mermaid
flowchart TB
  A["Source data<br/>image + user + assistant"]

  subgraph P["Parallel"]
    direction LR

    subgraph BA["A: TTS benchmark"]
      direction TB
      BA1["Gen audio<br/>m x lang x voice x id"]
      BA2["Score<br/>Sim + UTMOSv2"]
      BA3["Select configs<br/>best m-lang-voice"]
      BA1 --> BA2 --> BA3
    end

    subgraph BB["B: Rephrase"]
      direction TB
      BB1["Rephrase user turns"]
      BB2["Spoken style<br/>meaning preserved"]
      BB1 --> BB2
    end
  end

  M["Merge<br/>configs + rephrased"]
  S["Synthesize at scale<br/>multi-lang, multi-voice"]
  T["Package triplets<br/>(image,audio)->text<br/>+ manifests"]

  A --> P
  BA3 --> M
  BB2 --> M
  M --> S --> T
````

**Inputs**

* `image` (or image reference/path)
* `user_text` (written prompt/instruction/query)
* `assistant_text` (ground-truth target response)

**Outputs**

* **Train-ready triplets**: `(image, audio_prompt) -> target_text`
* **Audio files** (WAV, standardized sampling rate as configured)
* **Manifests / metadata** (JSONL/CSV with paths and per-sample attributes)
* **Benchmark artifacts** (tables for model/language/voice comparison and selected configurations)

## Reproducible Entry Points (Recommended Order)

1. **Setup:** [`benchmark/set up.md`](benchmark/set_up.md)  
2. **Benchmark configs and runners:** [`benchmark/configs/`](benchmark/configs), [`benchmark/models/`](benchmark/models)  
3. **Benchmark summary tables:** [`benchmark/model_comparison.md`](benchmark/model_comparison.md)  
4. **Rephrasing:** [`rephrase/rephrase.md`](rephrase/rephrase.md)  
5. **Generation inputs:** [`generate/input/text/`](generate/input/text)


## 2) Repository Structure

This repository follows a **module-first layout** aligned with the project pipeline:  
**benchmarking → model selection → rephrasing → synthesis → QC/evaluation → packaging**.

## Setup (HPC / Bristen)

Environment setup (Conda, Slurm GPU usage, common CSCS pitfalls) is documented here:
- [`benchmark/set up.md`](benchmark/set_up.md)


### `benchmark/` — Benchmarking, model runners, evaluation, and outputs
- **`benchmark/configs/`** — benchmark and generation configs (language/model/voices/audio format).  
  See: [`benchmark/configs/`](benchmark/configs)

- **`benchmark/input/`** — benchmark prompt sets, translated prompts, and optional voice assets (when applicable).  
  See: [`benchmark/input/`](benchmark/input)

- **`benchmark/models/`** — per-model runners and model-specific guides.  
  Start here if you want to run a specific TTS model:
  - [`benchmark/models/cosyvoice/`](benchmark/models/cosyvoice)
  - [`benchmark/models/chatterbox/`](benchmark/models/chatterbox)
  - [`benchmark/models/fishspeech/`](benchmark/models/fishspeech)
  - [`benchmark/models/higgs-audio/`](benchmark/models/higgs-audio)
  - [`benchmark/models/indextts/`](benchmark/models/indextts)
  - [`benchmark/models/kokoro/`](benchmark/models/kokoro)


### `rephrase/` — Query rephrasing (spoken-style prompts)
This module rewrites **user turns only** into a more colloquial spoken style while keeping assistant answers unchanged.

- **Documentation:** [`rephrase/rephrase.md`](rephrase/rephrase.md)  
- **Dataset dump script:** [`rephrase/dump_clevr_1000.py`](rephrase/dump_clevr_1000.py)  
- **Rephrasing script:** [`rephrase/rephrase_clevr_1000.py`](rephrase/rephrase_clevr_1000.py)  
- **Example artifacts (JSONL):** `rephrase/clevr_first1000_raw.jsonl`, `rephrase/clevr_first1000_rephrased.jsonl`

See: [`rephrase/`](rephrase)

### `generate/` — Large-scale synthesis inputs and generation artifacts
This module stores (and/or produces) inputs for large-scale audio synthesis after selecting the best model per language.

- **TTS input texts (per language):** [`generate/input/text/`](generate/input/text)  
  - `tts_inputs_en.jsonl`, `tts_inputs_zh.jsonl`, `tts_inputs_ja.jsonl`, `tts_inputs_fr.jsonl`, `tts_inputs_de.jsonl`

See: [`generate/`](generate)

> **How to run a model:** open the corresponding folder under `benchmark/models/<model>/` and follow its guide.  
> **How to rephrase prompts:** follow [`rephrase/rephrase.md`](rephrase/rephrase.md).  
> **How to locate synthesis inputs:** see [`generate/input/text/`](generate/input/text).

## Where to Start (Documentation Map)

- **Benchmark and compare models:** start at [`benchmark/`](benchmark), then see [`benchmark/eval/`](benchmark/eval) for QC/aggregation.
- **Rephrase user queries (spoken-style):** [`rephrase/rephrase.md`](rephrase/rephrase.md)
- **Run a specific TTS model:** pick a folder under [`benchmark/models/`](benchmark/models)
- **Use generation inputs (per language JSONL):** [`generate/input/text/`](generate/input/text)

## 3)TTS Benchmarking (Model Selection for Scalable Synthesis)

We benchmark multiple open-source TTS models under multilingual and multi-voice settings to select robust model–language configurations before large-scale synthesis. Selection is performed at benchmark time using a small set of human-labeled samples to calibrate an automatic quality gate, combining two automatic signals (Whisper→SBERT similarity and UTMOSv2) with a lightweight logistic-regression selector, rather than relying on post-hoc filtering of the final dataset.

**Automatic signals**
- **Audio accuracy:** Whisper ASR transcription → SBERT semantic similarity to the original prompt
- **Naturalness/quality:** UTMOSv2 MOS predictor score

### Selected model per language (used for large-scale generation)

| Language | Selected TTS model |
|---|---|
| English | [CosyVoice](benchmark/models/cosyvoice/) |
| French  | [CosyVoice](benchmark/models/cosyvoice/) |
| Japanese | [Chatterbox](benchmark/models/chatterbox/) |
| German | [Chatterbox](benchmark/models/chatterbox/) |
| Chinese | [Index-TTS](benchmark/models/indextts/) |

Full benchmark tables and comparisons: [`benchmark/model_comparison.md`](benchmark/model_comparison.md)

**Reproducibility (navigation)**
- Benchmark configs: [`benchmark/configs/`](benchmark/configs/)
- Per-model runners and docs& QC + aggregation outputs: [`benchmark/models/`](benchmark/models/)

## Query Rephrasing (Spoken-Style Prompts)

Many instruction datasets contain user prompts written in a formal or templated style, which can lead to unnatural prosody when directly synthesized by TTS. We therefore **rephrase user turns only** into a more colloquial, spoken style while **preserving meaning** and keeping **assistant answers unchanged**.

- **Data source:** the **CLEVR** configuration is streamed from the Hugging Face dataset
  [`mvp-lab/LLaVA-OneVision-1.5-Instruct-Data`](https://huggingface.co/datasets/mvp-lab/LLaVA-OneVision-1.5-Instruct-Data)
  and dumped deterministically (first **N** samples).
- **Model:** `Qwen/Qwen2.5-7B-Instruct` (Transformers), deterministic decoding.
- **Artifacts:** `*_raw.jsonl` and `*_rephrased.jsonl` with identical structure; only user text is rewritten.

Reproduce: [`rephrase/rephrase.md`](rephrase/rephrase.md)  
Scripts: [`rephrase/dump_clevr_1000.py`](rephrase/dump_clevr_1000.py), [`rephrase/rephrase_clevr_1000.py`](rephrase/rephrase_clevr_1000.py)

## Large-Scale Generation (Synthesis) and Triplet Construction

This stage merges:
1) the **benchmark-selected TTS configuration** (best model per language), and  
2) the **rephrased spoken-style prompts**,  
to synthesize audio prompts at scale and package train-ready **(image, audio_prompt) → target_text** triplets with manifests.

### Inputs
- **Per-language TTS input JSONL:** [`generate/input/text/`](generate/input/text)  
  (`tts_inputs_en.jsonl`, `tts_inputs_fr.jsonl`, `tts_inputs_de.jsonl`, `tts_inputs_ja.jsonl`, `tts_inputs_zh.jsonl`)
- **Rephrased prompts (if used as source for building TTS inputs):** [`rephrase/`](rephrase)

### Selected TTS backends (from benchmarking)
Generation uses the best-performing model per language:

- **English / French:** [CosyVoice](benchmark/models/cosyvoice/)  
- **Japanese / German:** [Chatterbox](benchmark/models/chatterbox/)  
- **Chinese:** [Index-TTS](benchmark/models/indextts/)  

Full benchmark tables and comparisons: [`benchmark/model_comparison.md`](benchmark/model_comparison.md)

### Where to run (model docs + runner scripts)
Each selected backend is executed via its model-specific runner under `benchmark/models/<model>/`.

- **CosyVoice**
  - Docs: [`benchmark/models/cosyvoice/cosyvoice.md`](benchmark/models/cosyvoice/cosyvoice.md)
  - Runner: [`benchmark/models/cosyvoice/cosyvoice_gen.py`](benchmark/models/cosyvoice/cosyvoice_gen.py)

- **Chatterbox**
  - Docs: [`benchmark/models/chatterbox/chatterbox.md`](benchmark/models/chatterbox/chatterbox.md)
  - Runner: [`benchmark/models/chatterbox/gen_chatterbox.py`](benchmark/models/chatterbox/gen_chatterbox.py)

- **Index-TTS**
  - Docs: [`benchmark/models/indextts/indextts.md`](benchmark/models/indextts/indextts.md)
  - Runner: [`benchmark/models/indextts/gen_indextts.py`](benchmark/models/indextts/gen_indextts.py)

### Outputs
Generated audio and manifests follow a standardized layout:
- WAV audio files (organized by language / voice)
- `metadata_*.jsonl` and `failed_*.jsonl`
- logs for reproducibility

Benchmark outputs live under: [`benchmark/output/`](benchmark/output)
**Final triplet dataset (Hugging Face):** [[TBD](<HF_DATASET_URL>)](https://huggingface.co/datasets/kkkyao/triplets_audio_image_text_v1)

### Triplet Dataset Summary (Train-Ready Artifacts)

**Triplet format**
- Each example is packaged as: **`(image, audio_prompt) → target_text`**
- `target_text` is copied verbatim from the source instruction dataset (assistant response); only the user prompt is converted into speech.

**Language & voice expansion**
- **5 languages:** Chinese (zh), English (en), Japanese (ja), French (fr), German (de)
- **Up to 5 voices per configuration**
  - If a backend supports **voice cloning**, we synthesize multiple speaker styles using a shared set of reference voices.
  - Otherwise, we use the model’s **native speaker presets** (when available).

**Benchmark-driven selection**
- For large-scale synthesis, we use the **per-language best backend** selected from the benchmarking stage (see the table in *TTS Benchmarking* above), rather than synthesizing with all candidate models.

## References and Citations

This repository builds on prior work and third-party tools/datasets. Please cite the original sources when appropriate.  
Full BibTeX entries are provided here: [`docs/references.bib`](docs/references.bib)

### Key References (Papers)

- Visual Instruction Tuning (LLaVA): [`arXiv:2304.08485`](https://arxiv.org/abs/2304.08485)
- Scaling Speech-Text Pre-training with Synthetic Interleaved Data: [`arXiv:2411.17607`](https://arxiv.org/abs/2411.17607)
- Qwen2.5 Technical Report: [`arXiv:2412.15115`](https://arxiv.org/abs/2412.15115)
- Whisper (Robust Speech Recognition): [`ICML 2023`](https://proceedings.mlr.press/v202/radford23a.html)
- Sentence-BERT: [`arXiv:1908.10084`](https://arxiv.org/abs/1908.10084)

### Tools and Libraries Referenced

- Sentence-Transformers (SBERT similarity): [`Documentation`](https://sbert.net/)
- UTMOSv2 (MOS prediction for TTS): [`GitHub`](https://github.com/sarulab-speech/UTMOSv2)
- UTMOS / VoiceMOS Challenge system description: [`arXiv:2204.02152`](https://arxiv.org/abs/2204.02152)

### External Datasets (Hugging Face)

- Instruction dataset (CLEVR configuration):  
  [`mvp-lab/LLaVA-OneVision-1.5-Instruct-Data`](https://huggingface.co/datasets/mvp-lab/LLaVA-OneVision-1.5-Instruct-Data)

- Speech/voice datasets used for reference voice assets and/or auxiliary examples:
  - [`i4ds/spc_r`](https://huggingface.co/datasets/i4ds/spc_r)
  - [`BrunoHays/multilingual-TEDX-fr`](https://huggingface.co/datasets/BrunoHays/multilingual-TEDX-fr)
  - [`google/fleurs`](https://huggingface.co/datasets/google/fleurs)

### Community Resources (Candidate Discovery)

- TTS Arena V2 (HF Space): [`TTS-AGI/TTS-Arena-V2`](https://huggingface.co/spaces/TTS-AGI/TTS-Arena-V2)
- Open Source TTS Gallery (HF Space): [`Inferless/Open-Source-TTS-Gallary`](https://huggingface.co/spaces/Inferless/Open-Source-TTS-Gallary)

### Licensing, Voice Assets, and Intended Use

- **Licensing:** Datasets, model weights, and third-party code used by this project are subject to their **original licenses**. Users are responsible for ensuring compliance with all upstream terms when reproducing results or redistributing artifacts.
- **Reference voice assets:** To reduce privacy/impersonation risk, **reference voice recordings used for voice cloning are not included in this repository** (and should not be redistributed unless you have explicit rights to do so).
- **Intended use:** Outputs are intended for **research and training-data construction** in an *Image+Audio→Text* setting (e.g., mid-training / alignment of multimodal instruction-following models). This pipeline is **not intended** for generating deceptive or identity-impersonating audio.
