# Towards Train-Ready Image+Audio→Text Instruction Data  
## Reproducible TTS Benchmarking, Query Rewriting, and Triplet Construction

This repository builds **train-ready Image+Audio→Text instruction data** by extending an existing **image–text instruction corpus** into a multimodal format with **spoken user prompts**. The pipeline combines (i) **reproducible multilingual TTS benchmarking with benchmark-time model/voice selection**, (ii) **query rewriting into colloquial spoken-style prompts**, and (iii) **large-scale synthesis plus triplet packaging** to produce standardized **training triplets, manifests/metadata, and benchmark tables** suitable for downstream mid-training (e.g., Apertus 1.5).

---

## Pipeline at a Glance

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

## 3) Repository Structure

This repository follows a **module-first layout** aligned with the project pipeline:  
**benchmarking → model selection → rephrasing → synthesis → QC/evaluation → packaging**.

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

- **`benchmark/output/`** — generated audio (WAV), logs, and manifests (`metadata_*.jsonl`, `failed_*.jsonl`).  
  See: [`benchmark/output/`](benchmark/output)

- **`benchmark/eval/`** — QC and evaluation (Whisper→SBERT similarity, UTMOSv2, human labels, selector outputs, aggregated tables).  
  See: [`benchmark/eval/`](benchmark/eval)

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

## TTS Benchmarking (Model Selection for Scalable Synthesis)

We benchmark multiple open-source TTS models under multilingual and multi-voice settings to select **robust model–language configurations** *before* large-scale synthesis. The benchmark is used for **benchmark-time selection** (choosing what to generate with), rather than post-hoc filtering of the final dataset.

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
- Per-model runners and docs: [`benchmark/models/`](benchmark/models/)
- QC + aggregation outputs: [`benchmark/eval/`](benchmark/eval/)
