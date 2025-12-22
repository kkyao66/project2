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

````

