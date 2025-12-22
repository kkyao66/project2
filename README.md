# Towards Train-Ready Image+Audio→Text Instruction Data: Reproducible TTS Benchmarking,Query Rewriting, and Triplet Construction
````md
# Towards Train-Ready Image+Audio→Text Instruction Data  
## Reproducible TTS Benchmarking, Query Rewriting, and Triplet Construction

This repository builds **train-ready Image+Audio→Text instruction data** by extending an existing **image–text instruction corpus** into a multimodal format with **spoken user prompts**. The pipeline combines **(i) reproducible multilingual TTS benchmarking with benchmark-time model/voice selection**, **(ii) query rewriting into colloquial spoken-style prompts**, and **(iii) large-scale synthesis plus triplet packaging** to produce standardized **training triplets, manifests/metadata, and benchmark tables** suitable for downstream mid-training (e.g., Apertus 1.5).

---

## Pipeline at a Glance

```mermaid
flowchart TB
  A[Source instruction data<br/>(image, user text, assistant text)]

  subgraph P[Parallel components]
    direction LR

    subgraph BA[Branch A: Reproducible TTS benchmarking]
      direction TB
      BA1[Generate benchmark audio<br/>(model × language × voice × text_id)]
      BA2[Score automatic signals<br/>Similarity (Whisper→SBERT) + UTMOSv2]
      BA3[Benchmark-time selection<br/>best model/language/voice map]
      BA1 --> BA2 --> BA3
    end

    subgraph BB[Branch B: Query rewriting (spoken-style prompts)]
      direction TB
      BB1[Rewrite user turns only<br/>(keep assistant answers unchanged)]
      BB2[Preserve meaning<br/>improve colloquial spoken phrasing]
      BB1 --> BB2
    end
  end

  M[Merge: selected configs + rephrased prompts]
  S[Large-scale synthesis<br/>(multi-language, multi-voice)]
  T[Triplet construction + packaging<br/>(image, audio_prompt) → target_text<br/>+ manifests/metadata]

  A --> P
  BA3 --> M
  BB2 --> M
  M --> S --> T


**Inputs**

* `image` (or image reference/path)
* `user_text` (written prompt/instruction/query)
* `assistant_text` (ground-truth target response)

**Outputs**

* **Train-ready triplets**: `(image, audio_prompt) → target_text`
* **Audio files** (WAV, standardized sampling rate as configured)
* **Manifests / metadata** (JSONL/CSV with paths and per-sample attributes)
* **Benchmark artifacts** (tables for model/language/voice comparison and selected configurations)

```
::contentReference[oaicite:0]{index=0}
```
