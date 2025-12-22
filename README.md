# Towards Train-Ready Image+Audio→Text
Instruction Data: Reproducible TTS Benchmarking,
Query Rewriting, and Triplet Construction
````md
# Towards Train-Ready Image+Audio→Text Instruction Data  
## Reproducible TTS Benchmarking, Query Rewriting, and Triplet Construction

This repository builds **train-ready Image+Audio→Text instruction data** by extending an existing **image–text instruction corpus** into a multimodal format with **spoken user prompts**. The pipeline combines **(i) reproducible multilingual TTS benchmarking with benchmark-time model/voice selection**, **(ii) query rewriting into colloquial spoken-style prompts**, and **(iii) large-scale synthesis plus triplet packaging** to produce standardized **training triplets, manifests/metadata, and benchmark tables** suitable for downstream mid-training (e.g., Apertus 1.5).

---

## Pipeline at a Glance

```text
Source instruction data (image, user text, assistant text)
        |
        |------------------------------ Parallel ------------------------------|
        |                                                                       |
        |  Branch A: Reproducible TTS benchmarking                              |
        |    - Generate benchmark audio (model × language × voice × text_id)    |
        |    - Score with automatic signals (Similarity, UTMOSv2)               |
        |    - Benchmark-time selection → best model/language/voice map         |
        |                                                                       |
        |  Branch B: Query rewriting (spoken-style prompts)                     |
        |    - Rewrite user turns only (keep assistant answers unchanged)       |
        |    - Preserve meaning; improve colloquial spoken phrasing             |
        |                                                                       |
        |------------------------------- Merge --------------------------------|
                                |
                                v
Selected TTS configs + rephrased prompts
        |
        v
Large-scale synthesis (multi-language, multi-voice)
        |
        v
Triplet construction + packaging
(image, audio_prompt) → target_text  + manifests/metadata
````

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
