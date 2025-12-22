#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate benchmark wavs for multiple TTS models/languages/voices from a single JSON config.

Outputs:
  - {output.audio_dir}/{model}/{lang}/{voice}/s{sid:04d}.wav
  - {output.metadata_path} (JSONL; one record per attempted sample)

Design goals:
  - deterministic / reproducible
  - resume-friendly
  - metadata-first (downstream eval/QC depends on it)

Important environment note:
  - This script intentionally avoids torchaudio.save() because some clusters ship torchaudio
    builds that route saving through torchcodec/ffmpeg, which may be unavailable.
  - We write WAV PCM16 using Python stdlib (wave) + numpy.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import traceback
import io  # <-- added (needed for HTTP FishSpeechBackend)
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests  # <-- added (needed for HTTP FishSpeechBackend)


# -------------------------
# Utilities
# -------------------------

def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

def safe_str(x: Any) -> str:
    return str(x).replace("/", "_")

def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"Invalid JSON on line {i} of {path}: {e}")
    return rows

def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def load_done_keys(metadata_path: str) -> set[Tuple[str, str, str, int]]:
    """
    Resume support: return keys that are already successfully generated (status == "ok").
    We only treat "ok" as done to avoid skipping failed attempts.
    """
    done: set[Tuple[str, str, str, int]] = set()
    if not os.path.exists(metadata_path):
        return done
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if obj.get("status") != "ok":
                    continue
                k = (obj.get("model"), obj.get("language"), obj.get("voice"), int(obj.get("sid")))
                if all(k):
                    done.add(k)  # type: ignore[arg-type]
            except Exception:
                continue
    return done

def atomic_write_wav(path: Path, write_fn) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}.{int(time.time())}")
    try:
        write_fn(tmp)
        ensure_dir(path.parent)
        os.replace(str(tmp), str(path))
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass

def save_wav_pcm16_mono(path: Path, wav: Any, sample_rate: int) -> Tuple[int, int]:
    """
    Save mono WAV PCM16 using stdlib wave.

    Accepts:
      - torch.Tensor [T] or [1,T] or [C,T]
      - numpy-like array

    Returns:
      (n_samples, sample_rate)
    """
    import wave
    import numpy as np

    # torch -> numpy
    try:
        import torch
        if isinstance(wav, torch.Tensor):
            wav = wav.detach().cpu()
            if wav.dim() == 2:
                wav = wav[0]  # take first channel
            wav = wav.numpy()
    except Exception:
        pass

    arr = np.asarray(wav).reshape(-1).astype(np.float32, copy=False)
    arr = np.clip(arr, -1.0, 1.0)
    pcm16 = (arr * 32767.0).astype(np.int16)

    def _write(dst: Path) -> None:
        ensure_dir(dst.parent)
        with wave.open(str(dst), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(int(sample_rate))
            wf.writeframes(pcm16.tobytes())

    atomic_write_wav(path, _write)
    return int(pcm16.shape[0]), int(sample_rate)

def try_resample(wav: Any, sr_in: int, sr_out: int) -> Any:
    """
    Resample using torchaudio if available. If torchaudio is unavailable, return input as-is.
    """
    if int(sr_in) == int(sr_out):
        return wav
    try:
        import torch
        import torchaudio
        if not isinstance(wav, torch.Tensor):
            wav = torch.tensor(wav)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        wav = torchaudio.functional.resample(wav.to(torch.float32), int(sr_in), int(sr_out))
        return wav
    except Exception:
        # Last resort: no resampling
        return wav


# -------------------------
# Backends
# -------------------------

@dataclass
class AudioResult:
    wav: Any            # torch.Tensor [1,T] or numpy-like
    sample_rate: int

class ChatTTSBackend:
    """
    ChatTTS backend (robust minimal path):
      - sys.path += chattts_root (the directory that contains package 'ChatTTS')
      - ChatTTS.Chat()
      - chat.load_models() if present
      - chat.infer([text], params_refine_text=..., params_infer_code=...)

    Voice control:
      - deterministic via torch.manual_seed(voice_seed)
      - speaker embedding sampled via chat.sample_random_speaker() if available
      - also sets InferCodeParams.manual_seed if supported
    """

    def __init__(self, chattts_root: Optional[str], device: str = "cuda"):
        self.chattts_root = chattts_root
        self.device = device
        self._chat = None
        self._ChatTTS = None

    def _lazy_init(self):
        if self._chat is not None:
            return

        if self.chattts_root:
            sys.path.insert(0, os.path.expanduser(self.chattts_root))

        import ChatTTS  # type: ignore
        self._ChatTTS = ChatTTS

        if not hasattr(ChatTTS, "Chat"):
            raise RuntimeError("ChatTTS module has no attribute 'Chat'. Check --chattts_root path.")

        chat = ChatTTS.Chat()

        # load models (forks differ)
        if hasattr(chat, "load_models"):
            chat.load_models()
        elif hasattr(chat, "load"):
            chat.load()
        else:
            # If neither exists, still try infer (some forks auto-load on first call)
            pass

        self._chat = chat

    @staticmethod
    def _norm_keys(d: Dict[str, Any]) -> Dict[str, Any]:
        dd = dict(d)
        # some configs use top_p/top_k, but ChatTTS uses top_P/top_K
        if "top_p" in dd and "top_P" not in dd:
            dd["top_P"] = dd.pop("top_p")
        if "top_k" in dd and "top_K" not in dd:
            dd["top_K"] = dd.pop("top_k")
        return dd

    def synthesize(
        self,
        text: str,
        lang: str,
        voice_seed: int,
        out_sr: int,
        inference_cfg: Dict[str, Any],
    ) -> AudioResult:
        self._lazy_init()
        import torch

        chat = self._chat
        ChatTTS = self._ChatTTS

        # Determinism
        torch.manual_seed(int(voice_seed))

        # Build params objects if available
        params_refine_text = None
        params_infer_code = None

        # Optional: allow user config to override params (if present)
        refine_over = self._norm_keys(inference_cfg.get("params_refine_text", {})) if isinstance(inference_cfg.get("params_refine_text", {}), dict) else {}
        infer_over  = self._norm_keys(inference_cfg.get("params_infer_code", {}))  if isinstance(inference_cfg.get("params_infer_code", {}), dict) else {}

        # Silence tqdm to keep logs manageable on cluster
        refine_over.setdefault("show_tqdm", False)
        infer_over.setdefault("show_tqdm", False)

        # Use benchmark seed as "content seed", and voice_seed as "voice seed"
        base_seed = inference_cfg.get("seed", None)
        if base_seed is not None and "manual_seed" not in infer_over:
            # keep voice_seed as primary seed for speaker; incorporate base_seed to reduce collisions
            infer_over["manual_seed"] = int(voice_seed) ^ int(base_seed)

        # Speaker embedding
        spk_emb = None
        if hasattr(chat, "sample_random_speaker"):
            try:
                spk_emb = chat.sample_random_speaker()
            except Exception:
                spk_emb = None
        if spk_emb is not None:
            infer_over.setdefault("spk_emb", spk_emb)

        # Instantiate dataclass params if present
        if hasattr(ChatTTS.Chat, "RefineTextParams") and isinstance(refine_over, dict):
            try:
                params_refine_text = ChatTTS.Chat.RefineTextParams(**refine_over)
            except Exception:
                params_refine_text = None
        if hasattr(ChatTTS.Chat, "InferCodeParams") and isinstance(infer_over, dict):
            try:
                params_infer_code = ChatTTS.Chat.InferCodeParams(**infer_over)
            except Exception:
                params_infer_code = None

        # Infer (try a few kw patterns)
        last_err = None
        wav = None

        for _ in range(3):
            try:
                if params_refine_text is not None or params_infer_code is not None:
                    wavs = chat.infer(
                        [text],
                        params_refine_text=params_refine_text,
                        params_infer_code=params_infer_code,
                    )
                else:
                    wavs = chat.infer([text])
                wav = wavs[0]
                break
            except TypeError as e:
                # Some forks don't accept params_refine_text; try only infer code
                last_err = e
                try:
                    wavs = chat.infer([text], params_infer_code=params_infer_code)
                    wav = wavs[0]
                    break
                except Exception as e2:
                    last_err = e2
                    wav = None
            except Exception as e:
                last_err = e
                wav = None

        if wav is None:
            raise RuntimeError(f"ChatTTS synthesize failed: {last_err}")

        # Detect sr
        sr = getattr(chat, "sample_rate", None) or getattr(chat, "sr", None) or out_sr
        sr = int(sr)

        # Normalize tensor shape
        if not isinstance(wav, torch.Tensor):
            wav = torch.tensor(wav)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        if sr != out_sr:
            wav = try_resample(wav, sr, out_sr)
            sr = out_sr

        return AudioResult(wav=wav, sample_rate=sr)


class FishSpeechBackend:
    """
    FishSpeech backend (HTTP server, same logic as scripts/batch_generate_fishspeech.py).

    Default:
      host=http://127.0.0.1:7860
      endpoint=/v1/tts

    It assumes the FishSpeech server is running on the SAME node (127.0.0.1).
    """

    def __init__(self, device: str = "cuda", host: Optional[str] = None, endpoint: Optional[str] = None):
        self.device = device
        self.host = (host or os.environ.get("FISHSPEECH_HOST", "http://127.0.0.1:7860")).rstrip("/")
        self.endpoint = (endpoint or os.environ.get("FISHSPEECH_ENDPOINT", "/v1/tts"))

    def synthesize(
        self,
        text: str,
        lang: str,
        reference_id: str,
        out_sr: int,
        inference_cfg: Dict[str, Any],
    ) -> AudioResult:
        import numpy as np
        import wave

        # Allow per-model config override via inference_cfg
        host = str(inference_cfg.get("host", self.host)).rstrip("/")
        endpoint = str(inference_cfg.get("endpoint", self.endpoint))
        url = f"{host}{endpoint}"

        payload = {"text": text, "format": "wav", "reference_id": str(reference_id)}

        # Avoid proxy hijacking (you already saw HTML responses without no_proxy)
        r = requests.post(
            url,
            json=payload,
            proxies={"http": None, "https": None},
            timeout=600,
        )
        r.raise_for_status()

        data = r.content
        if len(data) < 16 or data[:4] != b"RIFF":
            raise RuntimeError(f"FishSpeech server did not return RIFF wav. First bytes: {data[:64]!r}")

        with wave.open(io.BytesIO(data), "rb") as wf:
            sr = int(wf.getframerate())
            n = int(wf.getnframes())
            sampwidth = int(wf.getsampwidth())
            channels = int(wf.getnchannels())
            raw = wf.readframes(n)

        if sampwidth != 2:
            raise RuntimeError(f"FishSpeech server produced sampwidth={sampwidth}, expected 16-bit PCM wav.")

        audio = np.frombuffer(raw, dtype=np.int16)
        if channels > 1:
            audio = audio.reshape(-1, channels)[:, 0]

        wav = (audio.astype(np.float32) / 32767.0)[None, :]  # [1,T]

        if sr != int(out_sr):
            wav = try_resample(wav, sr, int(out_sr))
            sr = int(out_sr)

        return AudioResult(wav=wav, sample_rate=sr)


# -------------------------
# Main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to benchmark config JSON")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing wavs")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of sids per language (0 = no limit)")
    ap.add_argument("--only_models", default="", help="Comma-separated allowlist, e.g. chattts or chattts,fishspeech")
    ap.add_argument("--chattts_root", default=os.environ.get("CHATTTS_ROOT", ""), help="ChatTTS repo root (contains package ChatTTS/)")
    ap.add_argument("--device", default="cuda", help="cuda or cpu")
    args = ap.parse_args()

    cfg = read_json(args.config)

    benchmark_id = cfg.get("benchmark_id", "benchmark_v1")
    texts_map: Dict[str, str] = cfg["texts"]
    audio_cfg: Dict[str, Any] = cfg.get("audio", {})
    out_cfg: Dict[str, str] = cfg.get("output", {})

    sample_rate = int(audio_cfg.get("sample_rate", 24000))
    audio_format = audio_cfg.get("format", "wav_pcm16")

    audio_dir = out_cfg.get("audio_dir", f"{benchmark_id}/audio")
    metadata_path = out_cfg.get("metadata_path", f"{benchmark_id}/metadata.jsonl")
    logs_dir = out_cfg.get("logs_dir", f"{benchmark_id}/logs")

    ensure_dir(audio_dir)
    ensure_dir(Path(metadata_path).parent)
    ensure_dir(logs_dir)

    allow = set()
    if args.only_models.strip():
        allow = {x.strip().lower() for x in args.only_models.split(",") if x.strip()}

    done_keys = load_done_keys(metadata_path)

    models_cfg: Dict[str, Any] = cfg["models"]

    # Read texts
    lang_rows: Dict[str, List[Dict[str, Any]]] = {}
    for lang, path in texts_map.items():
        rows = read_jsonl(path)
        if args.limit and args.limit > 0:
            rows = rows[: args.limit]
        for r in rows:
            if "sid" not in r or "text" not in r:
                raise SystemExit(f"Bad JSONL format in {path}. Need {{'sid','text'}} per line.")
        lang_rows[lang] = rows

    print(f"[{now_ts()}] benchmark_id={benchmark_id}")
    print(f"[{now_ts()}] sample_rate={sample_rate} format={audio_format}")
    print(f"[{now_ts()}] audio_dir={audio_dir}")
    print(f"[{now_ts()}] metadata_path={metadata_path}")

    chattts_backend: Optional[ChatTTSBackend] = None
    fish_backend: Optional[FishSpeechBackend] = None

    start_time = time.time()

    for model_name, mcfg in models_cfg.items():
        mname = str(model_name).lower()
        if allow and mname not in allow:
            continue

        model_langs = list(mcfg.get("languages", []))
        voices = list(mcfg.get("voices", []))
        voice_mode = mcfg.get("voice_mode", "unknown")
        infer_cfg = dict(mcfg.get("inference", {})) if isinstance(mcfg.get("inference", {}), dict) else {}

        # also allow params_* under model root
        if "params_refine_text" in mcfg and "params_refine_text" not in infer_cfg:
            infer_cfg["params_refine_text"] = mcfg.get("params_refine_text")
        if "params_infer_code" in mcfg and "params_infer_code" not in infer_cfg:
            infer_cfg["params_infer_code"] = mcfg.get("params_infer_code")

        for lang in model_langs:
            if lang not in lang_rows:
                print(f"[WARN] No texts for lang={lang} in config.texts; skip model={model_name}")
                continue

            rows = lang_rows[lang]

            for voice in voices:
                voice_str = safe_str(voice)
                out_subdir = Path(audio_dir) / mname / lang / voice_str
                ensure_dir(out_subdir)

                for r in rows:
                    sid = int(r["sid"])
                    text = str(r["text"]).strip()

                    key = (mname, lang, voice_str, sid)
                    out_wav = out_subdir / f"s{sid:04d}.wav"

                    # resume
                    if (not args.overwrite) and out_wav.exists() and key in done_keys:
                        continue

                    meta_base = {
                        "ts": now_ts(),
                        "benchmark_id": benchmark_id,
                        "sid": sid,
                        "model": mname,
                        "language": lang,
                        "voice": voice_str,
                        "voice_mode": voice_mode,
                        "text": text,
                        "text_path": texts_map.get(lang, ""),
                        "audio_path": str(out_wav),
                        "audio": {"sample_rate": sample_rate, "format": audio_format},
                        "inference": infer_cfg,
                    }

                    try:
                        t0 = time.time()

                        if mname == "chattts":
                            if chattts_backend is None:
                                chattts_backend = ChatTTSBackend(
                                    chattts_root=args.chattts_root or None,
                                    device=args.device,
                                )
                            voice_seed = int(voice)
                            res = chattts_backend.synthesize(
                                text=text,
                                lang=lang,
                                voice_seed=voice_seed,
                                out_sr=sample_rate,
                                inference_cfg=infer_cfg,
                            )

                        elif mname == "fishspeech":
                            if fish_backend is None:
                                fish_backend = FishSpeechBackend(device=args.device)
                            reference_id = str(voice)
                            res = fish_backend.synthesize(
                                text=text,
                                lang=lang,
                                reference_id=reference_id,
                                out_sr=sample_rate,
                                inference_cfg=infer_cfg,
                            )

                        else:
                            raise RuntimeError(f"Unknown model '{model_name}' in config.")

                        wav = res.wav
                        sr = int(res.sample_rate)

                        # normalize shape and save
                        try:
                            import torch
                            if isinstance(wav, torch.Tensor):
                                if wav.dim() == 2:
                                    wav_mono = wav[0]
                                elif wav.dim() == 1:
                                    wav_mono = wav
                                else:
                                    raise ValueError(f"Unexpected ChatTTS wav dim={wav.dim()}")
                                n_samples = int(wav_mono.shape[-1])
                                dur = float(n_samples) / float(sr)
                                save_wav_pcm16_mono(out_wav, wav_mono, sr)
                            else:
                                # numpy-like [T] or [1,T]
                                import numpy as np
                                arr = np.asarray(wav)
                                if arr.ndim == 2:
                                    arr = arr[0]
                                n_samples = int(arr.shape[-1])
                                dur = float(n_samples) / float(sr)
                                save_wav_pcm16_mono(out_wav, arr, sr)
                        except Exception:
                            # fallback: try direct
                            n_samples, _ = save_wav_pcm16_mono(out_wav, wav, sr)
                            dur = float(n_samples) / float(sr)

                        t1 = time.time()
                        meta = dict(meta_base)
                        meta.update(
                            {
                                "status": "ok",
                                "duration_sec": float(dur),
                                "n_samples": int(n_samples),
                                "rtf": (t1 - t0) / max(dur, 1e-6),
                            }
                        )
                        append_jsonl(metadata_path, meta)

                    except Exception as e:
                        meta = dict(meta_base)
                        meta.update(
                            {
                                "status": "error",
                                "error": repr(e),
                                "traceback": traceback.format_exc(limit=50),
                            }
                        )
                        append_jsonl(metadata_path, meta)
                        continue

    elapsed = time.time() - start_time
    print(f"[{now_ts()}] DONE in {elapsed/60:.1f} min. metadata={metadata_path}")


if __name__ == "__main__":
    main()
