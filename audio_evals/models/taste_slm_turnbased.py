"""UltraEval-Audio adapter for the trained TASTE-S turn-based SLM.

The adapter deliberately delegates raw-audio preprocessing, causal TASTE extraction,
the ``<lis> U <spk> A <eob>`` grammar, and EOB decoding to
``train_slm/evaluation/turnbased_eval``.  It only bridges that inference authority
to UEA's audio-in/text-out contract, optionally synthesizing the same generated answer for S2S QA.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import sys
import threading
import time
import zlib
from typing import Any

from audio_evals.base import PromptStruct
from audio_evals.models.model import OfflineModel

logger = logging.getLogger(__name__)
ADAPTER_VERSION = "uea_taste_slm_turnbased_adapter_v3"
_BUILD_LOCK = threading.Lock()


def _audio_path_from_prompt(prompt: PromptStruct) -> str:
    paths, texts = [], []
    for turn in prompt:
        for content in turn.get("contents", []):
            if content.get("type") == "audio":
                paths.append(content["value"])
            elif content.get("type") == "text":
                texts.append(content["value"])
    if texts:
        raise ValueError("TurnBasedTasteSLM is audio-in only; canonical benchmark text must not be supplied")
    if len(paths) != 1:
        raise ValueError(f"expected exactly one audio item, got {len(paths)}")
    return paths[0]


def _local_cuda_device(gpu_id: Any) -> str:
    """Translate UEA's physical GPU id into its local CUDA_VISIBLE_DEVICES ordinal."""
    if gpu_id is None:
        return "cuda:0"
    if isinstance(gpu_id, str) and "," in gpu_id:
        gpu_id = gpu_id.split(",", 1)[0].strip()
    gid = int(gpu_id)
    visible = [x.strip() for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
    ids = [int(x) for x in visible if x.lstrip("-").isdigit()]
    return f"cuda:{ids.index(gid) if gid in ids else gid}"


class TurnBasedTasteSLM(OfflineModel):
    """One trained user turn from audio -> one generated agent turn as text and optional audio."""

    def __init__(self, config_path: str, gpu_id=None, sample_params=None, **overrides):
        super().__init__(is_chat=True, sample_params=sample_params)
        self.config_path = os.path.abspath(os.path.expandvars(config_path))
        self.gpu_id = gpu_id
        self.overrides = overrides
        self._cfg: dict[str, Any] | None = None
        self._models = None
        self._torch = None
        self._lock = threading.Lock()
        self._last_generation = None
        self._last_result = None

    def _build(self):
        if self._models is not None:
            return self._models
        with _BUILD_LOCK:
            if self._models is None:
                self._build_locked()
        return self._models

    def _build_locked(self):
        import yaml

        with open(self.config_path, encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if raw.get("adapter") != "taste_s_turnbased":
            raise ValueError(f"{self.config_path}: expected adapter 'taste_s_turnbased'")
        cfg = dict(raw["config"])
        for key, value in list(cfg.items()):
            if key.endswith(("_dir", "_path")) and isinstance(value, str):
                cfg[key] = os.path.abspath(os.path.expandvars(value))
        cfg.update(self.overrides)
        if self.gpu_id is not None:
            cfg["device"] = _local_cuda_device(self.gpu_id)

        root = cfg["turnbased_eval_dir"]
        parent = os.path.dirname(root)
        for path in (root, parent):
            if path not in sys.path:
                sys.path.insert(0, path)
        import torch
        from turnbased_eval.generate import load_models

        self._models = load_models(
            cfg["ckpt_dir"], cfg["taste_s_tokenizer_dir"], cfg["qwen_model_dir"],
            cfg.get("device", "cuda:0"), cfg.get("recipe_path") or None,
        )
        self._cfg, self._torch = cfg, torch
        logger.info("Turn-based TASTE-SLM loaded from %s", cfg["ckpt_dir"])

    def describe(self) -> dict:
        self._build()
        return {"adapter_version": ADAPTER_VERSION, "config_path": self.config_path,
                "effective_config": self._cfg}

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        audio_path = _audio_path_from_prompt(prompt)
        sample_id = os.path.splitext(os.path.basename(audio_path))[0]
        with self._lock:
            models = self._build()
            cfg, torch = self._cfg, self._torch
            started = time.perf_counter()
            seed = zlib.crc32(f"{cfg.get('seed', 42)}:{sample_id}".encode()) & 0x7FFFFFFF
            # Sampling remains per-sample and independent of UEA scheduling order.
            with torch.random.fork_rng(devices=[models.device] if str(models.device).startswith("cuda") else []):
                torch.manual_seed(seed)
                if str(models.device).startswith("cuda"):
                    torch.cuda.manual_seed(seed)
                # Keep the benchmark bridge thin: `run_one()` is the sole TB inference
                # authority and writes the diagnostic dialogue as question + gap + answer.
                from turnbased_eval import generate
                synthesize_audio = bool(cfg.get("synthesize_audio", False))
                artifact_dir = cfg.get("artifact_dir")
                if synthesize_audio:
                    if artifact_dir:
                        os.makedirs(artifact_dir, exist_ok=True)
                    run_dir = tempfile.mkdtemp(prefix=f"taste_tb_{sample_id}_", dir=artifact_dir or None)
                    cleanup = None
                else:
                    cleanup = tempfile.TemporaryDirectory(prefix=f"taste_tb_{sample_id}_")
                    run_dir = cleanup.name
                try:
                    result = generate.run_one(
                        models, audio_path, run_dir,
                        temperature=float(cfg.get("temperature", 0.8)),
                        top_k=int(cfg.get("top_k", 25)), top_p=float(cfg.get("top_p", 1.0)),
                        greedy=bool(cfg.get("greedy", True)), max_words=int(cfg.get("max_words", 80)),
                        eob_logit_bias=float(cfg.get("eob_logit_bias", 0.0)),
                        quantize_input=bool(cfg.get("input_quantization", True)),
                        input_resampler=cfg.get("input_resampler", "torchaudio"),
                        unit_top_k=int(cfg.get("unit_top_k", 25)),
                        unit_top_p=float(cfg.get("unit_top_p", 0.8)),
                        unit_temperature=float(cfg.get("unit_temperature", 1.0)),
                        gap_seconds=float(cfg.get("dialogue_gap_seconds", 0.5)),
                        output_sample_rate=int(cfg.get("output_sample_rate", 24000)),
                        skip_audio=not synthesize_audio,
                    )
                    with open(os.path.join(run_dir, "generation.json"), encoding="utf-8") as handle:
                        generated = json.load(handle)
                    with open(os.path.join(run_dir, "input_tokens.json"), encoding="utf-8") as handle:
                        tokens = json.load(handle)
                finally:
                    if cleanup is not None:
                        cleanup.cleanup()
                text = str(generated["predicted_text"]).strip()
                forced = bool(generated["forced_eob"])
                trace = generated["trace"]
                audio_path_out = os.path.join(run_dir, "agent_generated.wav") if synthesize_audio else None
                self._last_result = result if synthesize_audio else None
                self._last_generation = generated

        out = {
            "text": text,
            "status": "ok" if text else "empty",
            "sample_id": sample_id,
            "audio_path": audio_path,
            "adapter_version": ADAPTER_VERSION,
            "protocol_version": "turnbased_lis_user_spk_agent_eob_v1",
            "finish_reason": "max_words" if forced else "eob",
            "truncated": forced,
            "answer_tokens": len(self._last_generation["text_ids"]),
            "n_predicted_text_tokens": len(self._last_generation["text_ids"]),
            "forced_eob": forced,
            "eob_text_id": int(models.slm.eob_text_id),
            "final_eob_margin": trace[-1].get("eob_margin") if trace else None,
            "prompt_positions": int(self._last_generation["prompt_positions"]),
            "effective_decoding": {
                "text": "greedy" if cfg.get("greedy", True) else "sample",
                "temperature": cfg.get("temperature", 0.8), "top_k": cfg.get("top_k", 25),
                "top_p": cfg.get("top_p", 1.0), "eob_logit_bias": cfg.get("eob_logit_bias", 0.0),
                "sample_seed": seed,
            },
            "audio_meta": result["input"],
            "ctc_transcript": result["ctc_text"],
            "ctc_num_tokens": result["n_input_text_tokens"],
            "tokenizer_windowing": tokens.get("windowing"),
            "elapsed_sec": round(time.perf_counter() - started, 4),
        }
        if bool(cfg.get("synthesize_audio", False)):
            out.update(audio=audio_path_out, audio_synthesis={
                "path": audio_path_out, "sample_rate": int(cfg.get("output_sample_rate", 24000)),
                "native_sample_rate": self._last_result["audio"]["native_vocoder_sample_rate"],
                "n_unit_tokens": self._last_result["audio"]["n_unit_tokens"],
                "voice": "zero_192d_generic", "generation_dir": run_dir,
                "dialogue_path": os.path.join(run_dir, "dialogue_predicted.wav"),
            })
        return json.dumps(out, ensure_ascii=False)
