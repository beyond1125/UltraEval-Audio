"""UltraEval-Audio adapter for the trained TASTE-S turn-based SLM.

The adapter deliberately delegates raw-audio preprocessing, causal TASTE extraction,
the ``<lis> U <spk> A <eob>`` grammar, and EOB decoding to
``train_slm/evaluation/turnbased_eval``.  It only bridges that inference authority
to UEA's audio-in/text-out contract.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import zlib
from typing import Any

from audio_evals.base import PromptStruct
from audio_evals.models.model import OfflineModel

logger = logging.getLogger(__name__)
ADAPTER_VERSION = "uea_taste_slm_turnbased_adapter_v1"
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
    """One trained user turn from audio -> one generated agent turn as text."""

    def __init__(self, config_path: str, gpu_id=None, sample_params=None, **overrides):
        super().__init__(is_chat=True, sample_params=sample_params)
        self.config_path = os.path.abspath(os.path.expandvars(config_path))
        self.gpu_id = gpu_id
        self.overrides = overrides
        self._cfg: dict[str, Any] | None = None
        self._models = None
        self._torch = None
        self._lock = threading.Lock()

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
            from turnbased_eval.core.audio import load_user_audio
            from turnbased_eval.core.tokenizer import tokenize_user_audio
            from turnbased_eval.generate import generate_turn

            started = time.perf_counter()
            wav, audio_meta = load_user_audio(
                audio_path,
                quantize=bool(cfg.get("input_quantization", True)),
                resampler=cfg.get("input_resampler", "torchaudio"),
            )
            user, tokens = tokenize_user_audio(
                models.taste_extractor, wav, models.slm.num_codebooks, models.text_tokenizer,
            )
            seed = zlib.crc32(f"{cfg.get('seed', 42)}:{sample_id}".encode()) & 0x7FFFFFFF
            # Sampling is optional, but must stay stable across evaluation order when enabled.
            with torch.random.fork_rng(devices=[models.device] if str(models.device).startswith("cuda") else []):
                torch.manual_seed(seed)
                if str(models.device).startswith("cuda"):
                    torch.cuda.manual_seed(seed)
                generated = generate_turn(
                    models.slm, [], user,
                    int(models.cfg.data_config.get("pad_text_id", 0)), models.text_tokenizer,
                    models.device, float(cfg.get("temperature", 0.8)), int(cfg.get("top_k", 25)),
                    float(cfg.get("top_p", 1.0)), bool(cfg.get("greedy", True)),
                    int(cfg.get("max_words", 80)), float(cfg.get("eob_logit_bias", 0.0)),
                )

        text = str(generated["text"]).strip()
        forced = bool(generated["forced_eob"])
        trace = generated["trace"]
        out = {
            "text": text,
            "status": "ok" if text else "empty",
            "sample_id": sample_id,
            "audio_path": audio_path,
            "adapter_version": ADAPTER_VERSION,
            "protocol_version": "turnbased_lis_user_spk_agent_eob_v1",
            "finish_reason": "max_words" if forced else "eob",
            "truncated": forced,
            "answer_tokens": len(generated["text_ids"]),
            "forced_eob": forced,
            "eob_text_id": int(models.slm.eob_text_id),
            "final_eob_margin": trace[-1].get("eob_margin") if trace else None,
            "prompt_positions": int(generated["prompt_positions"]),
            "effective_decoding": {
                "text": "greedy" if cfg.get("greedy", True) else "sample",
                "temperature": cfg.get("temperature", 0.8), "top_k": cfg.get("top_k", 25),
                "top_p": cfg.get("top_p", 1.0), "eob_logit_bias": cfg.get("eob_logit_bias", 0.0),
                "sample_seed": seed,
            },
            "audio_meta": audio_meta,
            "ctc_transcript": tokens["ctc_text"],
            "ctc_num_tokens": tokens["n_text_tokens"],
            "tokenizer_windowing": tokens["windowing"],
            "elapsed_sec": round(time.perf_counter() - started, 4),
        }
        return json.dumps(out, ensure_ascii=False)
