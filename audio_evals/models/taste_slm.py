"""
UltraEval-Audio model adapter for the joint TASTE-S SLM (TASTE-SLM).

ADDED BY US. This file is an ADAPTER, not an upstream bug fix. It is the only new
model class in this checkout; see PATCHES.md for the (separate) mechanical fixes.

What it does and does not do
----------------------------
* The model receives ONLY the question audio (UEA's `direct-aqa` prompt carries a single
  `audio` content and no text). The benchmark's canonical `Questions`/`QuestionText`
  column and the `Answer` reference are never read here.
* It calls the REAL TASTE-SLM free-running joint rollout
  (`eval/adapters/taste_s_speech.py::TasteSSpeechAdapter.generate`, protocol
  `spoken_qa_oracle_end_v3`). There is no CTC-to-original-Qwen substitution, no oracle
  transcript, no reference conditioning and no TTS stand-in.
* Checkpoint-compatible conditioning is preserved: the delay and the speech branch come
  from the checkpoint's own `training_config.yaml` (`speech_token_delay: 5`,
  `fusion.speech_gate: 1.0`) and `speech_feedback: sample` keeps the trained joint
  text+speech rollout running even though we only report the TEXT stream. Reporting S2T
  does not switch the speech branch off.

Output contract
---------------
Returns a JSON STRING so UEA's `extract_text` (`JsonExtract(extract_key="text")`) works.
This matters for purely numeric answers: `JsonExtract` does `json.loads(answer)` first, so
a bare `"42"` would parse to the int `42` and then `d["text"]` raises `TypeError`. Emitting
`{"text": "42", ...}` makes numeric answers survive post-processing unchanged.

Extra keys carry diagnostics; `extract_text` ignores them, and UEA's recorder stores the
whole raw string under `type: "inference"`, so nothing is lost.
"""
import json
import logging
import os
import sys
import threading
from typing import Dict

from audio_evals.base import PromptStruct
from audio_evals.models.model import OfflineModel

logger = logging.getLogger(__name__)

ADAPTER_VERSION = "uea_taste_slm_adapter_v1"

# transformers 5.x resolves submodules lazily, and that resolution is NOT thread-safe. With
# IsolatedModelPool the instances are built lazily inside worker threads, so several threads hit
# the first `from transformers import AutoTokenizer` at once and one of them observes a
# half-initialised module ("cannot import name 'AutoTokenizer' from 'transformers'"). Warming the
# import here, at module-import time, happens once on the main thread before any pool thread runs.
from transformers import AutoTokenizer as _WARM_AutoTokenizer  # noqa: F401,E402

# Model construction is additionally serialised across ALL instances: building two 9B checkpoints
# concurrently gives no speedup (it is disk- and CPU-bound) and multiplies peak host memory.
_BUILD_LOCK = threading.Lock()


def _audio_path_from_prompt(prompt: PromptStruct) -> str:
    """Pull the single audio path out of UEA's chat-style prompt struct.

    Raises if the prompt carries text: this model must not be handed the canonical
    question text, and a silent fallback would hide that.
    """
    if isinstance(prompt, str):
        raise ValueError("TasteSLM expects an audio prompt struct, got a bare string")
    paths, texts = [], []
    for turn in prompt:
        for c in turn.get("contents", []):
            if c.get("type") == "audio":
                paths.append(c["value"])
            elif c.get("type") == "text":
                texts.append(c["value"])
    if texts:
        raise ValueError(
            f"TasteSLM received text content in the prompt ({texts!r}). This adapter is "
            f"audio-in only; use the `direct-aqa` prompt."
        )
    if len(paths) != 1:
        raise ValueError(f"expected exactly 1 audio content, got {len(paths)}")
    return paths[0]


class TasteSLM(OfflineModel):
    """Joint TASTE-S SLM, question-audio -> answer-text."""

    def __init__(
        self,
        slm_root: str,
        config_path: str = None,
        sample_params: Dict[str, any] = None,
        gpu_id=None,
        **overrides,
    ):
        # `gpu_id` is what IsolatedModelPool injects (audio_evals/models/model_pool.py), and its
        # presence in this signature is exactly what `model_supports_pool` looks for. It may be an
        # int or a list of ints when there are more GPUs than instances; we pin to the first, since
        # one TASTE-SLM instance lives on a single device.
        super().__init__(is_chat=True, sample_params=sample_params)
        if isinstance(gpu_id, (list, tuple)):
            gpu_id = gpu_id[0] if gpu_id else None
        self.gpu_id = gpu_id
        # Expand $VAR/${VAR} so a shared registry entry does not have to hardcode one machine's
        # layout. An absolute path with no '$' is returned unchanged, so this is backward compatible.
        self.slm_root = os.path.abspath(os.path.expandvars(slm_root))
        self.config_path = os.path.expandvars(config_path) if config_path else config_path
        self.overrides = overrides
        self._adapter = None
        self._meta = None
        self._lock = threading.Lock()
        # Put the TASTE-SLM tree on sys.path at construction time: `_inference` imports
        # `eval.schemas` before it takes the lock, so deferring this to `_build` would
        # fail on the very first call.
        if self.slm_root not in sys.path:
            sys.path.insert(0, self.slm_root)

    # ------------------------------------------------------------------ loading
    def _build(self):
        if self._adapter is not None:
            return self._adapter
        with _BUILD_LOCK:
            if self._adapter is None:
                self._build_locked()
        return self._adapter

    def _build_locked(self):
        if self.slm_root not in sys.path:
            sys.path.insert(0, self.slm_root)
        import yaml
        from eval.adapters.taste_s_speech import TasteSSpeechAdapter, TasteSSpeechConfig

        cfg_path = self.config_path or os.path.join(
            self.slm_root, "eval/configs/tastes_slm_0p8b_spokenqa.yaml"
        )
        with open(cfg_path) as f:
            raw = yaml.safe_load(f)
        if raw.get("adapter") != "taste_s_speech":
            raise ValueError(f"{cfg_path}: expected adapter 'taste_s_speech'")
        conf = dict(raw["config"])
        # Same treatment for the profile's own paths, so a shared config can be written against
        # ${TASTE_S_BUNDLE} instead of an absolute path on one machine.
        for _k in ("slm_exp", "slm_ckpt", "tokenizer_dir", "llm_dir", "lm_backbone_dir",
                   "text_tokenizer_dir"):
            if isinstance(conf.get(_k), str):
                conf[_k] = os.path.expandvars(conf[_k])
        conf.update(self.overrides)          # recorded verbatim in effective_config
        if self.gpu_id is not None:
            # IsolatedModelPool takes its ids straight from CUDA_VISIBLE_DEVICES
            # (model_pool.get_available_gpu_ids), so with CUDA_VISIBLE_DEVICES=3,4,5 it hands out
            # gpu_id=3,4,5. Inside that process only cuda:0..2 exist, and `cuda:3` raises
            # "CUDA error: invalid device ordinal" on every sample. Translate the PHYSICAL id back
            # to its LOCAL ordinal (its position in CUDA_VISIBLE_DEVICES).
            # UEA hands out a comma-joined STRING ("0,1,2") when there are more visible GPUs
            # than pool instances; int() on that raised
            #   ValueError: invalid literal for int() with base 10: '0,1,2'
            # and every sample then recorded as an error. Take the first id in that case.
            _g = self.gpu_id
            if isinstance(_g, str) and "," in _g:
                _g = _g.split(",")[0].strip()
            gid = int(_g)
            vis = [v.strip() for v in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
                   if v.strip() != ""]
            if vis:
                ids = [int(v) for v in vis if v.lstrip("-").isdigit()]
                gid = ids.index(gid) if gid in ids else gid
            import torch
            n = torch.cuda.device_count()
            if gid >= n:
                raise ValueError(
                    f"resolved local GPU ordinal {gid} but only {n} device(s) are visible "
                    f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}). "
                    f"Refusing to start rather than fail on every sample.")
            conf["device"] = f"cuda:{gid}"
        self._effective_config = conf
        self._config_path = cfg_path
        adapter = TasteSSpeechAdapter(TasteSSpeechConfig.from_dict(conf))
        adapter.load()                        # loads weights now, so failures surface early
        self._meta = adapter.run_meta_fragment()
        self._adapter = adapter
        logger.info("TASTE-SLM loaded: %s", json.dumps(self._meta.get("model", {}))[:400])
        return adapter

    def describe(self) -> dict:
        self._build()
        return {
            "adapter_version": ADAPTER_VERSION,
            "config_path": self._config_path,
            "effective_config": self._effective_config,
            "run_meta_fragment": self._meta,
        }

    # ------------------------------------------------------------------ inference
    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        from eval.schemas import AudioInput, Status

        audio_path = _audio_path_from_prompt(prompt)
        # Stable per-sample identity -> stable speech-sampling seed (sha256(base_seed:sample_id)).
        sample_id = os.path.splitext(os.path.basename(audio_path))[0]

        with self._lock:
            adapter = self._build()
            res = adapter.generate(
                AudioInput(
                    sample_id=sample_id,
                    benchmark="llama-questions-s2t",
                    audio_path=audio_path,
                )
            )

        md = res.metadata or {}
        out = {
            # `text` is what extract_text takes. Always a string, never None.
            "text": res.response_text if res.status == Status.OK else "",
            "status": res.status,
            "sample_id": sample_id,
            "audio_path": audio_path,
            "raw_output": res.raw_output,
            "adapter_version": ADAPTER_VERSION,
            "protocol_version": md.get("protocol_version"),
            "effective_delay": md.get("effective_delay"),
            "absence_mode": md.get("absence_mode"),
            "speech_feedback": md.get("speech_feedback"),
            "finish_reason": md.get("finish_reason"),
            "truncated": md.get("truncated"),
            "answer_tokens": md.get("answer_tokens"),
            "joint_steps": md.get("joint_steps"),
            "stop_id": md.get("stop_id"),
            "speech_feedback_steps": md.get("speech_feedback_steps"),
            "sample_seed": md.get("sample_seed"),
            "layout": md.get("layout"),
            "effective_decoding": md.get("effective_decoding"),
            "audio_meta": md.get("audio"),
            # DIAGNOSTIC ONLY -- the model's own CTC read of the question. Never a scoring input
            # and never fed back into generation.
            "ctc_transcript": md.get("ctc_transcript"),
            "ctc_num_tokens": md.get("ctc_num_tokens"),
            "no_final_answer_reason": md.get("no_final_answer_reason"),
            "elapsed_sec": md.get("elapsed_sec"),
        }
        return json.dumps(out, ensure_ascii=False)
