"""
UltraEval-Audio model adapter for the STAGE-2 FULL-DUPLEX TASTE-S SLM.

ADDED BY US, and SEPARATE from `taste_slm.py` (stage-1) which is left untouched along with every
stage-1 result in `res/`. This is an adapter, not an upstream fix; see PATCHES.md.

Model-side reference implementation
-----------------------------------
This wraps Jinhui's own inference package verbatim --
`TASTE-S_Full_duplex_SLM/train_slm/evaluation/full_duplex_eval` @ 728867b -- by calling its two
documented entry points, `generate.load_models()` (once per process) and `generate.run_one()`
(once per sample). Per that package's README, a benchmark integration is meant to call these two
functions rather than modify `generate.py`. Nothing in the author's generation loop, block
grammar, control heads or stopping rule is reimplemented here.

What the model receives
-----------------------
ONLY the question audio. UEA's `direct-aqa` prompt carries a single `audio` content and no text;
this adapter raises if any text content appears, so an oracle transcript cannot leak in silently.
There is no CTC->original-Qwen cascade: the author's `TasteExtractor` + `window_stitch` encode is
the model's own front end.

Generation mode (documented, not silently chosen)
-------------------------------------------------
* The question audio IS the user channel. The model listens block-by-block (0.8s blocks) and
  decides per block whether to speak, via its trained `spk_head`. We never force speaking, never
  touch the decision threshold, and never inject a prompt.
* After the question audio ends, the author's native mechanism keeps feeding SILENT user blocks
  for up to `max_extra_silent_blocks` (default 15 = 12.0s) so the model has room to begin and
  finish an answer. This is the author's default, not an override we invented.
* The only "oracle" boundary present is the benchmark's own file boundary: the question wav ends
  where the question ends. No end-of-question marker is injected into the model.
* `skip_audio=True`: this skips ONLY the vocoder waveform rendering, which S2T does not need. It
  is a late return in `run_one` AFTER generation -- the joint speech-token branch and its
  feedback run fully. Reporting S2T does not switch the speech branch off.

Answer extraction (predetermined, reference-independent)
--------------------------------------------------------
Concatenate `text` from every block with `speaking == True`, in block order. That is the model's
entire assistant emission for the session -- it is not chosen from the reference, not truncated to
a first sentence, and not conditional on whether it matches. The full per-block turn log is kept
at `trace.jsonl` under `trace_dir` and its path is recorded in the output.

A session where the model never speaks yields `text: ""` with `status: "ok"` and
`spoke_blocks: 0`. That is a VALID EMPTY RESPONSE, not an execution failure, and the two must be
counted differently in coverage accounting.

Output contract
---------------
Returns a JSON STRING so UEA's `extract_text` (`JsonExtract(extract_key="text")`) works. A bare
`"42"` would `json.loads` to an int and then `d["text"]` would raise, so numeric answers are
wrapped: `{"text": "42", ...}`. Extra keys are diagnostics; UEA stores the whole raw string under
`type: "inference"`.
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

ADAPTER_VERSION = "uea_taste_slm_fd_adapter_v1"
PROTOCOL_VERSION = "fullduplex_stage2_listen_then_speak_v1"

# See taste_slm.py: transformers 5.x lazy submodule resolution is not thread-safe; warm it on the
# main thread at import time.
from transformers import AutoTokenizer as _WARM_AutoTokenizer  # noqa: F401,E402

# Building two 9B checkpoints concurrently is disk/CPU-bound and multiplies peak host memory.
_BUILD_LOCK = threading.Lock()

# PATCH-S2-001 (STAGE2_9B_EVALUATION.md §9): the portable model_recipe.yaml ships unresolved
# ${...} placeholders. slm_loader._expand_env leaves unset vars literal and OmegaConf then reads
# them as interpolation KEYS, so `cfg.training_config.get("init_from_slm")` raises
# InterpolationKeyError. None of these is used at inference -- the loader deliberately blanks
# init_from_slm, and dataloaders are never built.
_RECIPE_PLACEHOLDERS = ("STAGE2_WEIGHTS_DIR", "AUDITED_TRAIN_DATA_DIRS",
                        "AUDITED_VALID_DATA_DIRS", "NEW_OUTPUT_DIR")


def _audio_path_from_prompt(prompt: PromptStruct) -> str:
    """Pull the single audio path out of UEA's chat-style prompt struct.

    Raises if the prompt carries text: this model must not be handed the canonical question text,
    and a silent fallback would hide that.
    """
    if isinstance(prompt, str):
        raise ValueError("TasteSLMFullDuplex expects an audio prompt struct, got a bare string")
    paths, texts = [], []
    for turn in prompt:
        for c in turn.get("contents", []):
            if c.get("type") == "audio":
                paths.append(c["value"])
            elif c.get("type") == "text":
                texts.append(c["value"])
    if texts:
        raise ValueError(
            f"TasteSLMFullDuplex received text content in the prompt ({texts!r}). This adapter "
            f"is audio-in only; use the `direct-aqa` prompt."
        )
    if len(paths) != 1:
        raise ValueError(f"expected exactly 1 audio content, got {len(paths)}")
    return paths[0]


class TasteSLMFullDuplex(OfflineModel):
    """Stage-2 full-duplex TASTE-S SLM: question-audio -> assistant text (S2T)."""

    def __init__(
        self,
        fd_eval_root: str,
        ckpt_dir: str,
        recipe: str,
        taste_s_tokenizer_dir: str,
        qwen_model_dir: str,
        trace_dir: str,
        device: str = "cuda:0",
        temperature: float = 0.8,
        top_k: int = 25,
        top_p: float = 1.0,
        max_words_per_block: int = 40,
        max_extra_silent_blocks: int = 15,
        sample_params: Dict[str, any] = None,
        gpu_id=None,
        **overrides,
    ):
        super().__init__(is_chat=True, sample_params=sample_params)
        # `gpu_id` is what IsolatedModelPool injects; its presence is what `model_supports_pool`
        # looks for. One instance lives on one device. NOTE: we pass `device` straight to
        # load_models and never touch CUDA_VISIBLE_DEVICES -- generate.py only overwrites that
        # inside its own main(), which we do not call.
        if isinstance(gpu_id, (list, tuple)):
            gpu_id = gpu_id[0] if gpu_id else None
        if gpu_id is not None:
            device = f"cuda:{gpu_id}"
        self.device = device
        # Expand $VAR/${VAR} so a shared registry entry need not hardcode one machine's layout.
        # An absolute path with no '$' is returned unchanged -> backward compatible.
        _x = os.path.expandvars
        self.fd_eval_root = os.path.abspath(_x(fd_eval_root))
        self.ckpt_dir = _x(ckpt_dir)
        self.recipe = _x(recipe)
        self.taste_s_tokenizer_dir = _x(taste_s_tokenizer_dir)
        self.qwen_model_dir = _x(qwen_model_dir)
        self.trace_dir = _x(trace_dir)
        self.gen_kwargs = dict(
            temperature=temperature, top_k=top_k, top_p=top_p,
            max_words_per_block=max_words_per_block,
            max_extra_silent_blocks=max_extra_silent_blocks,
        )
        self.overrides = overrides
        self._models = None
        self._generate = None
        self._lock = threading.Lock()
        if self.fd_eval_root not in sys.path:
            sys.path.insert(0, self.fd_eval_root)

    # ------------------------------------------------------------------ loading
    def _build(self):
        if self._models is not None:
            return self._models
        with _BUILD_LOCK:
            if self._models is None:
                self._build_locked()
        return self._models

    def _build_locked(self):
        for k in _RECIPE_PLACEHOLDERS:
            os.environ.setdefault(k, "__unused_for_inference__")
        if self.fd_eval_root not in sys.path:
            sys.path.insert(0, self.fd_eval_root)
        cwd = os.getcwd()
        try:
            # _bootstrap_fd resolves sibling modules relative to the package dir.
            os.chdir(self.fd_eval_root)
            import generate as _generate  # the author's module, imported not copied
        finally:
            os.chdir(cwd)
        self._generate = _generate
        logger.info("[taste-slm-fd] loading stage-2 checkpoint %s on %s",
                    self.ckpt_dir, self.device)
        self._models = _generate.load_models(
            self.ckpt_dir, self.taste_s_tokenizer_dir, self.qwen_model_dir,
            self.device, recipe=self.recipe,
        )
        logger.info("[taste-slm-fd] loaded; nq=%s eob_text_id=%s",
                    self._models.nq, self._models.eob_text_id)

    # ------------------------------------------------------------------ inference
    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        audio_path = _audio_path_from_prompt(prompt)
        sample_id = os.path.splitext(os.path.basename(audio_path))[0]
        out_dir = os.path.join(self.trace_dir, sample_id)

        with self._lock:
            models = self._build()
            summary = self._generate.run_one(
                models, audio_path, out_dir,
                agent_spk_emb=None,            # no ground-truth agent voice for a benchmark input
                reference_agent_audio=None,
                skip_audio=True,               # skips ONLY vocoder rendering; see module docstring
                verbose=False,
                **self.gen_kwargs,
            )

        # Predetermined extraction: every spoken block's text, in block order.
        trace_fp = os.path.join(out_dir, "trace.jsonl")
        parts, spoke_blocks, max_logit = [], 0, None
        with open(trace_fp) as f:
            for line in f:
                r = json.loads(line)
                lg = r.get("spk_logit")
                if lg is not None:
                    max_logit = lg if max_logit is None else max(max_logit, lg)
                if r.get("speaking"):
                    spoke_blocks += 1
                    if r.get("text"):
                        parts.append(r["text"])
        text = "".join(parts).strip()

        out = {
            # `text` is what extract_text takes. Always a string, never None.
            "text": text,
            # A silent session is a VALID EMPTY RESPONSE, not a failure.
            "status": "ok",
            "empty_response": text == "",
            "sample_id": sample_id,
            "audio_path": audio_path,
            "raw_output": "".join(parts),
            "adapter_version": ADAPTER_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "author_code_commit": "728867b407a539b8a3d3229f83c6a722aa2f7473",
            "modality": "S2T_native_text_stream",
            "n_blocks": summary.get("n_blocks"),
            "n_extra_blocks": summary.get("n_extra_blocks"),
            "spoke_blocks": spoke_blocks,
            "n_words": summary.get("n_words"),
            "max_spk_logit": max_logit,
            "trace_path": trace_fp,
            "decoding": dict(self.gen_kwargs),
        }
        return json.dumps(out, ensure_ascii=False)
