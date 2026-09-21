"""
UltraEval-Audio model adapter for the STAGE-2 FULL-DUPLEX TASTE-S SLM.

ADDED BY US, and SEPARATE from `taste_slm.py` (stage-1) which is left untouched along with every
stage-1 result in `res/`. This is an adapter, not an upstream fix; see README_TASTE_S.md.

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

S2S arm (`synthesize_audio: true`; off by default, so the S2T entry is unchanged)
---------------------------------------------------------------------------------
`run_one(skip_audio=False)` renders the author's own `agent_audio.wav`: the AGENT channel only,
every speaking span synthesised by the tokenizer's unit decoder + vocoder, and every silent block
as exactly `0.8 s` of zeros. Voice is the author's benchmark default, a zero speaker embedding
(`agent_spk_emb=None`); there is no reference agent voice for a benchmark input.

The ASR input is that file with its LEADING and TRAILING silent blocks cut off (written as
`agent_audio_spoken.wav`). The cut is exact, not energy-based: silence runs are inserted as
`round(n_blocks * 0.8 * sr)` zero samples, and only speaking spans can drift in length
(STAGE2_9B_EVALUATION.md §12.2), so a leading/trailing run can be computed from the block flags. The
cut samples are asserted to be all zero, so a mismatch with the author's assembly raises instead of
clipping speech. Interior silent gaps between spans are kept. Rationale: the leading run is the
model listening to the question -- often 3-10 s of digital silence that invites whisper to
hallucinate, and carries no answer content.

A session where the model never spoke returns `audio: ""` (and writes no ASR file). The
`speech2text-local-allow-empty` step maps that to an empty transcript without calling whisper, so it
is scored as a wrong answer inside the full denominator -- exactly how the S2T arm treats it --
rather than as a failure.

The same generation carries both `text` and `audio`, so one run can be scored as S2S
(`--task s2s-aqa-local-allow-empty`) and then re-scored as S2T from the saved inference records
(`--task loose-aqa --inf_file <that run's .jsonl>`), with no second generation. That pairing matters
here: decoding is sampled, so two separate runs would not produce the same text.

`seed` (off by default): when set, torch / numpy / random are seeded per sample from
`sha256(seed, sample_id)` before `run_one`, covering both generation and the unit decoder's
sampling, so a run is reproducible and independent of shard scheduling.
"""
import hashlib
import json
import logging
import os
import sys
import threading
from typing import Dict

from audio_evals.base import PromptStruct
from audio_evals.models.model import OfflineModel

logger = logging.getLogger(__name__)

ADAPTER_VERSION = "uea_taste_slm_fd_adapter_v3"   # v3: configurable tokenizer frontend + paired 4B FD profiles
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


def _per_sample_path(root: str, audio_path: str) -> str:
    """`<root>/<dataset>/<split>/<sample_id>` for a question at `.../<dataset>/<split>/<id>.wav`.

    Every benchmark numbers its rows from 0, so a path keyed by sample_id alone is shared between
    datasets: concurrent runs overwrite each other's per-sample outputs (and an S2S run could
    transcribe another dataset's wav). The dataset/split come from the question's own path, so
    nothing has to be set at launch. sample_id -- and hence the per-sample seed -- is unchanged.
    """
    parts = os.path.normpath(audio_path).split(os.sep)
    if len(parts) < 3:
        raise ValueError(f"cannot derive <dataset>/<split> from audio path {audio_path!r}")
    return os.path.join(root, parts[-3], parts[-2], os.path.splitext(parts[-1])[0])


def _sample_seed(base: int, sample_id: str) -> int:
    """Deterministic per-sample seed, independent of scheduling order (same as taste_slm_tb.py)."""
    h = hashlib.sha256(f"{base}:{sample_id}".encode()).digest()
    return int.from_bytes(h[:4], "big")


def _seed_everything(seed: int):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _spoken_region(out_dir: str, chunk_seconds: float) -> Dict[str, any]:
    """Cut the leading/trailing silent blocks off the author's `agent_audio.wav`; see docstring.

    Returns the fields merged into the output: `audio` (path, or "" if the model never spoke) and
    `audio_synthesis` (provenance).
    """
    import numpy as np
    import soundfile as sf

    speaking = [bool(x) for x in np.load(os.path.join(out_dir, "agent_generated.npz"),
                                         allow_pickle=True)["speaking"]]
    full_fp = os.path.join(out_dir, "agent_audio.wav")
    meta = {"agent_audio": full_fp, "voice": "zeros(192) -- author default, no reference voice",
            "n_blocks": len(speaking), "spoke_blocks": sum(speaking)}
    if not any(speaking):
        meta.update(empty=True, asr_input=None)
        return {"audio": "", "audio_synthesis": meta}

    wav, sr = sf.read(full_fp, dtype="float32")
    first = speaking.index(True)
    last = len(speaking) - 1 - speaking[::-1].index(True)
    lead = int(round(first * chunk_seconds * sr)) if first else 0
    trail_blocks = len(speaking) - 1 - last
    trail = int(round(trail_blocks * chunk_seconds * sr)) if trail_blocks else 0
    if lead + trail >= len(wav):
        raise RuntimeError(f"{full_fp}: silence cut {lead}+{trail} >= {len(wav)} samples")
    # the cut must be pure zeros, i.e. exactly the author's inserted silence runs
    if np.any(wav[:lead]) or (trail and np.any(wav[len(wav) - trail:])):
        raise RuntimeError(f"{full_fp}: non-zero samples inside the leading/trailing silence runs; "
                           f"the block->sample arithmetic no longer matches assemble_agent_audio")
    spoken = wav[lead:len(wav) - trail]
    asr_fp = os.path.join(out_dir, "agent_audio_spoken.wav")
    sf.write(asr_fp, spoken, sr)
    meta.update(empty=False, asr_input=asr_fp, sample_rate=sr,
                first_spoken_block=first, last_spoken_block=last,
                full_sec=round(len(wav) / sr, 3), asr_sec=round(len(spoken) / sr, 3),
                # drift = rendered speech longer/shorter than its block budget (§12.2)
                drift_sec=round(len(wav) / sr - len(speaking) * chunk_seconds, 3))
    return {"audio": asr_fp, "audio_synthesis": meta}


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
        end_after_consecutive_silent_blocks: int = 1,
        spk_logit_threshold: float = 0.0,
        max_tail_units: int = 500,
        input_resampler: str = "librosa",
        input_quantization: str = "on",
        code_revision: str = "728867b407a539b8a3d3229f83c6a722aa2f7473",
        protocol_version: str = PROTOCOL_VERSION,
        synthesize_audio: bool = False,
        seed: int = None,
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
        self.code_revision = str(code_revision)
        self.protocol_version = str(protocol_version)
        # YAML registry values are literal strings.  Expand these scalar knobs
        # as well as paths; if a launcher predates either environment variable,
        # retain the constructor default rather than attempting int/float on an
        # unresolved ``${...}`` placeholder.
        def _scalar_env(value, default):
            if isinstance(value, str):
                value = _x(value)
                if "$" in value:
                    return default
            return value

        end_after_consecutive_silent_blocks = _scalar_env(
            end_after_consecutive_silent_blocks, 1
        )
        spk_logit_threshold = _scalar_env(spk_logit_threshold, 0.0)
        if input_resampler not in ("librosa", "torchaudio"):
            raise ValueError(f"unsupported input_resampler={input_resampler!r}")
        if input_quantization not in ("on", "off"):
            raise ValueError(f"unsupported input_quantization={input_quantization!r}")
        self.gen_kwargs = dict(
            temperature=temperature, top_k=top_k, top_p=top_p,
            max_words_per_block=max_words_per_block,
            max_extra_silent_blocks=max_extra_silent_blocks,
            end_after_consecutive_silent_blocks=int(end_after_consecutive_silent_blocks),
            spk_logit_threshold=float(spk_logit_threshold),
            max_tail_units=max_tail_units,
            input_resampler=input_resampler,
            input_quantization=input_quantization,
        )
        self.synthesize_audio = bool(synthesize_audio)
        self.seed = None if seed is None else int(seed)
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
        out_dir = _per_sample_path(self.trace_dir, audio_path)

        seed = None if self.seed is None else _sample_seed(self.seed, sample_id)
        with self._lock:
            models = self._build()
            if seed is not None:
                _seed_everything(seed)
            summary = self._generate.run_one(
                models, audio_path, out_dir,
                agent_spk_emb=None,            # no ground-truth agent voice for a benchmark input
                reference_agent_audio=None,
                # S2T: skips ONLY vocoder rendering (see module docstring). S2S: render it.
                skip_audio=not self.synthesize_audio,
                verbose=False,
                **self.gen_kwargs,
            )
            synth = (_spoken_region(out_dir, self._generate.CHUNK_SECONDS)
                     if self.synthesize_audio else None)

        # Predetermined extraction: concatenate raw IDs from every speaking block, then
        # decode ONCE. Per-block decode can change whitespace/subword boundaries at 0.8 s cuts.
        trace_fp = os.path.join(out_dir, "trace.jsonl")
        all_text_ids, spoke_blocks, max_logit, hit_unit_tail_cap = [], 0, None, False
        with open(trace_fp) as f:
            for line in f:
                r = json.loads(line)
                lg = r.get("spk_logit")
                if lg is not None:
                    max_logit = lg if max_logit is None else max(max_logit, lg)
                if r.get("speaking"):
                    spoke_blocks += 1
                    all_text_ids.extend(r.get("text_ids", ()))
                hit_unit_tail_cap = hit_unit_tail_cap or any(
                    e.get("action") == "tail_cap" for e in r.get("decoder_events", ()))
        raw_output = models.text_tok.decode(all_text_ids, skip_special_tokens=True)
        text = raw_output.strip()

        out = {
            # `text` is what extract_text takes. Always a string, never None.
            "text": text,
            # A silent session is a VALID EMPTY RESPONSE, not a failure.
            "status": "ok",
            "empty_response": text == "",
            "sample_id": sample_id,
            "audio_path": audio_path,
            "raw_output": raw_output,
            "adapter_version": ADAPTER_VERSION,
            "protocol_version": self.protocol_version,
            "author_code_commit": self.code_revision,
            "modality": ("S2T_native_text_stream+S2S_agent_audio" if synth is not None
                         else "S2T_native_text_stream"),
            "seed": seed,
            "n_blocks": summary.get("n_blocks"),
            "n_extra_blocks": summary.get("n_extra_blocks"),
            "spoke_blocks": spoke_blocks,
            "n_words": summary.get("n_words"),
            "max_spk_logit": max_logit,
            "hit_unit_tail_cap": hit_unit_tail_cap,
            "input_audio_preprocessing": summary.get("input_audio_preprocessing"),
            "agent_voice_mode": "zero_embedding",
            "trace_path": trace_fp,
            "decoding": dict(self.gen_kwargs),
            # S2S: `audio` is what extract_audio reads ("" = never spoke). Absent on the S2T path.
            **(synth or {}),
        }
        return json.dumps(out, ensure_ascii=False)
