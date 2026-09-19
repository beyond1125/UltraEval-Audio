"""
UltraEval-Audio model adapter for the TURN-BASED TASTE-S SLM.

ADDED BY US, and SEPARATE from `taste_slm.py` (stage-1) and `taste_slm_fd.py` (stage-2
full-duplex), both of which are left untouched along with all of their results in `res/`.

Model-side reference implementation
-----------------------------------
This wraps Jinhui's own inference package verbatim --
`TASTE-S_Full_duplex_SLM/train_slm/evaluation/turnbased_eval` @ 5cbcca7 -- by calling its two
documented entry points, `generate.load_models()` (once per process) and `generate.run_one()`
(once per sample). That package's README names `generate.py` "the single-sample inference
authority" and says a benchmark integration should carry its results rather than reimplement the
loop. Nothing about the TB grammar, the EOB rule or the one-position agent-speech delay is
reimplemented here.

Why this exists separately from the full-duplex adapter
------------------------------------------------------
Same benchmarks, same scoring, DIFFERENT generation contract. Turn-based generation has no 0.8 s
block clock and does not use `spk_head` at all: it is handed one complete user turn and produces
exactly one agent turn, terminated by the model's own trained `<eob>`. Three consequences that
matter when reading the numbers:

  * There is no "never spoke" outcome. `generate_turn` sets the first position's EOB logit to
    -inf ("training has no empty `<spk> <eob>` turn") and asserts the emitted turn is non-empty,
    so unlike full-duplex there is no valid-empty-response class to account for.
  * The stop token is genuinely trained -- the checkpoint reports val eob_recall 0.955,
    precision 0.607 -- unlike stage-1, where the end-of-utterance transition is masked out of the
    loss and generation always runs to the token cap.
  * `forced_eob` is therefore the interesting failure mode: it is True only when the `max_words`
    cap fired instead of the model choosing `<eob>`. That is this model's truncation rate and it
    is reported per sample.

Audio input contract
--------------------
The author's turn-based front end is NOT the same as full-duplex's: channel 0, torchaudio 16 kHz
resample, then int16-grid rounding, matching how the DeepDialogue turn-based shards were built.
Both knobs are exposed and default to the author's values. Changing them silently would evaluate
the checkpoint under a data contract it was not trained for.

What the model receives
-----------------------
ONLY the question audio, as the single user turn, with empty prior history. UEA's `direct-aqa`
prompt carries one `audio` content and no text; this adapter raises if any text content appears,
so an oracle transcript cannot leak in silently. There is no CTC->Qwen cascade: the author's
causal `TasteExtractor` is the model's own front end.

`skip_audio=True` skips ONLY the vocoder waveform rendering, which S2T does not need. It is a late
branch in `run_one` AFTER generation, so the joint speech-token stream still runs in full.

Decoding
--------
The author's own EVALUATION defaults, unchanged: temperature 0.8, top_k 25, top_p 1.0,
max_words 80, eob_logit_bias 0.0 (`run.sh`). These are also the full-duplex adapter's values, so
the two models are compared under the same decoding. Note that the bundled `cmd.sh` demo passes
`--greedy` instead -- that is a deterministic smoke test, not the evaluation default; pass
`greedy: true` here to reproduce it.

`eob_logit_bias` stays at 0.0. The author provides it for calibration sweeps; moving it would be
tuning the stopping behaviour against the benchmark.

Because the default is sampling, this adapter seeds torch per sample from
`sha256(seed, sample_id)`, so a run is reproducible and -- unlike a single global seed -- does not
depend on the order in which samples happen to be scheduled across shards.

Recipe path rewrite (required, and NOT architectural)
-----------------------------------------------------
`slm_loader.build_eval_config` resolves dependency paths by exporting `$QWEN_MODEL_DIR` /
`$TASTE_S_TOKENIZER_DIR` and expanding `${...}` placeholders in the recipe. That works for a
`model_recipe.yaml`, which ships `llm_dir: ${QWEN_MODEL_DIR}`. But this checkpoint publishes a real
`training_config.yaml` carrying the TRAINING MACHINE's literal absolute paths:

    llm_dir: /home/jensen/.../llm/Qwen3.5-4B
    taste_s_tokenizer_dir: /home/jensen/.../taste_s_tokenizer/fsq_ourdata/valid_best

There is no placeholder to expand, so those survive and `model.py:361`
(`_load_frozen_quantizer(mc.taste_s_tokenizer_dir)`) reads the baked path -- which on any other
machine does not exist, and transformers then treats it as a HuggingFace repo id and raises
`Repo id must be in the form 'repo_name' or 'namespace/repo_name'`. Note that `load_models`'
`taste_s_tokenizer_dir` argument does NOT override it: it is used for the tokenizer/extractor, not
for the quantizer inside the SLM.

So this adapter writes a corrected COPY of the recipe under `trace_dir/_resolved_recipe/` with
exactly those two paths replaced by the local ones, and passes it as `recipe=`. Nothing
architectural is touched, the author's code is not modified, and the substitution is asserted
against the expected model family first (so a checkpoint from a different family fails loudly
rather than silently loading the wrong backbone). The resolved recipe path and both replacements
are recorded in every sample's output.

Answer extraction (predetermined, reference-independent)
--------------------------------------------------------
`result["predicted_text"]`: the author's decode of the one generated agent turn. Not chosen from
the reference, not truncated to a first sentence, not conditional on matching. The full token and
TASTE trace stays in `generation.json` under `trace_dir` and its path is recorded in the output.

Output contract
---------------
Returns a JSON STRING so UEA's `extract_text` (`JsonExtract(extract_key="text")`) works. A bare
`"42"` would `json.loads` to an int and then `d["text"]` would raise, so numeric answers are
wrapped: `{"text": "42", ...}`. Extra keys are diagnostics; UEA stores the whole raw string under
`type: "inference"`.
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

ADAPTER_VERSION = "uea_taste_slm_tb_adapter_v1"
PROTOCOL_VERSION = "turnbased_single_agent_turn_v1"
AUTHOR_CODE_COMMIT = "5cbcca7"

# See taste_slm.py: transformers 5.x lazy submodule resolution is not thread-safe; warm it on the
# main thread at import time.
from transformers import AutoTokenizer as _WARM_AutoTokenizer  # noqa: F401,E402

# Building two multi-billion-parameter checkpoints concurrently is disk/CPU-bound and multiplies
# peak host memory.
_BUILD_LOCK = threading.Lock()

# The published training_config.yaml carries the trainer's own absolute paths and may reference
# ${...} placeholders that OmegaConf would try to resolve as interpolation KEYS. None is used at
# inference: turnbased_loader blanks init_from_slm, disables text_kl, and never builds a
# dataloader. Same class of issue as PATCH-S2-001 for full-duplex.
_RECIPE_PLACEHOLDERS = ("AUDITED_TRAIN_DATA_DIRS", "AUDITED_VALID_DATA_DIRS",
                        "NEW_OUTPUT_DIR", "TURNBASED_WEIGHTS_DIR")


def _audio_path_from_prompt(prompt: PromptStruct) -> str:
    """Pull the single audio path out of UEA's chat-style prompt struct.

    Raises if the prompt carries text: this model must not be handed the canonical question text,
    and a silent fallback would hide that.
    """
    if isinstance(prompt, str):
        raise ValueError("TasteSLMTurnBased expects an audio prompt struct, got a bare string")
    paths, texts = [], []
    for turn in prompt:
        for c in turn.get("contents", []):
            if c.get("type") == "audio":
                paths.append(c["value"])
            elif c.get("type") == "text":
                texts.append(c["value"])
    if texts:
        raise ValueError(
            f"TasteSLMTurnBased received text content in the prompt ({texts!r}). This adapter is "
            f"audio-in only; use the `direct-aqa` prompt."
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
    """Deterministic per-sample seed, independent of scheduling order.

    A single global seed makes a sharded run irreproducible: sample N's draw depends on how many
    samples ran before it in that process. Deriving the seed from the sample id removes that.
    """
    h = hashlib.sha256(f"{base}:{sample_id}".encode()).digest()
    return int.from_bytes(h[:4], "big")


def _resolve_recipe(recipe_fp, out_dir, taste_s_tokenizer_dir, qwen_model_dir):
    """Write a copy of the recipe with the training machine's baked paths made local.

    Returns (path, replacements). Only `model_config.llm_dir` and
    `model_config.taste_s_tokenizer_dir` are touched, and only when the baked value does not exist
    locally -- a recipe that already resolves is passed through unchanged.
    """
    import yaml as _yaml

    cfg = _yaml.safe_load(open(recipe_fp))
    mc = cfg.get("model_config") or {}
    repl = {}
    for key, local, expect in (("llm_dir", qwen_model_dir, "Qwen3.5-4B"),
                               ("taste_s_tokenizer_dir", taste_s_tokenizer_dir, "fsq_ourdata")):
        baked = str(mc.get(key, ""))
        if not baked or os.path.isdir(baked) or baked.startswith("${"):
            continue  # already usable, or a placeholder build_eval_config will expand
        if expect not in baked:
            raise ValueError(
                f"recipe {recipe_fp} has {key}={baked!r}, which does not name {expect!r}. Refusing "
                f"to rewrite it to {local!r}: this checkpoint is not the family this adapter "
                f"expects, and silently substituting would evaluate the wrong dependency."
            )
        if not os.path.isdir(local):
            raise ValueError(f"local replacement for {key} does not exist: {local}")
        mc[key] = local
        repl[key] = {"from": baked, "to": local}
    if not repl:
        return recipe_fp, {}
    cfg["model_config"] = mc
    os.makedirs(out_dir, exist_ok=True)
    fp = os.path.join(out_dir, "training_config.resolved.yaml")
    with open(fp, "w") as f:
        _yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    return fp, repl


class TasteSLMTurnBased(OfflineModel):
    """Turn-based TASTE-S SLM: one question-audio user turn -> one agent turn's text (S2T)."""

    def __init__(
        self,
        tb_eval_root: str,
        ckpt_dir: str,
        taste_s_tokenizer_dir: str,
        qwen_model_dir: str,
        trace_dir: str,
        recipe: str = None,
        device: str = "cuda:0",
        temperature: float = 0.8,
        top_k: int = 25,
        top_p: float = 1.0,
        greedy: bool = False,
        max_words: int = 80,
        eob_logit_bias: float = 0.0,
        input_resampler: str = "torchaudio",
        quantize_input: bool = True,
        seed: int = 42,
        sample_params: Dict[str, any] = None,
        gpu_id=None,
        **overrides,
    ):
        super().__init__(is_chat=True, sample_params=sample_params)
        # `gpu_id` is what IsolatedModelPool injects; its presence is what `model_supports_pool`
        # looks for. One instance lives on one device. NOTE: we pass `device` straight to
        # load_models and never touch CUDA_VISIBLE_DEVICES -- generate.py only overwrites that
        # inside its own main() (generate.py:226), which we do not call.
        if isinstance(gpu_id, (list, tuple)):
            gpu_id = gpu_id[0] if gpu_id else None
        if gpu_id is not None:
            device = f"cuda:{gpu_id}"
        self.device = device
        # Expand $VAR/${VAR} so a shared registry entry need not hardcode one machine's layout.
        # An absolute path with no '$' is returned unchanged -> backward compatible.
        _x = os.path.expandvars
        self.tb_eval_root = os.path.abspath(_x(tb_eval_root))
        self.ckpt_dir = _x(ckpt_dir)
        # load_models defaults recipe to <ckpt_dir>/../training_config.yaml, which is exactly the
        # published layout; passing it explicitly keeps the resolved path in the record.
        self.recipe = _x(recipe) if recipe else None
        self.taste_s_tokenizer_dir = _x(taste_s_tokenizer_dir)
        self.qwen_model_dir = _x(qwen_model_dir)
        self.trace_dir = _x(trace_dir)
        self.seed = int(seed)
        self.gen_kwargs = dict(
            temperature=temperature, top_k=top_k, top_p=top_p, greedy=greedy,
            max_words=max_words, eob_logit_bias=eob_logit_bias,
            input_resampler=input_resampler, quantize_input=quantize_input,
        )
        self.overrides = overrides
        self._models = None
        self._generate = None
        self.resolved_recipe = None
        self.recipe_replacements = None
        self._lock = threading.Lock()
        if self.tb_eval_root not in sys.path:
            sys.path.insert(0, self.tb_eval_root)

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
        if self.tb_eval_root not in sys.path:
            sys.path.insert(0, self.tb_eval_root)
        cwd = os.getcwd()
        try:
            # _bootstrap_tb resolves sibling modules relative to the package dir.
            os.chdir(self.tb_eval_root)
            import generate as _generate  # the author's module, imported not copied
        finally:
            os.chdir(cwd)
        self._generate = _generate
        # load_models defaults recipe to <ckpt_dir>/../training_config.yaml
        recipe_fp = self.recipe or os.path.join(os.path.dirname(os.path.abspath(self.ckpt_dir)),
                                                "training_config.yaml")
        if not os.path.isfile(recipe_fp):
            raise FileNotFoundError(f"recipe not found: {recipe_fp}")
        resolved, repl = _resolve_recipe(
            recipe_fp, os.path.join(self.trace_dir, "_resolved_recipe"),
            self.taste_s_tokenizer_dir, self.qwen_model_dir)
        self.resolved_recipe = resolved
        self.recipe_replacements = repl
        for k, v in repl.items():
            logger.info("[taste-slm-tb] recipe %s: %s -> %s", k, v["from"], v["to"])
        logger.info("[taste-slm-tb] loading turn-based checkpoint %s on %s",
                    self.ckpt_dir, self.device)
        self._models = _generate.load_models(
            self.ckpt_dir, self.taste_s_tokenizer_dir, self.qwen_model_dir,
            self.device, recipe=resolved,
        )
        logger.info("[taste-slm-tb] loaded; eob_text_id=%s num_codebooks=%s",
                    getattr(self._models.slm, "eob_text_id", None),
                    getattr(self._models.slm, "num_codebooks", None))

    # ------------------------------------------------------------------ inference
    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        audio_path = _audio_path_from_prompt(prompt)
        sample_id = os.path.splitext(os.path.basename(audio_path))[0]
        out_dir = _per_sample_path(self.trace_dir, audio_path)
        seed = _sample_seed(self.seed, sample_id)

        with self._lock:
            models = self._build()
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            summary = self._generate.run_one(
                models, audio_path, out_dir,
                speaker_embedding_npy=None,    # no reference voice for a benchmark input
                reference_agent_audio=None,    # and no ground-truth agent continuation
                skip_audio=True,               # skips ONLY vocoder rendering; see docstring
                **self.gen_kwargs,
            )

        # Predetermined extraction: the author's decode of the single generated agent turn.
        text = (summary.get("predicted_text") or "").strip()

        out = {
            # `text` is what extract_text takes. Always a string, never None.
            "text": text,
            "status": "ok",
            # TB forces a non-empty turn, so this should never be True; recorded so that a
            # regression in the author's generation loop would show up rather than hide.
            "empty_response": text == "",
            "sample_id": sample_id,
            "audio_path": audio_path,
            "adapter_version": ADAPTER_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "author_code_commit": AUTHOR_CODE_COMMIT,
            "modality": "S2T_native_text_stream",
            "seed": seed,
            # True only when the max_words cap fired instead of the model choosing <eob>:
            # this model's truncation rate.
            "forced_eob": summary.get("forced_eob"),
            "n_predicted_text_tokens": summary.get("n_predicted_text_tokens"),
            "n_input_text_tokens": summary.get("n_input_text_tokens"),
            # the tokenizer's own CTC transcript of the question, for input-side diagnosis only --
            # it is never fed to the model and never scored
            "ctc_text": summary.get("ctc_text"),
            "input_contract": {
                "resampler": self.gen_kwargs["input_resampler"],
                "int16_quantization": self.gen_kwargs["quantize_input"],
            },
            "decoding": {k: self.gen_kwargs[k] for k in
                         ("temperature", "top_k", "top_p", "greedy", "max_words",
                          "eob_logit_bias")},
            "trace_dir": out_dir,
            "resolved_recipe": getattr(self, "resolved_recipe", None),
            "recipe_replacements": getattr(self, "recipe_replacements", None),
        }
        return json.dumps(out, ensure_ascii=False)
