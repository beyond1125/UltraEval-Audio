# Spoken QA for the stage-2 FULL-DUPLEX TASTE-S SLM

Same three benchmarks and the same upstream scoring chain as the stage-1 models — task
`loose-aqa`, prompt `direct-aqa`, evaluator `qa-exist-match`, aggregation `acc`, on pinned UEA
commit `637cc336`. Nothing in the scoring path is ours.

What differs is the model side, and one thing about how the results must be read.

**Jinhui's repo needs no changes at all.** The adapter imports his `generate.load_models()` and
calls `run_one(..., skip_audio=True)` unmodified; it never calls his `main()`.

| file | role |
|---|---|
| `audio_evals/models/taste_slm_fd.py` | the adapter |
| `registry/model/taste_slm_fd.yaml` | model id **`taste-slm-fd-9b-stage2`** |

Stage-1 entries (`taste_slm.yaml`, `taste_slm_datav2.yaml`, `taste_slm_4b_kl.yaml`,
`taste_slm_9b_v2.yaml`) and all of their results in `res/` are untouched by any of this.

## 1. What you need on disk

```bash
# the author's code, at the commit every reported number used
git clone git@github.com:Jensen-JinhuiLiu/TASTE-S_Full_duplex_SLM.git
cd TASTE-S_Full_duplex_SLM && git checkout 728867b407a539b8a3d3229f83c6a722aa2f7473
export TASTE_S_FD_REPO=$PWD && cd ..
```

The stage-2 checkpoint (gated; same repo as the stage-1 ones, different prefix):

```bash
export TASTE_S_STAGE2_CKPT=/abs/path/ckpt_stage2_9b
hf download JokeMaker/taste-s-stage1-full-f32-validbest8-v4-epochs \
    --revision 6ef188d8d388d58481167fdf6c2e0476b8f0e104 \
    --include "checkpoints/slm_stage2_qwen35_9b_fsdp_v1_recovery1/*" \
    --exclude "checkpoints/slm_stage2_qwen35_9b_fsdp_v1_recovery1/last.ckpt/*" \
    --local-dir "$TASTE_S_STAGE2_CKPT"
```

The `--exclude` matters: `last.ckpt/` is 8 × 13.4 GB of FSDP DCP shards plus optimizer state
(~107 GB). It is resume state and is not needed for inference.

What you end up using:

| path | what |
|---|---|
| `$TASTE_S_STAGE2_CKPT/valid_best/model.safetensors` | 35,826,930,624 B, sha256 `1dd15e2a3505…` (step 1089) |
| `$TASTE_S_STAGE2_CKPT/model_recipe.yaml` | architecture; `${QWEN_MODEL_DIR}` / `${TASTE_S_TOKENIZER_DIR}` are filled in by the adapter |
| `$TASTE_S_STAGE2_CKPT/source_overrides/` | the `fullduplex_model.py` the checkpoint was trained with, kept for provenance only — see §5 |

Dependencies, both pinned by `MANIFEST.json`:

| | repo | revision |
|---|---|---|
| backbone | `Qwen/Qwen3.5-9B` | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| frozen speech tokenizer | same checkpoint repo, `checkpoints/stage2_fsq_fast_full_float32_8gpu_v1/valid_best` | `39f51cfc26e6a4767528f8bb5dfc9994fe5fd77b` |

⚠️ **The stage-2 tokenizer is a different one from stage-1's `fsq_ourdata`.** Point
`taste_s_tokenizer_dir` at the stage-2 one; reusing the stage-1 tokenizer invalidates everything
on the speech side and does so silently.

## 2. Variables

```bash
export TASTE_S_FD_REPO=/abs/path/TASTE-S_Full_duplex_SLM        # @ 728867b
export TASTE_S_STAGE2_CKPT=/abs/path/ckpt_stage2_9b
export TASTE_S_STAGE2_OUT=/abs/path/for/per-sample/traces
export TASTE_S_SLM_ROOT=/abs/path/IntelliGen/.../slm            # holds pretrained9b/taste_s_tokenizer
export QWEN35_9B=/abs/path/Qwen3.5-9B
export UEA_ROOT=/abs/path/UltraEval-Audio
export PYTHONPATH=$UEA_ROOT
```

`registry/model/taste_slm_fd.yaml` expands these at load time, so it needs no hand-editing. Its
decoding arguments (`temperature: 0.8`, `top_k: 25`, `top_p: 1.0`, `max_words_per_block: 40`,
`max_extra_silent_blocks: 15`) are the author's own `generate.py` CLI defaults, unchanged.

## 3. Running

```bash
cd "$UEA_ROOT"
python audio_evals/main.py --dataset llama-questions-s2t \
    --model taste-slm-fd-9b-stage2 --use_model_pool off
```

Then `speech-triviaqa-s2t` and `speech-web-questions-s2t`, or their sharded views from
`registry/dataset/sharded_qa.yaml`.

⚠️ **Do not use the `CUDA_VISIBLE_DEVICES=N` idiom here.** The GPU is chosen by `device:` in the
registry entry and passed straight to `load_models()`. The author's `generate.py:463` and
`inference.py:52` *overwrite* `CUDA_VISIBLE_DEVICES` with `str(--gpu)`, so an outer value is
discarded. (This adapter never calls their `main()`, so it is unaffected — but set the GPU in
`device:`, not in the environment.) This is the opposite of the stage-1 likelihood scripts, which
take a physical index in `--gpu`.

## 4. How the answer is extracted — and how to read the result

Stage-2 is block-based: in each 0.8 s block the model decides **for itself** whether to speak, via
its trained `spk_head`. We never force speaking and never inject a prompt suffix.

> **Extraction rule:** concatenate `text` from every block with `speaking == True`, in block order.

The consequence is that **silence is a valid outcome, not an error**. An item where the model never
spoke is recorded as `status: "ok"` with `empty_response: true`, and it counts in the denominator.
Per-sample traces land in `$TASTE_S_STAGE2_OUT/uea_traces/<dataset>/<split>/<sample_id>/trace.jsonl`
(runs before 2026-09-19 used `uea_traces/<sample_id>/`, which let datasets overwrite each other).

So the accuracy figure alone is badly misleading here. On LlamaQ:

| | |
|---|---|
| acc (`qa-exist-match`) | 10/300 = **3.33 %** |
| coverage / errors | 300/300 · 0 |
| never spoke | 120/300 = **40.0 %** |
| acc among items where it did speak | 10/180 = 5.56 % |
| truncated at the trailing-block cap | 87/180 = 48 % |

Always report the never-spoke rate and the truncation rate next to the accuracy. For comparison,
stage-1 9B d=1 scores 16.33 % on the same benchmark, so this is a large regression; the analysis is
in `STAGE2_9B_EVALUATION.md`.

## 5. Two provenance notes

**Source override.** `MANIFEST.source_overrides` ships the `train_slm/fullduplex_model.py` the
checkpoint was trained with (upstream `61a1bd5d`). It differs from repo HEAD by 141 lines, all of
them training-loop / logging / FSDP hygiene — rank-0 print guards, `sync_dist` on a memory gauge, an
`all_reduce` in gradient-norm logging, an early return in `save_valid_best`, an FSDP `state_dict()`
collective-call fix. The modules and parameter definitions are identical and none of the differing
paths execute at inference, so we run repo HEAD and keep the override only for provenance.

**Stage-2 speech quality.** The generated speech does not match the generated text: teacher-forced
synthesis is fine, so the fault is in the generated taste codes, not the vocoder. The text stream
reported here is therefore the model's own text, not an ASR transcript of its audio. Section 14 of
`STAGE2_9B_EVALUATION.md` has the diagnosis.

## 6. S2S — scoring what the model says

Entry `taste-slm-fd-9b-stage2-s2s` (`registry/model/taste_slm_fd_s2s.yaml`): the same checkpoint,
code, tokenizer and decoding as `taste-slm-fd-9b-stage2`, plus `synthesize_audio: true` and
`seed: 42`. General S2S setup (local whisper, its venv) is in `README_TASTE_S.md` §S2S.

```bash
cd "$UEA_ROOT"
tools/run_s2s_qa.sh 0 taste-slm-fd-9b-stage2-s2s llama-questions-s2t
# -> res/taste-slm-fd-9b-stage2-s2s/llama-questions-s2t/<ts>_s2s.jsonl   (S2S)
#    res/taste-slm-fd-9b-stage2-s2s/llama-questions-s2t/<ts>_s2t.jsonl   (S2T of the same generation)
```

⚠️ **Device works the other way round from §3.** The S2S entry uses `device: cuda:0` and the
launcher exports `CUDA_VISIBLE_DEVICES=<gpu>`, because the isolated whisper subprocess takes its
device from that variable — this puts model and whisper on one card (~51 GB). The adapter still
never calls the author's `main()`, so nothing overwrites the variable.

**What gets transcribed.** `run_one(skip_audio=False)` renders the author's own `agent_audio.wav`:
agent channel only, each speaking span through the tokenizer's unit decoder + vocoder, each silent
block as exactly 0.8 s of zeros. The adapter cuts the leading and trailing silent blocks off
(`agent_audio_spoken.wav`) and asserts every cut sample is zero, so the cut can never remove speech.
The leading run is the model listening to the question — seconds of digital silence that invite
whisper to hallucinate. Interior gaps are kept. Voice: the author's benchmark default, a zero
speaker embedding.

**Never spoke.** The adapter returns `audio: ""`, which `speech2text-local-allow-empty` turns into
an empty transcript without calling whisper: scored wrong, kept in the denominator, exactly as the
S2T arm treats an empty text.

**Pairing.** Decoding is sampled, so a separate S2S run would not reproduce the S2T run's text.
The launcher's second step re-scores the S2S run's own records as S2T (`--inf_file`), so S2S − S2T
is measured on identical answers. `seed: 42` seeds each sample from `sha256(seed, sample_id)`,
covering generation and the unit decoder. The published S2T numbers in §4 came from an unseeded
run, so compare S2S with the paired `_s2t` file, not with §4.

**Per-sample record** (`inference` → `audio_synthesis`): `spoke_blocks`, `first/last_spoken_block`,
`full_sec`, `asr_sec`, and `drift_sec` — how far the rendered speech overran its block budget.

**Expect S2S ≈ 0**, and say why when reporting it. `STAGE2_9B_EVALUATION.md` §14: this checkpoint's
generated taste codes do not carry its generated text (its own CTC head recovers 1–5 tokens from
8–22 s of audio), while teacher-forced synthesis through the same path is intelligible. On the
first five LlamaQ items: text *"Well, this is the first president. I'm pretty sure it was
Jefferson"* → whisper *"with this present phrase with this separate"*. That is a model defect,
not an evaluation artifact.

**Cost.** Items where the model stays silent cost about what S2T costs; items where it speaks add
unit-decoder synthesis and whisper. Measured on 5 LlamaQ items (A100-80GB, GPU to
itself): **~25 s** for an item where the model spoke, **~3 s** for a silent one, plus ~5 min to
load. With the S2T silence rates (40 % / 84 % / 31 %) that is roughly 1.5 h for LlamaQ, 2 h for
TriviaQA and 10 h for WebQ on one GPU — shard WebQ. These are estimates from 5 items.

## 7. AlpacaEval

The S2S entry above also runs AlpacaEval (open-ended, GPT-judged) with no changes — generation is
`tools/run_alpaca_eval.sh generate <gpu> taste-slm-fd-9b-stage2-s2s`, then `judge`. Setup, the
alignment with upstream UEA and how to read the scores: `README_TASTE_S.md` §AlpacaEval. Long
open-ended answers make the trailing-block cap (`max_extra_silent_blocks: 15`, 12 s after the
question) bind more often than on QA; it stays at the author's default, so report the truncation
rate with the score.

## 8. Not done

SALMon and StoryCloze were **not** run on the stage-2 model. Neither were Full-Duplex-Bench v1/v1.5.
