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
Per-sample traces land in `$TASTE_S_STAGE2_OUT/uea_traces/<sample_id>/trace.jsonl`.

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

## 6. Not done

SALMon and StoryCloze were **not** run on the stage-2 model. Neither were Full-Duplex-Bench v1/v1.5.
