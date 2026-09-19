# TASTE-S additions to UltraEval-Audio

This fork adds TASTE-S SLM support to UltraEval-Audio. **Base commit:
`637cc336f8495c164eb837ae8e8c01737bc91fef`** (2026-08-25) — every result we have reported was
produced against that commit. Upstream's LICENSE is unchanged.

The evaluation chain itself is **entirely upstream**: task `loose-aqa`, prompt `direct-aqa`,
`extract_text`, evaluator `qa-exist-match`, aggregation `acc`. We add model adapters, dataset
views, a local-whisper S2S chain (§ [S2S](#s2s--speech-in-speech-out)), and three compatibility
fixes. No scoring code is touched.

## Files

### Compatibility fixes to upstream (metric-neutral)

| file | patch | what |
|---|---|---|
| `audio_evals/utils.py` | `patches/PATCH-001-pandas-applymap.diff` | pandas 3.0 removed `DataFrame.applymap`. Excel export only, runs **after** scoring. |
| `registry/dataset/webQ.yaml` | `patches/PATCH-002-webq-f_name.diff` | `f_name:` → `name:`; the HF loader takes `name`. Without it SpeechWebQuestions cannot load. |
| `audio_evals/isolate.py` | `patches/PATCH-003-isolate-pythonpath.diff` | `@isolated` models (whisper) run in their own venv, but `Popen` inherited the parent's `PYTHONPATH`; an overlay built for another python then broke `transformers.models.whisper` (`cannot import name '_regex'`). The isolated command now unsets `PYTHONPATH`. |

In this branch the patched files are simply edited in place; `patches/` holds the same changes as
diffs, for applying to a clean upstream checkout.

### Model adapters (new)

| file | for |
|---|---|
| `audio_evals/models/taste_slm.py` | **pretrain / stage-1** SLMs (0.8B, 9B d=1, 4B/9B datav2, KL 4B) |
| `audio_evals/models/taste_slm_fd.py` | **stage-2 full-duplex** SLM — setup and results in `README_FULL_DUPLEX.md` |
| `audio_evals/models/taste_slm_tb.py` | **turn-based** SLM, QA only — setup in `README_TURNBASED.md` |

`taste_slm.py` and `taste_slm_fd.py` also carry an optional **S2S arm** (off unless the registry
entry asks for it); see [S2S](#s2s--speech-in-speech-out).

Per-sample outputs (S2S wavs, full-duplex / turn-based traces) are written under
`<dir>/<dataset>/<split>/<sample_id>`, derived from the question's own audio path. Every benchmark
numbers its rows from 0, so a layout keyed by `sample_id` alone lets two datasets running at once
overwrite each other's files.

Both are audio-in only: they raise if the prompt carries text, so an oracle transcript cannot leak
in. Both return a JSON string so `extract_text` (`JsonExtract key=text`) works — a bare `"42"` would
otherwise `json.loads` to an int and crash `d["text"]`, losing every numeric answer.

### Dataset sharding (new)

`audio_evals/dataset/sharded_hf.py` + `registry/dataset/sharded_qa.yaml`.

The TASTE-S rollout is **GIL-bound**, so UEA's thread-based `IsolatedModelPool` does not scale —
measured on a 4B: **8 workers ≈ 32 s/item, slower than a single GPU**; 3 workers ≈ 14 s/item. What
scales is one OS **process** per GPU. But a single benchmark could not be split, because `--limit`
takes a prefix only (`quiz[:limit]`) with no offset, which made SpeechWebQuestions a ~7.6 h serial
floor.

`ShardedHuggingface` supplies the missing offset: `res[shard_id::num_shards]` partitions exactly.
Recover the benchmark total as `sum(matches)/sum(n)` — identical to the unsharded denominator.
Verified: WebQ shard sizes `[407, 407, 406, 406, 406]` = 2032, no overlap.

### Registry entries (new)

`registry/model/taste_slm.yaml` (0.8B, 9B d=1) · `taste_slm_datav2.yaml` (4B, 9B) ·
`taste_slm_4b_kl.yaml` (KL 4B) · `taste_slm_9b_v2.yaml` (re-uploaded 9B, 9B KL) ·
`taste_slm_fd.yaml` (stage-2 full-duplex) · `taste_slm_tb.yaml` (turn-based, sampled) ·
`taste_slm_tb_greedy.yaml` (turn-based, greedy).

S2S: `registry/model/taste_slm_s2s.yaml` (stage-1) · `taste_slm_fd_s2s.yaml` (full-duplex) ·
`whisper_local.yaml` · `registry/process/speech_local.yaml` · `registry/eval_task/aqa_s2s_local.yaml`
· `audio_evals/process/speech_allow_empty.py` · `tools/run_s2s_qa.sh`.

Diagnostic views (never for reported scores): `registry/dataset/s2s_diag.yaml` over
`audio_evals/dataset/idlist_hf.py` — named LlamaQ rows, e.g. the 24 the corrected 9B got right in
S2T.

**For end-to-end run instructions see `QUICKSTART.md` at the root of this sharing bundle**
(stage-1 / pretrain), or **`README_FULL_DUPLEX.md`** in this directory (stage-2 full-duplex, which
needs the author's repo and a different speech tokenizer). What follows is the reference for this
fork's own files.

## Setup

```bash
python -m pip install -r requirements.txt          # upstream deps
export UEA_ROOT=$PWD
export TASTE_S_SLM_ROOT=<IntelliGen>/egs/taste_s/ljh_exp/finetune_qwen3_5/slm
export TASTE_S_BUNDLES=<dir holding bundle_*/>
export PYTHONPATH=$UEA_ROOT
```

The adapters expand `${VAR}` in the registry entries and in the model profiles, so nothing has to be
edited by hand. Plain absolute paths still work.

## Running

```bash
# pretrain / stage-1, one process on one GPU
CUDA_VISIBLE_DEVICES=0 python audio_evals/main.py \
    --dataset llama-questions-s2t --model taste-slm-4b-kl --use_model_pool off

# sharded across 5 GPUs (one process each) -- then sum matches/n over the shards
for k in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=$k python audio_evals/main.py \
      --dataset speech-web-questions-s2t-sh${k}of5 --model taste-slm-4b-kl \
      --use_model_pool off &
done; wait

# stage-2 full-duplex (S2T) -- the GPU comes from `device:` in taste_slm_fd.yaml, NOT from
# CUDA_VISIBLE_DEVICES (README_FULL_DUPLEX.md §3); its S2S entry works the other way round
python audio_evals/main.py \
    --dataset llama-questions-s2t --model taste-slm-fd-9b-stage2 --use_model_pool off

# any S2S entry: S2S + the paired S2T from one generation (see "S2S" below)
tools/run_s2s_qa.sh 0 taste-slm-9b-v2-s2s llama-questions-s2t
```

### Two things that will bite you

**Device selection is inconsistent between harnesses.** UEA's `main.py` never touches
`CUDA_VISIBLE_DEVICES`, so its device comes from that variable and there is no `--gpu` flag. The
IntelliGen likelihood scripts (`eval_salmon.py:459`, `eval_storycloze.py:430`) do the opposite —
they *overwrite* `CUDA_VISIBLE_DEVICES` from `--gpu`, so there the physical index goes in `--gpu` and
`CUDA_VISIBLE_DEVICES` must be left unset. Setting both put a 9B instance on the wrong GPU next to
another job and took that card to 76.6/81.9 GB.

**Always pass `--use_model_pool off`.** See the sharding note above; the pool makes things slower.
If you do use the pool with fewer workers than visible GPUs, UEA hands each instance a comma-joined
string (`"0,1,2"`) — `taste_slm.py` now tolerates that, but earlier it raised
`ValueError: invalid literal for int()` and recorded every item as an error.

## S2S — speech in, speech out

S2S scores what the model **says**, not what it writes: its answer is synthesised to a wav,
transcribed by whisper-large-v3, and scored by the same `qa-exist-match` + `acc` as S2T. The chain
is upstream's `s2s-aqa` with the transcription step pointed at a local whisper:

```
S2T: extract_text                                     -> qa-exist-match -> acc
S2S: extract_audio -> speech2text-local[-allow-empty] -> qa-exist-match -> acc
```

### One generation, two scores

UEA runs one post-processing chain per task, so one run yields one score. But `--inf_file`
replays a previous run's saved `inference` records instead of calling the model, and every S2S
adapter output carries **both** `text` and `audio`. So:

1. generate once and score S2S (`--task s2s-aqa-local-allow-empty`);
2. re-score the same records as S2T (`--task loose-aqa --inf_file <run 1>.jsonl`) — no generation,
   and the adapters load the model lazily, so this takes seconds.

S2S − S2T is then the cost of synthesis + ASR on **the same answers**, item by item. That matters
most for sampled decoding (full-duplex), where two separate runs would not produce the same text.
`tools/run_s2s_qa.sh` does both steps:

```bash
tools/run_s2s_qa.sh <gpu> <s2s-model> <dataset> [extra main.py args]
# -> res/<model>/<dataset>/<ts>_s2s.jsonl   and   <ts>_s2t.jsonl
```

`allow-empty` only differs from `s2s-aqa-local` when an adapter returns `audio: ""` (a
full-duplex model that never spoke): that becomes an empty transcript, scored wrong inside the
denominator, as S2T scores an empty text. Any real audio takes the identical path.

### Setup (once)

* **whisper-large-v3, local.** `registry/model/whisper_local.yaml` points at an absolute directory
  (UEA's registry does no `${VAR}` expansion) — edit `path:` to your copy. Nothing is downloaded if it
  exists; upstream `whisper.yaml` would download `openai/whisper-large-v3`.
* **Its own venv.** Whisper runs `@isolated` in `envs/whisper` (python 3.10, `transformers==4.49.0`,
  from `audio_evals/lib/whisper/requirements.txt`), built on first use. PATCH-003 is what keeps the
  parent's `PYTHONPATH` out of it.
* **GPU.** `CUDA_VISIBLE_DEVICES` decides where whisper runs, so the launcher exports it and the S2S
  entries use `cuda:0`: model and whisper share one card. A 9B + whisper needs ~51 GB.

### Stage-1 (pretrain) models

Entries `taste-slm-{9b-v2,4b-datav2,4b-kl,9b-d1}-s2s` in `registry/model/taste_slm_s2s.yaml`, with
profiles `eval/configs/tastes_slm_*_s2s.yaml` in IntelliGen. Extra variables:

```bash
export TASTE_S_S2S_OUT=/abs/path/for/s2s/wavs     # wavs -> $TASTE_S_S2S_OUT/<tag>/<dataset>/<split>/<id>.wav
# campplus (questioner voice) is read from $TASTE_S_SLM_ROOT/pretrained/cosyvoice
```

What is different from S2T, and why:

* **`tail_steps: -1`** — `text[t]`'s speech lands at `t + delay`, so after the last answer token
  there are still `delay` frames of its speech to emit. S2T profiles use `tail_steps: 0` (correct:
  S2T never reads the speech stream), which would leave the audio clipped. The adapter raises once,
  naming the profile, if the flush did not run, and `eval/vocode.py` refuses incomplete inputs.
* **Nothing else.** The profiles differ from their S2T twins in exactly `tail_steps` and
  `retain_synthesis_inputs`. Speech sampling is untouched on purpose: `speech_feedback: sample`
  feeds the sampled speech token back into the text stream, so changing it would change the TEXT.
* **Voice** — the questioner's (campplus over the question audio). The checkpoint has no agent
  voice and these benchmarks ship no reference speaker.
* Synthesis uses `eval/vocode.py`, the same code as the `diagnostics/spokenqa_audio.py` listening
  dump, so what is scored is what was listened to.

Example — the corrected 9B on four GPUs (0–3). One 9B + whisper per card; two do not fit in 80 GB.
Run from the repo root with the environment above:

```bash
S=tools/run_s2s_qa.sh; M=taste-slm-9b-v2-s2s
( $S 0 $M speech-triviaqa-s2t-sh0of2      ; $S 0 $M llama-questions-s2t ) > s2s_gpu0.log 2>&1 &
( $S 1 $M speech-triviaqa-s2t-sh1of2      ; $S 1 $M speech-web-questions-s2t-sh4of5 ) > s2s_gpu1.log 2>&1 &
( $S 2 $M speech-web-questions-s2t-sh0of5 ; $S 2 $M speech-web-questions-s2t-sh1of5 ) > s2s_gpu2.log 2>&1 &
( $S 3 $M speech-web-questions-s2t-sh2of5 ; $S 3 $M speech-web-questions-s2t-sh3of5 ) > s2s_gpu3.log 2>&1 &
wait
```

Recover each benchmark as `sum(matches)/sum(n)` over its shards, separately for the `_s2s` and
`_s2t` files. Before committing GPUs for days, a cheap first measurement is the **24 LlamaQ items
the model got right in S2T** (`--dataset llama-questions-s2tcorrect24`): it isolates what synthesis +
ASR lose. That is a diagnostic — the rows were chosen with the reference — never a reported score.

**Cost.** S2S is dominated by the unit decoder, which generates ~25 units per text token.
Stage-1 has no trained stop token, so answers run to the 64-token cap and usually repeat themselves;
in S2S that becomes ~30 s of audio and ~750 units per item (cap 768). Measured on the corrected 9B, A100-80GB,
GPU to itself: **~40 s per item** once loaded (S2T: ~14 s), plus a one-off 4–6 min to load the SLM
and start whisper. For the whole 9B that is 3356 items ≈ 37 GPU-hours; the 4-GPU layout below
balances to **~10 h** wall-clock (per GPU: 9.0 / 10.2 / 9.0 / 9.0 h). Two jobs on one GPU do not fit.

Record the unit-cap hit rate (`audio_synthesis.hit_unit_cap` in each inference record) rather than
raising the cap: a higher cap only produces longer repetition.

### Full-duplex

See `README_FULL_DUPLEX.md` §S2S. Entry `taste-slm-fd-9b-stage2-s2s`; same launcher.

## Checking coverage

UEA's printed `acc(%)` uses a **success-only** denominator (`eval_task.py` drops failures before
averaging). Verify independently:

```bash
python tools/verify_coverage.py --result_jsonl res/<model>/<dataset>/<ts>.jsonl \
       --expected_n 300 --out_dir coverage/<model>_<dataset>
```

Give each benchmark its **own** `--out_dir`: the filename it writes is fixed, so reusing one
directory silently overwrites the previous benchmark's report.

It also breaks out statuses. That is how we found the KL 4B emitting `status: empty` on 32–48 % of
items — 64 generated tokens that are **all newlines**, which the accuracy alone hides.
