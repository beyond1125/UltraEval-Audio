# TASTE-S additions to UltraEval-Audio

This fork adds TASTE-S SLM support to UltraEval-Audio. **Base commit:
`637cc336f8495c164eb837ae8e8c01737bc91fef`** (2026-08-25) — every result we have reported was
produced against that commit. Upstream's LICENSE is unchanged.

The evaluation chain itself is **entirely upstream**: task `loose-aqa`, prompt `direct-aqa`,
`extract_text`, evaluator `qa-exist-match`, aggregation `acc`. We add a model, a dataset view, and
two compatibility fixes. No scoring code is touched.

## Files

### Compatibility fixes to upstream (metric-neutral)

| file | patch | what |
|---|---|---|
| `audio_evals/utils.py` | `patches/PATCH-001-pandas-applymap.diff` | pandas 3.0 removed `DataFrame.applymap`. Excel export only, runs **after** scoring. |
| `registry/dataset/webQ.yaml` | `patches/PATCH-002-webq-f_name.diff` | `f_name:` → `name:`; the HF loader takes `name`. Without it SpeechWebQuestions cannot load. |

Both modified files are copied under `MODIFIED_UPSTREAM/` for reference — apply the diffs to
upstream rather than copying those over blindly.

### Model adapters (new)

| file | for |
|---|---|
| `audio_evals/models/taste_slm.py` | **pretrain / stage-1** SLMs (0.8B, 9B d=1, 4B/9B datav2, KL 4B) |
| `audio_evals/models/taste_slm_fd.py` | **stage-2 full-duplex** SLM — setup and results in `README_FULL_DUPLEX.md` |

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
`taste_slm_4b_kl.yaml` (KL 4B) · `taste_slm_fd.yaml` (stage-2 full-duplex).

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

# stage-2 full-duplex
CUDA_VISIBLE_DEVICES=0 python audio_evals/main.py \
    --dataset llama-questions-s2t --model taste-slm-fd-9b-stage2 --use_model_pool off
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
