# Spoken QA for the TURN-BASED TASTE-S SLM

Same three knowledge-QA benchmarks and the same upstream scoring chain as the stage-1 and
full-duplex models — task `loose-aqa`, prompt `direct-aqa`, evaluator `qa-exist-match`,
aggregation `acc`, on pinned UEA commit `637cc336`. Nothing in the scoring path is ours.

QA only. SALMon and StoryCloze are not part of this.

**Jinhui's repo needs no changes.** The adapter imports his `generate.load_models()` and calls
`run_one(..., skip_audio=True)` unmodified; it never calls his `main()`.

| file | role |
|---|---|
| `audio_evals/models/taste_slm_tb.py` | the adapter |
| `registry/model/taste_slm_tb.yaml` | model id **`taste-slm-tb-4b`** |

## Setup

```bash
git clone git@github.com:Jensen-JinhuiLiu/TASTE-S_Full_duplex_SLM.git
cd TASTE-S_Full_duplex_SLM && git checkout 5cbcca7
export TASTE_S_TB_EVAL=$PWD/train_slm/evaluation/turnbased_eval
```

Checkpoint — a **4B**, and its two dependencies are the same ones the stage-1 4B models use, so if
you have those nothing else is needed:

```bash
DL=/abs/path/ckpt_tb_4b
hf download JimHue/TASTE-S-SLM \
    --revision b863fe9827ed0408a2376fb37038e9ddbe4fd12f \
    --include "slm/turnbased_4b_fsdp_eob_w1_protected/*" \
    --local-dir "$DL"
# --include keeps the repo's own two levels, and this is the path that has
# training_config.yaml one level ABOVE valid_best -- which is where load_models looks:
export TASTE_S_TB_CKPT=$DL/slm/turnbased_4b_fsdp_eob_w1_protected
```

| | |
|---|---|
| weights | `valid_best/model.safetensors`, 19,373,134,284 B, step 1314 |
| recipe | `training_config.yaml`, one level **above** `valid_best/` — which is exactly where `load_models` looks by default, so `recipe:` can stay unset |
| backbone | `Qwen/Qwen3.5-4B` (named by the checkpoint's own config) |
| speech tokenizer | `fsq_ourdata/valid_best` — the **same** frozen FSQ tokenizer as the stage-1 4B models, and *not* the stage-2 full-duplex one |

The checkpoint was trained with `text_kl` (weight 0.5, frozen teacher). `turnbased_loader` disables
it for inference itself and prints `[tb] text-KL disabled for inference`. KL is training-only; no
special handling is needed.

```bash
export TASTE_S_TB_OUT=/abs/path/for/traces
export QWEN35_4B=/abs/path/Qwen3.5-4B
export TASTE_S_BUNDLES=/abs/path/holding/tokenizer_fsq_ourdata
export UEA_ROOT=/abs/path/UltraEval-Audio
export PYTHONPATH=$UEA_ROOT
```

## Running

```bash
cd "$UEA_ROOT"
python audio_evals/main.py --dataset llama-questions-s2t \
    --model taste-slm-tb-4b --use_model_pool off
```

Then `speech-triviaqa-s2t` and `speech-web-questions-s2t`, or their sharded views from
`registry/dataset/sharded_qa.yaml`. Recover a sharded total as `sum(matches)/sum(n)`, never the
mean of shard accuracies.

⚠️ **Do not use the `CUDA_VISIBLE_DEVICES=N` idiom.** The GPU comes from `device:` in the registry
entry and is passed straight to `load_models()`. The author's `generate.py:226` and
`rollout.py:223` *overwrite* `CUDA_VISIBLE_DEVICES` with `str(--gpu)`, so an outer value would be
discarded. The adapter never calls their `main()`, so it is unaffected — but set the GPU in
`device:`.

## How this differs from full-duplex

Same benchmarks, same scoring, **different generation contract**. Turn-based has no 0.8 s block
clock and does not use `spk_head` at all: it receives one complete user turn and emits exactly one
agent turn, ended by the model's own trained `<eob>`.

| | stage-1 | full-duplex | **turn-based** |
|---|---|---|---|
| stop token | not trained — always runs to the 64-token cap | per-block `spk_head` decision | **trained `<eob>`**: val recall 0.955, precision 0.607 |
| can return nothing? | no | **yes** — 40 % of LlamaQ items never spoke | **no** — the first position's EOB logit is forced to −inf ("training has no empty `<spk> <eob>` turn") and the turn is asserted non-empty |
| truncation signal | every item | trailing-block cap | `forced_eob` — the `max_words` cap fired instead of `<eob>` |

So there is no valid-empty-response class to account for here, unlike full-duplex. The metric to
watch instead is **`forced_eob`**, recorded per sample: it is this model's truncation rate.

## Decoding

The author's own **evaluation** defaults from `run.sh`, unchanged: `temperature 0.8`, `top_k 25`,
`top_p 1.0`, `max_words 80`, `eob_logit_bias 0.0`. These are identical to the full-duplex
adapter's values, so the two models are compared under the same decoding.

The bundled `cmd.sh` demo passes `--greedy` instead. **For QA, use greedy:** `run.sh`'s sampling
defaults are for generating DeepDialogue conversational turns, while QA wants the single best answer
— and every stage-1 QA number is greedy. The entry `taste-slm-tb-4b-greedy`
(`registry/model/taste_slm_tb_greedy.yaml`) differs from `taste-slm-tb-4b` only in `greedy: true`:

| LlamaQ (300) | acc | mean answer tokens | `forced_eob` |
|---|---:|---:|---:|
| `taste-slm-tb-4b-greedy` | **87 = 29.00 %** | 26.29 | 0 |
| author's own greedy table | 86 = 28.67 % | 26.22 | 1 |
| `taste-slm-tb-4b` (sampled, T 0.8 / top-k 25) | 55 = 18.33 % | 20.70 | 0 |

The greedy entry reproduces the author's number; on the same items greedy is right 49 times where
sampling is wrong, and the reverse 17 times (exact McNemar p ≈ 0.0001).

`eob_logit_bias` stays at `0.0`. The author exposes it for calibration sweeps; moving it would be
tuning the stopping behaviour against the benchmark.

Because the default is sampling, the adapter seeds torch per sample from
`sha256(seed, sample_id)`. A single global seed would make a sharded run irreproducible, since a
sample's draw would depend on how many samples ran before it in that process.

## Audio input contract

The turn-based front end is **not** the same as full-duplex's: channel 0, torchaudio 16 kHz
resample, then int16-grid rounding, matching how the DeepDialogue turn-based shards were built.
Both knobs are exposed (`input_resampler`, `quantize_input`) and default to the author's values.
Changing them would evaluate the checkpoint under a data contract it was not trained for.

## Answer extraction

`result["predicted_text"]` — the author's own decode of the one generated agent turn. Not selected
using the reference, not truncated to a first sentence, not conditional on matching. The full token
and TASTE trace stays in `generation.json` under `trace_dir`, whose path is recorded in the output.

## Reading the results

Check coverage, not just accuracy — UEA's printed `acc(%)` uses a success-only denominator:

```bash
python tools/verify_coverage.py --result_jsonl res/taste-slm-tb-4b/<dataset>/<ts>.jsonl \
       --expected_n 300 --out_dir coverage/tb-4b_<dataset>
```

Report the **`forced_eob` rate** alongside accuracy. `empty_response` is recorded too, but should
always be false; if it is ever true, that is a regression in the author's generation loop, not a
valid outcome.

## Not covered

Multi-turn rollout (`rollout.py`, carrying each predicted `{text, taste}` agent event into the next
prompt) is the author's reference implementation for genuine multi-turn evaluation. The QA
benchmarks here are single-exchange, so this adapter deliberately calls the one-turn path with
empty prior history. SALMon, StoryCloze and Full-Duplex-Bench are out of scope.
