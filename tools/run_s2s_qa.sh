#!/usr/bin/env bash
# ADDED BY US. Spoken QA scored as S2S AND S2T from ONE generation.
#
#   tools/run_s2s_qa.sh <gpu> <model> <dataset> [extra main.py args, e.g. --limit 5]
#
#   <gpu>      physical GPU index; exported as CUDA_VISIBLE_DEVICES so the model (cuda:0) and the
#              isolated whisper subprocess share that card
#   <model>    an S2S registry entry: taste-slm-{9b-v2,4b-datav2,4b-kl,9b-d1}-s2s (stage-1) or
#              taste-slm-fd-9b-stage2-s2s (full-duplex)
#   <dataset>  a normal -s2t entry, or one shard of it (speech-web-questions-s2t-sh0of5, ...)
#
# Writes, under res/<model>/<dataset>/:
#   <ts>_s2s.jsonl  generation + synthesis + whisper -> qa-exist-match      (the S2S score)
#   <ts>_s2t.jsonl  the SAME inference records re-scored from `text`        (the paired S2T score)
# Step 2 replays step 1's records via --inf_file and never generates (the adapters load the model
# lazily, so it takes seconds). Run from the repo root with the environment of README_TASTE_S.md.
set -euo pipefail
[ $# -ge 3 ] || { sed -n '4,16p' "$0"; exit 2; }
GPU=$1; MODEL=$2; DATASET=$3; shift 3
PY=${PY:-python}
TS=$(date -u +%Y-%m-%d_%H-%M-%S)
OUT=res/$MODEL/$DATASET
export CUDA_VISIBLE_DEVICES=$GPU

# `allow-empty` differs from upstream-style `s2s-aqa-local` only when the adapter returns
# `audio: ""` (a full-duplex model that never spoke): that becomes an empty transcript, scored wrong
# inside the denominator, instead of a failure. Any non-empty audio takes the identical path.
echo "[1/2] S2S  $MODEL / $DATASET on gpu$GPU  -> $OUT/${TS}_s2s.jsonl"
"$PY" -u audio_evals/main.py --dataset "$DATASET" --model "$MODEL" \
    --task s2s-aqa-local-allow-empty --use_model_pool off --save "${TS}_s2s" "$@"

echo "[2/2] S2T  re-score of the same generation   -> $OUT/${TS}_s2t.jsonl"
"$PY" -u audio_evals/main.py --dataset "$DATASET" --model "$MODEL" \
    --task loose-aqa --inf_file "$OUT/${TS}_s2s.jsonl" --use_model_pool off --save "${TS}_s2t" "$@"
