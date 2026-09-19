#!/usr/bin/env bash
# ADDED BY US. AlpacaEval (speech-chatbot-alpaca-eval) for the TASTE-S SLMs, in two phases so the
# free part (generation, GPU) and the paid part (GPT judge) run independently:
#
#   tools/run_alpaca_eval.sh generate <gpu> <model> [extra main.py args]
#       -> res/<model>/speech-chatbot-alpaca-eval/<ts>_gen.jsonl        no API calls
#   tools/run_alpaca_eval.sh judge <gpu> <model> <..._gen.jsonl> [extra main.py args]
#       -> <ts>_s2t.jsonl  judged from the model's text                  needs OPENAI_API_KEY
#          <ts>_s2s.jsonl  judged from whisper of the model's audio      (whisper runs on <gpu>)
#
#   <model>  an entry whose output has `text` AND `audio`: taste-slm-{9b-v2,4b-datav2,4b-kl,9b-d1}-
#            alpaca-s2s (stage-1) or taste-slm-fd-9b-stage2-s2s (full-duplex)
#
# Everything that scores is upstream UEA: dataset `speech-chatbot-alpaca-eval`, prompt `direct-aqa`,
# tasks `glm-alpaca-eval-s2t` / `glm-alpaca-eval`, judge `chatbot_eval` (gpt4o-mini, "[[1-10]]"),
# agg `geval`. The only departures, both forced by the adapters returning JSON with text+audio:
#   S2T  --post_process extract_text        (upstream's is [], which would hand the judge raw JSON)
#   S2S  speech2text-local-allow-empty      (same whisper-large-v3 weights, local; "" = never spoke)
# The judge phase replays the generation's saved `inference` records via --inf_file, so both scores
# judge the SAME answers and nothing is generated twice. Pass the same --limit to both phases.
set -euo pipefail
usage() { sed -n '2,21p' "$0"; exit 2; }
[ $# -ge 3 ] || usage
PHASE=$1; GPU=$2; MODEL=$3; shift 3
DS=speech-chatbot-alpaca-eval
PY=${PY:-python}
export CUDA_VISIBLE_DEVICES=$GPU

case "$PHASE" in
generate)
    TS=$(date -u +%Y-%m-%d_%H-%M-%S)
    echo "[generate] $MODEL / $DS on gpu$GPU -> res/$MODEL/$DS/${TS}_gen.jsonl (no judge)"
    # evaluator/agg `dump` record the answers without scoring them.
    "$PY" -u audio_evals/main.py --dataset "$DS" --model "$MODEL" \
        --task glm-alpaca-eval-s2t --post_process extract_text \
        --evaluator dump --agg dump --use_model_pool off --save "${TS}_gen" "$@"
    ;;
judge)
    [ $# -ge 1 ] || usage
    GEN=$1; shift
    [ -f "$GEN" ] || { echo "no such generation file: $GEN"; exit 1; }
    [ -n "${OPENAI_API_KEY:-}" ] || { echo "OPENAI_API_KEY is not set; the judge is gpt4o-mini"; exit 1; }
    STEM=$(basename "$GEN" .jsonl); STEM=${STEM%_gen}
    echo "[judge 1/2] S2T  $GEN -> res/$MODEL/$DS/${STEM}_s2t.jsonl"
    "$PY" -u audio_evals/main.py --dataset "$DS" --model "$MODEL" \
        --task glm-alpaca-eval-s2t --post_process extract_text \
        --inf_file "$GEN" --use_model_pool off --save "${STEM}_s2t" "$@"
    echo "[judge 2/2] S2S  $GEN -> res/$MODEL/$DS/${STEM}_s2s.jsonl"
    "$PY" -u audio_evals/main.py --dataset "$DS" --model "$MODEL" \
        --task glm-alpaca-eval --post_process extract_audio speech2text-local-allow-empty \
        --inf_file "$GEN" --use_model_pool off --save "${STEM}_s2s" "$@"
    ;;
*) usage ;;
esac
