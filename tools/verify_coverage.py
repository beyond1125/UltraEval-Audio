"""
Independent coverage / denominator accounting for a UEA run.

Written because UEA's own aggregation DROPS failed items before averaging
(`audio_evals/eval_task.py`: `res = [item for item in res if item is not None]`),
so its printed `acc(%)` is a SUCCESS-ONLY denominator. We do not change that metric;
we recompute alongside it with the full selected denominator and report both.
"""
import argparse, collections, json, os, sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_jsonl", required=True)
    ap.add_argument("--expected_n", type=int, required=True)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    rows = [json.loads(l) for l in open(a.result_jsonl) if l.strip()]
    by_type = collections.defaultdict(dict)
    for r in rows:
        t = r.get("type")
        if t is None or "id" not in r:
            continue
        by_type[t][r["id"]] = r.get("data")

    ids_prompt = set(by_type["prompt"])
    ids_inf = set(by_type["inference"])
    ids_post = set(by_type["post_process"])
    ids_eval = set(by_type["eval"])
    ids_err = set(by_type["error"])
    all_ids = ids_prompt | ids_inf | ids_post | ids_eval | ids_err

    per_sample, n_match, n_zero, n_bad = [], 0, 0, 0
    statuses = collections.Counter()
    finish = collections.Counter()
    for i in sorted(all_ids):
        ev = by_type["eval"].get(i)
        inf = by_type["inference"].get(i)
        meta = {}
        if isinstance(inf, dict) and isinstance(inf.get("content"), str):
            try:
                meta = json.loads(inf["content"])
            except Exception:
                meta = {}
        statuses[meta.get("status", "<no-status>")] += 1
        finish[meta.get("finish_reason", "<none>")] += 1
        m = ev.get("match") if isinstance(ev, dict) else None
        if m is None:
            n_bad += 1
        elif m:
            n_match += 1
        else:
            n_zero += 1
        per_sample.append(dict(
            id=i, sample_id=meta.get("sample_id"),
            audio_path=meta.get("audio_path"),
            ctc_transcript=meta.get("ctc_transcript"),
            prediction=(ev or {}).get("pred") if isinstance(ev, dict) else None,
            reference=(ev or {}).get("ref") if isinstance(ev, dict) else None,
            match=m, status=meta.get("status"),
            finish_reason=meta.get("finish_reason"), truncated=meta.get("truncated"),
            answer_tokens=meta.get("answer_tokens"),
            joint_steps=meta.get("joint_steps"),
            speech_feedback_steps=meta.get("speech_feedback_steps"),
            effective_delay=meta.get("effective_delay"),
            elapsed_sec=meta.get("elapsed_sec"),
            errored=bool(i in ids_err),
        ))

    n_scored = n_match + n_zero
    out = dict(
        result_jsonl=os.path.abspath(a.result_jsonl),
        expected_selected=a.expected_n,
        counts=dict(
            ids_seen=len(all_ids), prompted=len(ids_prompt), inferred=len(ids_inf),
            post_processed=len(ids_post), evaluated=len(ids_eval), errored=len(ids_err),
            scored=n_scored, match=n_match, no_match=n_zero, unscored=n_bad,
        ),
        coverage_complete=bool(len(ids_eval) == a.expected_n and not ids_err and n_bad == 0),
        missing_ids=sorted(set(range(a.expected_n)) - ids_eval)[:50],
        accuracy_full_denominator=(n_match / a.expected_n) if a.expected_n else None,
        accuracy_full_denominator_pct=(round(100.0 * n_match / a.expected_n, 4)
                                       if a.expected_n else None),
        accuracy_success_only_denominator=(n_match / n_scored) if n_scored else None,
        accuracy_success_only_pct=(round(100.0 * n_match / n_scored, 4) if n_scored else None),
        denominator_note=("UEA's printed acc(%) drops failed items before averaging; "
                          "accuracy_full_denominator uses every selected item, counting an "
                          "error/empty as incorrect."),
        model_statuses=dict(statuses), finish_reasons=dict(finish),
    )
    with open(os.path.join(a.out_dir, "coverage_verification.json"), "w") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(a.out_dir, "per_sample.jsonl"), "w") as f:
        for r in per_sample:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    c = out["counts"]
    print(f"expected selected : {a.expected_n}")
    print(f"prompted/inferred/post/eval : {c['prompted']}/{c['inferred']}/"
          f"{c['post_processed']}/{c['evaluated']}   errored={c['errored']}")
    print(f"scored={c['scored']}  match={c['match']}  no_match={c['no_match']}  unscored={c['unscored']}")
    print(f"coverage_complete : {out['coverage_complete']}")
    print(f"accuracy (FULL denominator {a.expected_n})   : {out['accuracy_full_denominator_pct']} %")
    print(f"accuracy (success-only denominator {c['scored']}) : {out['accuracy_success_only_pct']} %")
    print(f"statuses: {dict(statuses)}")
    print(f"finish_reasons: {dict(finish)}")
    print("->", os.path.join(a.out_dir, "coverage_verification.json"))


if __name__ == "__main__":
    main()
