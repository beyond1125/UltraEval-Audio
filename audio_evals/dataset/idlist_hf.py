"""A dataset view restricted to an explicit list of row indices.

ADDED BY US, alongside `sharded_hf.ShardedHuggingface`. Both exist because upstream's `--limit`
takes a PREFIX only (`quiz[:limit]`) with no way to address particular rows:

  * `ShardedHuggingface` partitions a benchmark across processes -- `res[k::N]`.
  * `IdListHuggingface` (here) selects named rows -- for diagnosis, not for a reported score.

WHY THIS IS NEEDED
    Comparing S2S against S2T on a prefix measures almost nothing: on LlamaQ the 9B answers only
    24/300 items correctly in S2T and the first is row 11, so a 50-row prefix contains exactly ONE
    item whose S2T answer was right. On items the model already got wrong, S2S and S2T are both
    wrong and the comparison cannot separate "synthesis and ASR lost the answer" from "the model
    never had it". Restricting to the S2T-correct rows measures the synthesis+ASR loss directly.

⚠️ A score computed over a hand-picked subset is a DIAGNOSTIC, never a benchmark number: the rows
   are chosen using the reference, so the subset is not representative by construction. Reported
   accuracies must always come from the full dataset (or from `ShardedHuggingface` shards summed as
   sum(matches)/sum(n)).

`ids` are 0-based positions in the loaded split, which is what UEA uses as the record `id`, so they
line up with the `id` field of an earlier run's result JSONL.
"""
from typing import List

from audio_evals.dataset.huggingface import Huggingface


class IdListHuggingface(Huggingface):
    def __init__(self, ids: List[int], **kwargs):
        super().__init__(**kwargs)
        if not ids:
            raise ValueError("IdListHuggingface needs a non-empty `ids` list")
        bad = [i for i in ids if not isinstance(i, int) or i < 0]
        if bad:
            raise ValueError(f"ids must be non-negative ints, got {bad[:5]}")
        # keep the caller's order and drop duplicates, so the log order is predictable
        seen, out = set(), []
        for i in ids:
            if i not in seen:
                seen.add(i)
                out.append(i)
        self.ids = out

    def load(self, limit: int = 0):
        # Slice the FULL dataset first, then select: applying `limit` upstream would cut rows away
        # before we could address them. Same reasoning as ShardedHuggingface.
        res = super().load(0)
        n = len(res)
        over = [i for i in self.ids if i >= n]
        if over:
            raise IndexError(f"ids out of range for a split of {n} rows: {over[:5]}")
        picked = [res[i] for i in self.ids]
        return picked[:limit] if limit > 0 else picked
