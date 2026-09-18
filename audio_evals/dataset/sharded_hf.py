"""ADDED BY US -- a strided shard view over an upstream Huggingface dataset.

WHY: the TASTE-SLM joint rollout is GIL-bound, so UEA's thread-based IsolatedModelPool does not
scale (measured: 8 workers ~32 s/item, SLOWER than one GPU; 3 workers ~14 s/item). The only thing
that scales is one OS PROCESS per GPU -- but a single benchmark could not be split across processes,
because `--limit` takes a prefix only (`quiz[:limit]`) and there is no offset. That made
speech-web-questions (2032 items x ~13.4 s) a ~7.6 h serial floor on the wall clock.

This class supplies the missing offset. `Huggingface.load()` returns a plain list, so
`res[shard_id::num_shards]` partitions it exactly, deterministically, and with no overlap:
every item belongs to exactly one shard and the shard sizes differ by at most one.

Scoring is unaffected: each shard is evaluated by the SAME upstream task/prompt/evaluator, and the
benchmark total is recovered as sum(matches)/sum(n) over the shards, which equals the unsharded
denominator. Nothing about the metric changes -- this only decides which process computes which row.

Registry usage:

    speech-web-questions-s2t-sh0of5:
      class: audio_evals.dataset.sharded_hf.ShardedHuggingface
      args:
        default_task: loose-aqa
        name: TwinkStart/speech-web-questions
        ref_col: answers
        split: test
        shard_id: 0
        num_shards: 5
"""
from typing import Any, Dict, List

from audio_evals.dataset.huggingface import Huggingface


class ShardedHuggingface(Huggingface):
    def __init__(self, shard_id: int, num_shards: int, **kwargs):
        super().__init__(**kwargs)
        if num_shards < 1:
            raise ValueError(f"num_shards must be >= 1, got {num_shards}")
        if not (0 <= shard_id < num_shards):
            raise ValueError(f"shard_id {shard_id} out of range for num_shards {num_shards}")
        self.shard_id = int(shard_id)
        self.num_shards = int(num_shards)

    def load(self, limit: int = 0) -> List[Dict[str, Any]]:
        # NOTE: slice the FULL dataset, then stride. Applying `limit` first would make the shards
        # cover only a prefix of the benchmark.
        res = super().load(0)
        shard = res[self.shard_id:: self.num_shards]
        return shard[:limit] if limit > 0 else shard
