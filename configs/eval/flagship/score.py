"""Soft metric vector of the flagship and no-aux generation caches (every shard).

Run::

    kogine run scripts/eval/score.py --config configs/eval/flagship/score.py
"""

GEN_DIR = "outputs/gen"
RUNS = ["flagship", "no_aux"]
SHARDS = []
WORKERS = 24
OUT = "outputs/gen/scores_flagship.json"
