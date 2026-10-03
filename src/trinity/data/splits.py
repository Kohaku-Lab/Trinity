"""Deterministic train / dev splits over the rows of the Lance train store.

The dev split has two disjoint parts, both disjoint from train:

* ``dev_per_n`` -- ``per_n_k`` layouts per block count (21..120), uniform over N like the
  official 100-case set.
* ``dev_random`` -- ``random_size`` layouts drawn uniformly from the remaining rows,
  following the train distribution of block counts.

Row ids are derived from a fixed seed and cached as JSON, keyed on the split parameters.
The official 100-case set is a separate artifact and never part of these splits.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class DataSplits:
    """Row-id lists over the Lance store (disjoint), tagged with the params that built them."""

    train: list[int]
    dev_per_n: list[int]
    dev_random: list[int]
    params: dict | None = None  # {n_rows, per_n_k, random_size, seed} -- cache key

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "train": self.train,
                    "dev_per_n": self.dev_per_n,
                    "dev_random": self.dev_random,
                    "params": self.params,
                }
            )
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "DataSplits":
        d = json.loads(Path(path).read_text())
        return cls(
            train=d["train"],
            dev_per_n=d["dev_per_n"],
            dev_random=d["dev_random"],
            params=d.get("params"),
        )


def make_splits(
    block_counts: np.ndarray,
    per_n_k: int = 10,
    random_size: int = 1000,
    seed: int = 20090220,
) -> DataSplits:
    """Split the rows of ``block_counts`` (the per-row ``n``) into train and the two dev parts.

    ``per_n_k`` rows per distinct block count form ``dev_per_n``, ``random_size`` rows of
    the rest form ``dev_random``, everything else is train. Deterministic in ``seed``.
    """
    rng = np.random.default_rng(seed)
    n_rows = len(block_counts)
    held = np.zeros(n_rows, dtype=bool)

    dev_per_n: list[int] = []
    for n in np.unique(block_counts):
        rows = np.nonzero(block_counts == n)[0]
        take = min(per_n_k, len(rows))
        chosen = rng.choice(rows, size=take, replace=False)
        dev_per_n.extend(int(i) for i in chosen)
        held[chosen] = True

    remaining = np.nonzero(~held)[0]
    rand_take = min(random_size, len(remaining))
    dev_random = [int(i) for i in rng.choice(remaining, size=rand_take, replace=False)]
    held[dev_random] = True

    train = [int(i) for i in np.nonzero(~held)[0]]
    params = {
        "n_rows": n_rows,
        "per_n_k": per_n_k,
        "random_size": random_size,
        "seed": seed,
    }
    return DataSplits(
        train=train,
        dev_per_n=sorted(dev_per_n),
        dev_random=sorted(dev_random),
        params=params,
    )


def load_or_make_splits(
    block_counts: np.ndarray,
    cache_path: str | Path,
    per_n_k: int = 10,
    random_size: int = 1000,
    seed: int = 20090220,
) -> DataSplits:
    """The cached split at ``cache_path`` when its ``{n_rows, per_n_k, random_size, seed}``
    match, else a freshly built split (written to ``cache_path``)."""
    cache_path = Path(cache_path)
    want = {
        "n_rows": len(block_counts),
        "per_n_k": per_n_k,
        "random_size": random_size,
        "seed": seed,
    }
    if cache_path.exists():
        cached = DataSplits.from_json(cache_path)
        if cached.params == want:
            return cached
    splits = make_splits(
        block_counts, per_n_k=per_n_k, random_size=random_size, seed=seed
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    splits.to_json(cache_path)
    return splits
