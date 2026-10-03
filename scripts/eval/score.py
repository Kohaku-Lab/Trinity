"""Score cached generation shards with the soft metric vector.

Reads ``GEN_DIR/<run>/<shard>.npz`` (written by ``generate.py`` /
``generate_prior.py``), decodes every cached draw to boxes, scores the metric columns of
``trinity.floorplan.scoring.vector.COLS`` and writes one JSON with, per ``run|shard``,
the mean over cases of the per-case mean over draws (``mean_k``) and of the draw with
the lowest ``soft_cost`` (``best_k``).

Work is split into ``(run, shard, case chunk)`` tasks on a process pool;
each worker opens its own Lance store and loads only the chunks it scores.

Run::

    kogine run scripts/eval/score.py --config configs/eval/flagship/score.py
"""

import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from trinity.augment_ops import drop_wires
from trinity.data.splits import dev_split_ids
from trinity.floorplan.data import LanceFloorplanStore, find_train_lance
from trinity.floorplan.parameterize import z_to_xywh
from trinity.floorplan.scoring.vector import COLS, metric_vector

TRAIN_LANCE: str | None = None
DEV_PER_N_K: int = 100
DEV_RANDOM_SIZE: int = 2000
SPLIT_SEED: int = 20090220

GEN_DIR: str = "outputs/gen"
RUNS: list[str] = []  # [] = every run directory under GEN_DIR
SHARDS: list[str] = []  # [] = every shard of each run
WIRE_DROP: float = 0.0  # must equal the generation's WIRE_DROP
WIRE_DROP_SEED: int = 20260829

WORKERS: int = 24
CHUNK: int = 1000  # dev cases per task
OUT: str = "outputs/gen/scores.json"

RANK_COL = COLS.index("soft_cost")

# Per-worker state, set by ``init_worker``.
_worker: dict = {}


def dev_ids() -> list[int]:
    _, ids, _ = dev_split_ids(TRAIN_LANCE, DEV_PER_N_K, DEV_RANDOM_SIZE, SPLIT_SEED)
    return ids


def init_worker(ids, gen_dir, wire_drop, wire_drop_seed, chunk, lance) -> None:
    _worker.update(
        ids=ids,
        gen_dir=gen_dir,
        wire_drop=wire_drop,
        wire_drop_seed=wire_drop_seed,
        chunk=chunk,
        store=LanceFloorplanStore(str(find_train_lance(lance))),
        cases={},
    )


def chunk_cases(chunk: int, n_cases: int):
    """The instances of case chunk ``chunk`` of a
    shard holding the first ``n_cases`` dev ids."""
    key = (chunk, n_cases)
    if key not in _worker["cases"]:
        size = _worker["chunk"]
        lo = chunk * size
        cases = _worker["store"].instances(_worker["ids"][lo : min(lo + size, n_cases)])
        if _worker["wire_drop"] > 0:
            seed = _worker["wire_drop_seed"]
            cases = [
                drop_wires(
                    case, _worker["wire_drop"], np.random.default_rng(seed + lo + i)
                )
                for i, case in enumerate(cases)
            ]
        _worker["cases"][key] = cases
    return _worker["cases"][key]


def score_chunk(task):
    """``(run, shard, sum of per-case means, sum
    of per-case best draws, case count)``."""
    run, shard, chunk = task
    data = np.load(Path(_worker["gen_dir"]) / run / f"{shard}.npz")
    lat, bcount, k = data["lat"], data["bcount"], int(data["k"])
    offsets = np.concatenate([[0], np.cumsum(bcount.astype(np.int64) * k)])
    cases = chunk_cases(chunk, int(bcount.shape[0]))
    mean_sum = np.zeros(len(COLS))
    best_sum = np.zeros(len(COLS))
    for local, inst in enumerate(cases):
        i = chunk * _worker["chunk"] + local
        draws = lat[offsets[i] : offsets[i + 1]].reshape(k, int(bcount[i]), 3)
        rows = np.array(
            [
                metric_vector(
                    z_to_xywh(z.astype(np.float64), inst.area_targets, inst.s), inst
                )
                for z in draws
            ]
        )
        mean_sum += rows.mean(axis=0)
        best_sum += rows[int(np.argmin(rows[:, RANK_COL]))]
    return run, shard, mean_sum, best_sum, len(cases)


def list_tasks() -> list[tuple[str, str, int]]:
    """Every ``(run, shard, chunk)`` to score."""
    root = Path(GEN_DIR)
    runs = RUNS or sorted(p.name for p in root.iterdir() if p.is_dir())
    tasks = []
    for run in runs:
        for path in sorted((root / run).glob("*.npz")):
            if SHARDS and path.stem not in SHARDS:
                continue
            n_cases = int(np.load(path)["bcount"].shape[0])
            n_chunks = (n_cases + CHUNK - 1) // CHUNK
            tasks.extend((run, path.stem, chunk) for chunk in range(n_chunks))
    return tasks


def main():
    ids = dev_ids()
    tasks = list_tasks()
    print(f"scoring {len(tasks)} chunk tasks over {len(ids)} dev cases", flush=True)
    sums = {}
    started = time.perf_counter()
    init_args = (ids, GEN_DIR, WIRE_DROP, WIRE_DROP_SEED, CHUNK, TRAIN_LANCE)
    with mp.get_context("forkserver").Pool(
        min(WORKERS, len(tasks)), initializer=init_worker, initargs=init_args
    ) as pool:
        results = pool.imap_unordered(score_chunk, tasks)
        for done, (run, shard, mean_sum, best_sum, n) in enumerate(results, 1):
            key = f"{run}|{shard}"
            mean_acc, best_acc, count = sums.get(key, (0.0, 0.0, 0))
            sums[key] = (mean_acc + mean_sum, best_acc + best_sum, count + n)
            if done % 10 == 0 or done == len(tasks):
                elapsed = time.perf_counter() - started
                print(f"  [{done}/{len(tasks)}] chunks  {elapsed:6.1f}s", flush=True)
    results = {}
    for key, (mean_acc, best_acc, count) in sorted(sums.items()):
        mean_k = (mean_acc / count).tolist()
        best_k = (best_acc / count).tolist()
        results[key] = {"mean_k": mean_k, "best_k": best_k, "n_cases": count}
        print(f"  {key:<40} soft_cost={mean_k[RANK_COL]:.5f}  n={count}")
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps({"cols": COLS, "results": results}, indent=1))
    print(f"done in {time.perf_counter() - started:.1f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
