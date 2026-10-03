"""Run a classical ``SOLVER`` over the dev split and write a generation-style cache.

Writes ``OUT_DIR/<NAME>/free_nfe1.npz`` in the latent layout of ``scripts/eval/generate.py``
(``lat`` = ``(cx/s, cy/s, rho)``, ``bcount``, ``k``), so ``scripts/eval/score.py`` scores a
classical placer with the metric vector of the learned placers. ``K`` solves per case, each with
its own seed, fill the ``k`` draws. Run::

    kogine run scripts/baselines/solve_classical.py --config configs/baselines/solve_sp_sa.py
"""

import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

import trinity_baselines.classical  # noqa: F401  (register: parsac, sp_sa)
from trinity.data.splits import load_or_make_splits
from trinity.floorplan.data import LanceFloorplanStore, find_train_lance
from trinity.floorplan.parameterize import xywh_to_z
from trinity.floorplan.registry import SOLVER, build

TRAIN_LANCE: str | None = None  # None = data/floorset_lite.lance
DEV_PER_N_K: int = 100
DEV_RANDOM_SIZE: int = 2000
SPLIT_SEED: int = 20090220
DEV_LIMIT: int = 0  # >0 keeps only the first DEV_LIMIT dev cases
SOLVER_SPEC: dict = {"name": "sp_sa"}
NAME: str = "sp_sa"
K: int = 4  # solves per case, one seed each
WORKERS: int = 24
OUT_DIR: str = "outputs/gen"

# Per-worker state, set by the pool initializer.
_worker_ids = None
_worker_store = None
_worker_spec = None
_worker_k = None


def dev_ids() -> list[int]:
    """The dev-split row ids."""
    path = str(find_train_lance(TRAIN_LANCE))
    store = LanceFloorplanStore(path)
    splits = load_or_make_splits(
        store.block_counts,
        str(Path(path).parent / "splits.json"),
        per_n_k=DEV_PER_N_K,
        random_size=DEV_RANDOM_SIZE,
        seed=SPLIT_SEED,
    )
    ids = splits.dev_per_n + splits.dev_random
    return ids[:DEV_LIMIT] if DEV_LIMIT > 0 else ids


def init_worker(ids, spec, k, train_lance):
    global _worker_ids, _worker_store, _worker_spec, _worker_k
    _worker_ids, _worker_spec, _worker_k = ids, spec, k
    _worker_store = LanceFloorplanStore(str(find_train_lance(train_lance)))


def solve_case(i: int):
    """Solve dev case ``i`` ``K`` times; return ``(i, n, (K, n, 3) latents)``."""
    inst = _worker_store.instances([_worker_ids[i]])[0]
    latents = np.empty((_worker_k, inst.block_count, 3), dtype=np.float32)
    for j in range(_worker_k):
        solver = build({**_worker_spec, "seed": 1000 * j + 1}, SOLVER)
        xywh = solver.solve(inst).xywh
        latents[j] = xywh_to_z(xywh, inst.area_targets, inst.s)
    return i, inst.block_count, latents


def main():
    ids = dev_ids()
    out = Path(OUT_DIR) / NAME
    out.mkdir(parents=True, exist_ok=True)
    print(
        f"{NAME}: {SOLVER_SPEC} x K={K} over {len(ids)} dev cases on {WORKERS} workers",
        flush=True,
    )
    started = time.perf_counter()
    latents = [None] * len(ids)
    bcount = np.zeros(len(ids), dtype=np.int32)
    context = mp.get_context("forkserver")
    init_args = (ids, SOLVER_SPEC, K, TRAIN_LANCE)
    with context.Pool(WORKERS, initializer=init_worker, initargs=init_args) as pool:
        results = pool.imap_unordered(solve_case, range(len(ids)))
        for done, (i, n, case_latents) in enumerate(results, 1):
            latents[i], bcount[i] = case_latents, n
            if done % 200 == 0 or done == len(ids):
                elapsed = time.perf_counter() - started
                print(f"  [{done}/{len(ids)}]  {elapsed:7.1f}s", flush=True)
    flat = np.concatenate([lat.reshape(-1, 3) for lat in latents], axis=0)
    expected = int((bcount.astype(np.int64) * K).sum())
    if flat.shape[0] != expected:
        raise RuntimeError(
            f"cache has {flat.shape[0]} latent rows, expected {expected}"
        )
    np.savez(out / "free_nfe1.npz", lat=flat, bcount=bcount, k=np.int32(K))
    meta = {
        "run": NAME,
        "solver": SOLVER_SPEC,
        "k": K,
        "n_cases": len(ids),
        "nfe": [1],
        "projection_sets": {"free": []},
        "layout": "lat is case-major then draw; case i draw j is rows "
        "[K*sum(bcount[:i]) + j*n_i : ... + n_i]",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"done in {time.perf_counter() - started:.1f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
