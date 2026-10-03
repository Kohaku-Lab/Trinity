"""Distribution metrics of generation caches against the ground-truth dev layouts.

Every cached draw of every run and shard, and the ground-truth layout of every dev case, is
embedded with the density descriptors (grid ``G`` in ``GRIDS``, frame ``bbox`` / ``fixed``)
and the position descriptor. Per ``run|shard`` and descriptor it reports the Fréchet distance
to the ground truth (total, mean term, covariance term) and the unbiased RBF MMD²; for the
descriptor ``TEST_DESC`` it adds a permutation p-value of ``MMD²(run) < MMD²(CONTROL)``.
Embeddings are cached as ``.npy`` under ``EMB_DIR`` and reused by later runs.

Run::

    kogine run scripts/eval/dist_metrics.py --config configs/eval/flagship/dist_metrics.py
"""

import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from trinity.data.splits import load_or_make_splits
from trinity.floorplan.data import LanceFloorplanStore, find_train_lance
from trinity.floorplan.parameterize import z_to_xywh
from trinity.floorplan.scoring.descriptors import (
    density_descriptor,
    position_descriptor,
)
from trinity.floorplan.scoring.distribution import (
    frechet,
    median_bandwidth,
    mmd2,
    mmd_permutation_test,
)

TRAIN_LANCE: str | None = None
DEV_PER_N_K: int = 100
DEV_RANDOM_SIZE: int = 2000
SPLIT_SEED: int = 20090220

GEN_DIR: str = "outputs/gen"
EMB_DIR: str = "outputs/gen/_emb"
RUNS: list[str] = []  # [] = every run directory under GEN_DIR holding all SHARDS
SHARDS: list[str] = ["projected_nfe1", "projected_nfe32"]
CONTROL: str | None = None  # run the permutation test compares against

GRIDS: tuple[int, ...] = (8, 12)
FRAMES: tuple[str, ...] = ("bbox", "fixed")
TEST_DESC: str = "density_g8_bbox"
MMD_SUBSAMPLE: int = 4000  # rows per set for MMD²
TEST_SUBSAMPLE: int = 2000  # rows per set inside the permutation test
N_PERM: int = 500

WORKERS: int = 24
CHUNK: int = 1000  # dev cases per embedding task
OUT: str = "outputs/gen/dist_metrics.json"

# Per-worker state, set by ``init_worker``.
_worker: dict = {}


def dev_ids() -> list[int]:
    lance_path = find_train_lance(TRAIN_LANCE)
    store = LanceFloorplanStore(str(lance_path))
    splits = load_or_make_splits(
        store.block_counts,
        lance_path.parent / "splits.json",
        per_n_k=DEV_PER_N_K,
        random_size=DEV_RANDOM_SIZE,
        seed=SPLIT_SEED,
    )
    return splits.dev_per_n + splits.dev_random


def descriptor_names() -> list[str]:
    names = [f"density_g{g}_{frame}" for g in GRIDS for frame in FRAMES]
    return names + ["position"]


def embed(xywh: np.ndarray, inst) -> list[np.ndarray]:
    """Every descriptor of one layout, in ``descriptor_names`` order."""
    out = [density_descriptor(xywh, inst, G=g, frame=f) for g in GRIDS for f in FRAMES]
    out.append(position_descriptor(xywh, inst))
    return out


def init_worker(ids, gen_dir, chunk, lance) -> None:
    _worker.update(
        ids=ids,
        gen_dir=gen_dir,
        chunk=chunk,
        store=LanceFloorplanStore(str(find_train_lance(lance))),
        cases={},
    )


def chunk_cases(chunk: int, n_cases: int):
    key = (chunk, n_cases)
    if key not in _worker["cases"]:
        size = _worker["chunk"]
        ids = _worker["ids"][chunk * size : min((chunk + 1) * size, n_cases)]
        _worker["cases"][key] = _worker["store"].instances(ids)
    return _worker["cases"][key]


def embed_chunk(task):
    """Embed one chunk of one shard (of the ground truth when ``shard`` is None).

    ``run`` holds the case count for the ground truth. Returns per descriptor an array of
    rows (one per layout).
    """
    run, shard, chunk = task
    rows = [[] for _ in descriptor_names()]
    if shard is None:
        for inst in chunk_cases(chunk, int(run)):
            for d, vec in enumerate(embed(inst.gt_positions.astype(np.float64), inst)):
                rows[d].append(vec)
        return run, shard, chunk, [np.array(r) for r in rows]
    data = np.load(Path(_worker["gen_dir"]) / run / f"{shard}.npz")
    lat, bcount, k = data["lat"], data["bcount"], int(data["k"])
    offsets = np.concatenate([[0], np.cumsum(bcount.astype(np.int64) * k)])
    for local, inst in enumerate(chunk_cases(chunk, int(bcount.shape[0]))):
        i = chunk * _worker["chunk"] + local
        draws = lat[offsets[i] : offsets[i + 1]].reshape(k, int(bcount[i]), 3)
        for z in draws:
            xywh = z_to_xywh(z.astype(np.float64), inst.area_targets, inst.s)
            for d, vec in enumerate(embed(xywh, inst)):
                rows[d].append(vec)
    return run, shard, chunk, [np.array(r) for r in rows]


def embedding_paths(run: str, shard: str | None) -> dict[str, Path]:
    tag = f"gt{run}" if shard is None else f"{run}__{shard}"
    return {name: Path(EMB_DIR) / f"{tag}__{name}.npy" for name in descriptor_names()}


def ensure_embeddings(pool, jobs, n_cases: int) -> None:
    """Compute and cache the embeddings of every ``(run, shard)`` of ``jobs`` not on disk."""
    todo = [
        (run, shard)
        for run, shard in jobs
        if not all(p.exists() for p in embedding_paths(run, shard).values())
    ]
    if not todo:
        return
    n_chunks = (n_cases + CHUNK - 1) // CHUNK
    tasks = [(run, shard, c) for run, shard in todo for c in range(n_chunks)]
    parts = {}
    started = time.perf_counter()
    for done, (run, shard, chunk, arrays) in enumerate(
        pool.imap_unordered(embed_chunk, tasks), 1
    ):
        parts.setdefault((run, shard), {})[chunk] = arrays
        if done % 10 == 0 or done == len(tasks):
            elapsed = time.perf_counter() - started
            print(
                f"  [{done}/{len(tasks)}] chunks embedded  {elapsed:6.1f}s", flush=True
            )
    Path(EMB_DIR).mkdir(parents=True, exist_ok=True)
    for (run, shard), chunks in parts.items():
        for d, path in enumerate(embedding_paths(run, shard).values()):
            arrays = [chunks[c][d] for c in sorted(chunks) if chunks[c][d].ndim == 2]
            np.save(path, np.concatenate(arrays))


def load_embedding(run: str, shard: str | None, name: str) -> np.ndarray:
    return np.load(embedding_paths(run, shard)[name])


def subsample(x: np.ndarray, rng, n: int) -> np.ndarray:
    if x.shape[0] <= n:
        return x
    return x[rng.choice(x.shape[0], n, replace=False)]


def list_runs() -> list[str]:
    if RUNS:
        return list(RUNS)
    root = Path(GEN_DIR)
    return sorted(
        p.name
        for p in root.iterdir()
        if p.is_dir()
        and not p.name.startswith("_")
        and all((p / f"{shard}.npz").exists() for shard in SHARDS)
    )


def distances(runs, n_cases, rng) -> tuple[dict, dict, dict]:
    """Fréchet and MMD² per ``run|shard`` and descriptor; also the scales and bandwidths."""
    gt_key = str(n_cases)
    gt = {name: load_embedding(gt_key, None, name) for name in descriptor_names()}
    scale = {name: gt[name].std(0) + 1e-9 for name in gt}
    bandwidth = {name: median_bandwidth(gt[name] / scale[name]) for name in gt}
    results = {}
    for run in runs:
        for shard in SHARDS:
            row = {}
            for name in descriptor_names():
                x = load_embedding(run, shard, name)
                total, mean_term, cov_term = frechet(x, gt[name])
                x_std = subsample(x / scale[name], rng, MMD_SUBSAMPLE)
                gt_std = subsample(gt[name] / scale[name], rng, MMD_SUBSAMPLE)
                row[name] = {
                    "frechet": total,
                    "frechet_mean": mean_term,
                    "frechet_cov": cov_term,
                    "mmd2": mmd2(x_std, gt_std, bandwidth[name]),
                }
            results[f"{run}|{shard}"] = row
            print(
                f"  {run}|{shard:<20} FLD8b={row['density_g8_bbox']['frechet']:.4f} "
                f"FPD={row['position']['frechet']:.4f} "
                f"MMD8b={row['density_g8_bbox']['mmd2']:.5f}",
                flush=True,
            )
    return results, scale, bandwidth


def control_tests(runs, n_cases, scale, bandwidth, rng) -> dict:
    """Permutation test of ``MMD²(run) < MMD²(CONTROL)`` on ``TEST_DESC`` per run and shard."""
    if CONTROL is None or CONTROL not in runs:
        return {}
    gt_std = load_embedding(str(n_cases), None, TEST_DESC) / scale[TEST_DESC]
    tests = {}
    for shard in SHARDS:
        ref = subsample(gt_std, rng, TEST_SUBSAMPLE)
        control = load_embedding(CONTROL, shard, TEST_DESC) / scale[TEST_DESC]
        control = subsample(control, rng, TEST_SUBSAMPLE)
        for run in runs:
            if run == CONTROL:
                continue
            arm = load_embedding(run, shard, TEST_DESC) / scale[TEST_DESC]
            arm = subsample(arm, rng, TEST_SUBSAMPLE)
            mmd_run, mmd_control, p = mmd_permutation_test(
                arm, control, ref, bandwidth[TEST_DESC], n_perm=N_PERM
            )
            tests[f"{run}|{shard}"] = {
                "mmd2_run": mmd_run,
                "mmd2_control": mmd_control,
                "p_closer": p,
            }
            print(
                f"  test {run}|{shard}: p(closer than {CONTROL}) = {p:.3f}", flush=True
            )
    return tests


def main():
    ids = dev_ids()
    runs = list_runs()
    first = Path(GEN_DIR) / runs[0] / f"{SHARDS[0]}.npz"
    n_cases = int(np.load(first)["bcount"].shape[0])
    print(f"runs: {runs}\nshards: {SHARDS}\ncases: {n_cases}", flush=True)
    jobs = [(str(n_cases), None)] + [(run, shard) for run in runs for shard in SHARDS]
    with mp.get_context("forkserver").Pool(
        WORKERS, initializer=init_worker, initargs=(ids, GEN_DIR, CHUNK, TRAIN_LANCE)
    ) as pool:
        ensure_embeddings(pool, jobs, n_cases)

    rng = np.random.default_rng(0)
    results, scale, bandwidth = distances(runs, n_cases, rng)
    tests = control_tests(runs, n_cases, scale, bandwidth, rng)
    meta = {
        "control": CONTROL,
        "grids": list(GRIDS),
        "frames": list(FRAMES),
        "test_desc": TEST_DESC,
        "mmd_subsample": MMD_SUBSAMPLE,
        "test_subsample": TEST_SUBSAMPLE,
        "n_perm": N_PERM,
        "bandwidth": {name: float(bw) for name, bw in bandwidth.items()},
        "n_gt": n_cases,
    }
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    report = {"meta": meta, "results": results, "tests": tests}
    Path(OUT).write_text(json.dumps(report, indent=1))
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
