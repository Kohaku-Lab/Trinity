"""The NFE x refiner-steps grid: refine a run's cached draws and score every cell.

Reads the shards ``GEN_DIR/<RUN>/<SHARD_PREFIX>_nfe<k>.npz`` (or one fixed ``SHARD``)
and refines their latents on the GPU with ``REFINER``: the closed-form refiner
(``closed``, or ``closed_positions`` for positions only) at ``WEIGHTS``, or a ported
published refiner. Every step count of ``STEPS`` is one cell (0 = the raw cache): under
a constant lr one descent yields every cell as a snapshot, under a decaying schedule
each cell is its own descent.

Every cell is scored with the batched metric vector (mean over draws, then over cases);
the cells in ``CPU_CHECK_STEPS`` are also scored with the numpy metric vector on a CPU
pool as a cross-check. Writes ``OUT`` (JSON ``{nfe: {steps: {"soft": [...], "cpu":
[...]}}}``) and an ``.npz`` of the per-case vectors.

Run::

    kogine run scripts/eval/refine.py --config configs/eval/flagship/refine.py
"""

import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import torch

from trinity.data.splits import dev_split_cases
from trinity.decode import z_to_xywh
from trinity.floorplan.data import LanceFloorplanStore, find_train_lance
from trinity.floorplan.parameterize import z_to_xywh as z_to_xywh_np
from trinity.floorplan.scoring.vector import COLS, metric_vector
from trinity.sampling.refine_case import build_refine_case
from trinity.sampling.refine_closed import refine_closed, total_steps
from trinity.sampling.score_batched import metric_vector_batched
from trinity_baselines.ports.refine import PORTS

torch.set_float32_matmul_precision("high")

TRAIN_LANCE: str | None = None
DEV_PER_N_K: int = 100
DEV_RANDOM_SIZE: int = 2000
SPLIT_SEED: int = 20090220
DEV_LIMIT: int = 0  # >0 keeps only the first DEV_LIMIT dev cases

# Input cache.
RUN: str = "flagship"
GEN_DIR: str = "outputs/gen"
SHARD_PREFIX: str = "projected"  # shards are <SHARD_PREFIX>_nfe<k>.npz
SHARD: str | None = None  # one fixed shard for every NFE (e.g. the prior cache)
NFES: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
DRAWS: int = 0  # draws per case refined; 0 = every cached draw

# Refiner: closed | closed_positions | a key of trinity_baselines.ports.refine.PORTS.
REFINER: str = "closed"
PORT_KWARGS: dict = {}
STEPS: tuple[int, ...] = (0, 10, 25, 50, 100, 200, 400)
WEIGHTS: dict[str, float] = {
    "overlap": 10.0,
    "group": 1.0,
    "mib": 1.0,
    "boundary": 1.0,
    "wl": 1.0,
    "area": 1.0,
}
LR: float = 0.001
LR_SCHEDULE: str | dict = "linear"  # constant | cosine | linear | an AnySchedule dict
OPTIMIZER: str = "adam"  # adam | sgd
BETAS: tuple[float, float] = (0.0, 0.99)
EPS: float = 1e-8
CHUNK: int = 5  # refiner steps per compiled call
COMPILE: bool = True

DEVICE: str = "cuda"
MAX_ROWS: int = 2000
CPU_CHECK_STEPS: tuple[int, ...] = ()  # cells also scored on the CPU pool
WORKERS: int = 16
OUT: str = "outputs/eval/refine/flagship.json"

# Per-worker state, set by ``init_worker``.
_worker: dict = {}


def init_worker(ids, lance) -> None:
    _worker.update(
        ids=ids, store=LanceFloorplanStore(str(find_train_lance(lance))), cases={}
    )


def worker_case(i: int):
    if i not in _worker["cases"]:
        _worker["cases"][i] = _worker["store"].instances([_worker["ids"][i]])[0]
    return _worker["cases"][i]


def cpu_score(job):
    """The numpy metric vector of case ``i``, mean over its draws."""
    i, z = job
    inst = worker_case(i)
    rows = [
        metric_vector(z_to_xywh_np(zj, inst.area_targets, inst.s), inst) for zj in z
    ]
    return i, np.mean(rows, axis=0)


def sync() -> None:
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def refine_snapshots(z0: torch.Tensor, case) -> dict[int, torch.Tensor]:
    """``{steps: z}`` for every step count of ``STEPS``; 0 is the raw input."""
    if REFINER in PORTS:
        out = {s: PORTS[REFINER](z0, case, s, **PORT_KWARGS) for s in STEPS if s > 0}
    else:
        options = {
            "betas": BETAS,
            "eps": EPS,
            "chunk": CHUNK,
            "compile_loop": COMPILE,
            "shape": REFINER != "closed_positions",
            "optimizer": OPTIMIZER,
        }
        if LR_SCHEDULE == "constant":
            snapshots = tuple(STEPS)
            out = refine_closed(
                z0,
                case,
                WEIGHTS,
                max(STEPS),
                LR,
                "constant",
                snapshots=snapshots,
                **options,
            )
        else:
            out = {
                s: refine_closed(z0, case, WEIGHTS, s, LR, LR_SCHEDULE, **options)[s]
                for s in STEPS
                if s > 0
            }
    out[0] = z0
    return out


def updates_per_row() -> int:
    """Refiner updates each row receives over one pass of ``STEPS``."""
    if REFINER in PORTS:
        return sum(s for s in STEPS if s > 0)
    return total_steps(LR_SCHEDULE, tuple(STEPS))


def shard_path(nfe: int) -> Path:
    name = SHARD or f"{SHARD_PREFIX}_nfe{nfe}"
    return Path(GEN_DIR) / RUN / f"{name}.npz"


def stack_draws(lat, offsets, bcount, k_all, k, indices, n_max) -> torch.Tensor:
    """The first ``k`` cached draws of every
    case in ``indices``, padded to ``n_max``."""
    z0 = torch.zeros(len(indices) * k, n_max, 3, device=DEVICE)
    for gi, ci in enumerate(indices):
        n = int(bcount[ci])
        draws = lat[offsets[ci] : offsets[ci + 1]].reshape(k_all, n, 3)[:k]
        z0[gi * k : (gi + 1) * k, :n] = torch.as_tensor(draws, device=DEVICE)
    return z0


def format_cell(nfe: int, steps: int, v) -> str:
    col = COLS.index
    return (
        f"  nfe={nfe:2d} steps={steps:4d} overlap={v[col('overlap_ratio')]:.5f}"
        f" group={v[col('group_gap')]:.5f} boundary={v[col('boundary_dist')]:.5f}"
        f" hpwl_gap={v[col('hpwl_gap')]:.4f} area_gap={v[col('area_gap')]:.4f}"
        f" soft_cost={v[col('soft_cost')]:.4f}"
    )


def sweep_nfe(nfe, cases, order, pool, timing):
    """Refine and score every cell of one NFE;
    return ``(cells, per-case GPU vectors)``."""
    data = np.load(shard_path(nfe))
    lat, bcount, k_all = data["lat"], data["bcount"], int(data["k"])
    k = k_all if DRAWS <= 0 else min(DRAWS, k_all)
    offsets = np.concatenate([[0], np.cumsum(bcount.astype(np.int64) * k_all)])
    cases_per_chunk = max(1, MAX_ROWS // k)
    gpu = {s: np.zeros((len(cases), len(COLS)), dtype=np.float32) for s in STEPS}
    cpu = {s: np.zeros((len(cases), len(COLS))) for s in STEPS if s in CPU_CHECK_STEPS}
    for start in range(0, len(order), cases_per_chunk):
        indices = order[start : start + cases_per_chunk]
        case = build_refine_case([cases[i] for i in indices], k, DEVICE)
        n_max = case.token_mask.shape[1]
        z0 = stack_draws(lat, offsets, bcount, k_all, k, indices, n_max)
        sync()
        t0 = time.perf_counter()
        snapshots = refine_snapshots(z0, case)
        sync()
        timing["refine_s"] += time.perf_counter() - t0
        timing["step_rows"] += updates_per_row() * z0.shape[0]
        ones = torch.ones(z0.shape[0], device=DEVICE)
        for s in STEPS:
            boxes = z_to_xywh(snapshots[s], case.area_norm, ones)
            vectors = metric_vector_batched(boxes, case).view(
                len(indices), k, len(COLS)
            )
            gpu[s][indices] = vectors.mean(1).cpu().numpy()
        for s in cpu:
            z = snapshots[s].cpu().numpy().astype(np.float64)
            jobs = [
                (ci, z[gi * k : (gi + 1) * k, : bcount[ci]])
                for gi, ci in enumerate(indices)
            ]
            for ci, row in pool.imap_unordered(cpu_score, jobs, chunksize=8):
                cpu[s][ci] = row
        done = min(start + cases_per_chunk, len(order))
        print(f"  [{RUN}] nfe={nfe} {done}/{len(order)} cases", flush=True)
    cells = {}
    for s in STEPS:
        cells[s] = {"soft": gpu[s].mean(0).tolist()}
        if s in cpu:
            cells[s]["cpu"] = cpu[s].mean(0).tolist()
            cells[s]["max_abs_gpu_cpu"] = float(np.abs(gpu[s] - cpu[s]).max())
        print(format_cell(nfe, s, cells[s]["soft"]), flush=True)
    return cells, gpu


def write_outputs(ids, results, per_case, timing, seconds, n_cases) -> None:
    out = Path(OUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        f"nfe{nfe}_steps{s}": per_case[nfe][s]
        for nfe in per_case
        for s in per_case[nfe]
    }
    np.savez_compressed(
        out.with_suffix(".npz"), ids=np.asarray(ids), cols=np.asarray(COLS), **arrays
    )
    meta = {
        "run": RUN,
        "refiner": REFINER,
        "port_kwargs": dict(PORT_KWARGS),
        "nfes": list(NFES),
        "steps": list(STEPS),
        "draws": DRAWS,
        "weights": WEIGHTS,
        "lr": LR,
        "lr_schedule": LR_SCHEDULE,
        "optimizer": OPTIMIZER,
        "betas": list(BETAS),
        "eps": EPS,
        "cols": list(COLS),
        "n_cases": n_cases,
        "seconds": seconds,
        "timing": timing,
        "us_per_step_row": 1e6 * timing["refine_s"] / max(timing["step_rows"], 1.0),
    }
    results = {
        str(nfe): {str(s): c for s, c in d.items()} for nfe, d in results.items()
    }
    out.write_text(json.dumps({"meta": meta, "results": results}, indent=1))
    print(f"wrote {OUT} in {seconds:.0f}s", flush=True)


def main():
    cases, ids, _ = dev_split_cases(
        TRAIN_LANCE, DEV_PER_N_K, DEV_RANDOM_SIZE, SPLIT_SEED, DEV_LIMIT
    )
    order = sorted(range(len(cases)), key=lambda i: cases[i].block_count)
    timing = {"refine_s": 0.0, "step_rows": 0.0}
    results, per_case = {}, {}
    started = time.perf_counter()
    pool = None
    if CPU_CHECK_STEPS:
        pool = mp.get_context("forkserver").Pool(
            WORKERS, initializer=init_worker, initargs=(ids, TRAIN_LANCE)
        )
    try:
        for nfe in NFES:
            results[nfe], per_case[nfe] = sweep_nfe(nfe, cases, order, pool, timing)
    finally:
        if pool is not None:
            pool.close()
    seconds = time.perf_counter() - started
    write_outputs(ids, results, per_case, timing, seconds, len(cases))


if __name__ == "__main__":
    main()
