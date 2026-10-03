"""Refine a run's cached draws, legalize every draw, and report the hard cost.

Per ``(NFE, steps)`` cell: the refiner (as in ``refine.py``) runs on the GPU over every
cached draw; each refined draw goes through ``LEGALIZE_ROUTE`` (None = ``scale_pack``
with its defaults) and is scored with ``SCORER`` on a CPU pool, timed per draw. Per case
the cell records, for every ``N`` in ``N_SELECT``:

* ``n1`` -- the legalized cost of draw 0,
* ``best<N>`` -- the cheapest legalized cost over the first ``N`` draws,
* ``soft<N>`` -- the legalized cost of the draw
  with the lowest pre-legalization ``soft_cost``.

Each reports the mean cost, feasible rate and the block-count-weighted hard-cost total.
Writes ``OUT`` (JSON) and an ``.npz`` with the per-draw hard columns and soft vectors
(and, with ``SAVE_LAYOUTS``, the refined latents and legalized boxes).

Run::

    kogine run scripts/eval/legalize.py --config configs/eval/flagship/legalize.py
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
from trinity.floorplan.legalize import apply_route, scale_pack
from trinity.floorplan.parameterize import z_to_xywh as z_to_xywh_np
from trinity.floorplan.scoring import validate
from trinity.floorplan.scoring.cost import M_PENALTY, use_fast_hpwl
from trinity.floorplan.scoring.total import total_exp
from trinity.floorplan.scoring.vector import COLS
from trinity.floorplan.types import Placement
from trinity.sampling.refine_case import build_refine_case
from trinity.sampling.refine_closed import refine_closed
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
NFES: tuple[int, ...] = (1, 8, 32)
DRAWS: int = 0  # draws per case; 0 = every cached draw
N_SELECT: tuple[int, ...] = (1, 4)  # best-of-N sizes reported

# Refiner: closed | closed_positions | a key of trinity_baselines.ports.refine.PORTS.
REFINER: str = "closed"
PORT_KWARGS: dict = {}
STEPS: tuple[int, ...] = (0, 400)
WEIGHTS: dict[str, float] = {
    "overlap": 10.0,
    "group": 1.0,
    "mib": 1.0,
    "boundary": 1.0,
    "wl": 1.0,
    "area": 1.0,
}
LR: float = 0.001
LR_SCHEDULE: str | dict = "linear"
OPTIMIZER: str = "adam"
BETAS: tuple[float, float] = (0.0, 0.99)
EPS: float = 1e-8
CHUNK: int = 5
COMPILE: bool = True

# Legalization and scoring.
LEGALIZE_ROUTE: list | None = None  # None = scale_pack with its defaults
SCORER: str = "full_fast"
SAVE_LAYOUTS: bool = False

DEVICE: str = "cuda"
MAX_ROWS: int = 2000
WORKERS: int = 24
OUT: str = "outputs/eval/legalize/flagship.json"

HARD = ("feasible", "cost", "hpwl_gap", "area_gap", "v_rel", "ms", "lam", "rung")

# Per-worker state, set by ``init_worker``.
_worker: dict = {}


def route_uses_scale_pack(route) -> bool:
    if route is None:
        return True
    return any(
        (s if isinstance(s, str) else s.get("name")) == "scale_pack" for s in route
    )


def init_worker(ids, route, scorer, lance) -> None:
    _worker.update(
        ids=ids,
        route=route,
        scorer=scorer,
        scale_pack_info=route_uses_scale_pack(route),
        store=LanceFloorplanStore(str(find_train_lance(lance))),
        cases={},
    )


def worker_case(i: int):
    if i not in _worker["cases"]:
        _worker["cases"][i] = _worker["store"].instances([_worker["ids"][i]])[0]
    return _worker["cases"][i]


def legalize_job(job):
    """Legalize and score every draw of case ``i``: ``(i, hard (draws, 8), boxes)``."""
    i, z = job
    inst = worker_case(i)
    hard = np.zeros((z.shape[0], len(HARD)))
    boxes = np.zeros((z.shape[0], inst.block_count, 4), dtype=np.float32)
    for j, zj in enumerate(z):
        xywh = z_to_xywh_np(zj, inst.area_targets, inst.s).astype(np.float64)
        started = time.perf_counter()
        with use_fast_hpwl(True):
            legal = apply_route(Placement(xywh=xywh, instance=inst), _worker["route"])
            score = validate(legal, _worker["scorer"]).score
        ms = 1000 * (time.perf_counter() - started)
        info = (
            scale_pack.last_info()
            if _worker["scale_pack_info"]
            else {"lam": 0, "rung": 0}
        )
        hard[j] = [
            float(score.feasible),
            score.cost,
            score.hpwl_gap,
            score.area_gap,
            score.v_rel,
            ms,
            info["lam"],
            info["rung"],
        ]
        boxes[j] = legal.xywh
    return i, hard, boxes


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


def summarize(cost, feasible, bcount) -> dict:
    """Mean cost, feasible rate and the block-count-weighted
    hard-cost total of per-case costs."""
    feasible_mask = feasible > 0.5
    return {
        "cost": float(cost.mean()),
        "feasible": float(feasible.mean()),
        "total_exp": float(total_exp(cost.tolist(), bcount.tolist())),
        "cost_median": float(np.median(cost)),
        "cost_p90": float(np.quantile(cost, 0.9)),
        "cost_feasible": (
            float(cost[feasible_mask].mean()) if feasible.any() else M_PENALTY
        ),
    }


def cell_summary(hard, soft, k, refine_s, bcount, arrays, key) -> dict:
    """The reported numbers of one ``(NFE, steps)``
    cell; per-case arrays go to ``arrays``."""
    col = HARD.index
    cell = {
        "soft": soft.mean((0, 1)).tolist(),
        "ms_per_draw": float(hard[:, :, col("ms")].mean()),
        "refine_s": refine_s,
        "draws": k,
        "feasible_all_draws": float(hard[:, :, col("feasible")].mean()),
        "lam": {
            name: float(np.quantile(hard[:, :, col("lam")], q))
            for name, q in (("p50", 0.5), ("p90", 0.9), ("max", 1.0))
        },
        "rung": np.bincount(
            hard[:, :, col("rung")].astype(np.int64).ravel(), minlength=9
        ).tolist(),
    }
    soft_cost = soft[:, :, COLS.index("soft_cost")]
    rows = np.arange(hard.shape[0])
    for n_select in N_SELECT:
        n = min(n_select, k)
        cost = hard[:, :n, col("cost")]
        feasible = hard[:, :n, col("feasible")]
        best = cost.argmin(1)
        label = "n1" if n == 1 else f"best{n}"
        cell[label] = summarize(cost[rows, best], feasible[rows, best], bcount)
        if n > 1:
            pick = soft_cost[:, :n].argmin(1)
            cell[f"soft{n}"] = summarize(cost[rows, pick], feasible[rows, pick], bcount)
        arrays[f"{key}_best{n}_cost"] = cost[rows, best].astype(np.float32)
    arrays[f"{key}_hard"] = hard.astype(np.float32)
    arrays[f"{key}_soft"] = soft
    return cell


def print_cell(nfe: int, steps: int, cell: dict) -> None:
    selections = " ".join(
        f"{name}: cost={v['cost']:.4f} feas={v['feasible']:.3f} "
        f"total={v['total_exp']:.4f}"
        for name, v in cell.items()
        if isinstance(v, dict) and "cost" in v
    )
    soft_cost = cell["soft"][COLS.index("soft_cost")]
    print(
        f"  nfe={nfe:2d} steps={steps:4d} soft_cost={soft_cost:.4f} {selections}"
        f" | {cell['ms_per_draw']:.0f} ms per draw",
        flush=True,
    )


def sweep_nfe(nfe, cases, order, bcount_all, pool, arrays) -> dict:
    """Refine, legalize and score every cell of one NFE."""
    data = np.load(shard_path(nfe))
    lat, bcount, k_all = data["lat"], data["bcount"], int(data["k"])
    k = k_all if DRAWS <= 0 else min(DRAWS, k_all)
    offsets = np.concatenate([[0], np.cumsum(bcount.astype(np.int64) * k_all)])
    cases_per_chunk = max(1, MAX_ROWS // k)
    hard = {s: np.zeros((len(cases), k, len(HARD))) for s in STEPS}
    soft = {s: np.zeros((len(cases), k, len(COLS)), dtype=np.float32) for s in STEPS}
    latents = {s: [None] * len(cases) for s in STEPS}
    boxes = {s: [None] * len(cases) for s in STEPS}
    refine_s = 0.0
    for start in range(0, len(order), cases_per_chunk):
        indices = order[start : start + cases_per_chunk]
        case = build_refine_case([cases[i] for i in indices], k, DEVICE)
        z0 = stack_draws(
            lat, offsets, bcount, k_all, k, indices, case.token_mask.shape[1]
        )
        sync()
        t0 = time.perf_counter()
        snapshots = refine_snapshots(z0, case)
        sync()
        refine_s += time.perf_counter() - t0
        ones = torch.ones(z0.shape[0], device=DEVICE)
        for s in STEPS:
            g = z_to_xywh(snapshots[s], case.area_norm, ones)
            vectors = metric_vector_batched(g, case).view(len(indices), k, len(COLS))
            soft[s][indices] = vectors.cpu().numpy()
            z = snapshots[s].cpu().numpy().astype(np.float64)
            jobs = [
                (ci, z[gi * k : (gi + 1) * k, : bcount[ci]])
                for gi, ci in enumerate(indices)
            ]
            for ci, row, legal in pool.imap_unordered(legalize_job, jobs, chunksize=4):
                hard[s][ci] = row
                boxes[s][ci] = legal.reshape(-1, 4)
            if SAVE_LAYOUTS:
                for gi, ci in enumerate(indices):
                    draws = z[gi * k : (gi + 1) * k, : bcount[ci]]
                    latents[s][ci] = draws.reshape(-1, 3).astype(np.float32)
        done = min(start + cases_per_chunk, len(order))
        print(f"  [{RUN}] nfe={nfe} {done}/{len(order)} cases", flush=True)
    cells = {}
    for s in STEPS:
        key = f"nfe{nfe}_steps{s}"
        cells[s] = cell_summary(hard[s], soft[s], k, refine_s, bcount_all, arrays, key)
        if SAVE_LAYOUTS:
            arrays[f"{key}_lat"] = np.concatenate(latents[s], axis=0)
            arrays[f"{key}_xywh"] = np.concatenate(boxes[s], axis=0)
        print_cell(nfe, s, cells[s])
    return cells, k


def main():
    cases, ids, _ = dev_split_cases(
        TRAIN_LANCE, DEV_PER_N_K, DEV_RANDOM_SIZE, SPLIT_SEED, DEV_LIMIT
    )
    order = sorted(range(len(cases)), key=lambda i: cases[i].block_count)
    bcount_all = np.array([case.block_count for case in cases])
    results, arrays = {}, {}
    started = time.perf_counter()
    init_args = (ids, LEGALIZE_ROUTE, SCORER, TRAIN_LANCE)
    with mp.get_context("forkserver").Pool(
        WORKERS, initializer=init_worker, initargs=init_args
    ) as pool:
        for nfe in NFES:
            results[nfe], k = sweep_nfe(nfe, cases, order, bcount_all, pool, arrays)
    seconds = time.perf_counter() - started
    out = Path(OUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "run": RUN,
        "refiner": REFINER,
        "port_kwargs": dict(PORT_KWARGS),
        "nfes": list(NFES),
        "steps": list(STEPS),
        "draws": DRAWS,
        "gen_dir": GEN_DIR,
        "shard": SHARD,
        "n_select": list(N_SELECT),
        "weights": dict(WEIGHTS),
        "lr": LR,
        "lr_schedule": LR_SCHEDULE,
        "optimizer": OPTIMIZER,
        "betas": list(BETAS),
        "legalize_route": LEGALIZE_ROUTE,
        "scorer": SCORER,
        "cols": list(COLS),
        "hard": list(HARD),
        "n_cases": len(cases),
        "seconds": seconds,
        "save_layouts": SAVE_LAYOUTS,
    }
    np.savez_compressed(
        out.with_suffix(".npz"),
        ids=np.asarray(ids),
        bcount=bcount_all,
        draws=np.int32(k),
        **arrays,
    )
    results = {str(n): {str(s): c for s, c in d.items()} for n, d in results.items()}
    out.write_text(json.dumps({"meta": meta, "results": results}, indent=1))
    print(f"wrote {OUT} in {seconds:.0f}s", flush=True)


if __name__ == "__main__":
    main()
