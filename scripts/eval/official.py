"""The full pipeline on the official FloorSet 100 validation cases, run and timed per case.

For every setting ``(NFE, refiner steps, draws)`` of ``SETTINGS`` and every case on its own:
sample the draws (one forward of that case alone), refine them with the closed-form refiner,
legalize every draw in parallel on a pool of ``WORKERS`` processes (one job per draw), score
with the hard cost and keep the cheapest draw (feasible first). Recorded per case: the
wall time of every stage (sampling, refinement, decoding, legalization = the slowest draw),
the legalizer's milliseconds per draw and the LP solver's seconds per draw. The
``torch.compile`` and pool warm-up runs once first and is reported separately.

``SAVE_LAYOUTS`` stores the kept legalized layout of every case and setting in one ``.npz``
(the start layouts of the pipeline -> classical-solver hand-over).

Run::

    kogine run scripts/eval/official.py --config configs/eval/flagship/official.py
"""

import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import torch
from ortools.linear_solver import pywraplp

from trinity.floorplan.data import load_validation_set
from trinity.floorplan.legalize import apply_route, scale_pack
from trinity.floorplan.parameterize import z_to_xywh as z_to_xywh_np
from trinity.floorplan.scoring import total_exp, validate
from trinity.floorplan.scoring.cost import use_fast_hpwl
from trinity.floorplan.types import Placement
from trinity.hub import load_model
from trinity.sampling.refine_case import build_refine_case
from trinity.sampling.refine_closed import refine_closed

torch.set_float32_matmul_precision("high")

# The model: a .ckpt, a release directory or a Hugging Face repo id (+ release subfolder).
CHECKPOINT: str = "outputs/train/flagship/checkpoints/last.ckpt"
SCALE: str | None = None
FLOORSET_ROOT: str | None = None
ALLOW_DOWNLOAD: bool = True

# Settings: (NFE, refiner steps, draws) per pass.
SETTINGS: tuple[tuple[int, int, int], ...] = ((8, 400, 16),)
SAMPLE_SOLVER: str = "euler"
SAMPLE_PROJECTIONS: list | None = None  # None = the sampler default, [] = free

# The closed-form refiner.
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
BETAS: tuple[float, float] = (0.0, 0.99)
CHUNK: int = 5

LEGALIZE_ROUTE: list | None = None  # None = scale_pack with its defaults
SCORER: str = "full_fast"
WORKERS: int = 16
SEED: int = 0
DEVICE: str = "cuda"
SAVE_LAYOUTS: str | None = None  # .npz path of the kept layout per case, or None
OUT: str = "outputs/eval/official/flagship.json"

FIELDS = (
    "block_count",
    "cost",
    "hpwl_gap",
    "area_gap",
    "v_rel",
    "feasible",
    "v_grouping",
    "v_mib",
    "v_boundary",
    "n_soft",
)
TIMES = (
    "sample_s",
    "refine_s",
    "decode_s",
    "legalize_s",
    "case_s",
    "legalize_ms_per_draw",
    "solve_s_per_draw",
)

# Per-worker state, set by ``init_worker``.
_worker: dict = {"solve_s": 0.0}


def init_worker(cases, route, scorer) -> None:
    """Keep the cases and route; wrap the LP ``Solve`` so its seconds are accumulated."""
    _worker.update(cases=cases, route=route, scorer=scorer)
    original_solve = pywraplp.Solver.Solve

    def timed_solve(self, *args, **kwargs):
        started = time.perf_counter()
        status = original_solve(self, *args, **kwargs)
        _worker["solve_s"] += time.perf_counter() - started
        return status

    pywraplp.Solver.Solve = timed_solve


def legalize_job(job):
    """``(case, draw, boxes)`` -> ``(case, draw, score row, ms, LP seconds, info, boxes)``."""
    i, j, boxes = job
    inst = _worker["cases"][i]
    _worker["solve_s"] = 0.0
    started = time.perf_counter()
    with use_fast_hpwl(True):
        legal = apply_route(Placement(boxes.astype(np.float64), inst), _worker["route"])
        score = validate(legal, _worker["scorer"]).score
    ms = 1000 * (time.perf_counter() - started)
    soft = score.soft
    row = [
        score.block_count,
        score.cost,
        score.hpwl_gap,
        score.area_gap,
        score.v_rel,
        float(score.feasible),
        soft.grouping if soft else 0,
        soft.mib if soft else 0,
        soft.boundary if soft else 0,
        soft.n_soft if soft else 0,
    ]
    info = scale_pack.last_info()
    return i, j, row, ms, _worker["solve_s"], info, legal.xywh.astype(np.float32)


def sync() -> None:
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def sample(placer, sampler, group, draws: int, seed: int) -> torch.Tensor:
    """One sampler forward over ``len(group) x draws`` rows from a seeded noise."""
    placer.samples = draws
    placer.sampler = sampler
    ctx = placer.build_group_ctx(group)
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    noise = torch.randn(
        ctx["rows"], ctx["max_n"], 3, device=DEVICE, generator=generator
    )
    return placer.sample_latents(ctx, noise=noise)


def refine(z, group, draws: int, steps: int) -> torch.Tensor:
    case = build_refine_case(group, draws, DEVICE)
    out = refine_closed(
        z,
        case,
        WEIGHTS,
        steps,
        LR,
        LR_SCHEDULE,
        betas=BETAS,
        chunk=CHUNK,
        compile_loop=True,
    )
    return out[steps]


def decode_jobs(i: int, inst, z: torch.Tensor) -> list:
    z = z.cpu().numpy().astype(np.float64)
    n = inst.block_count
    return [
        (i, j, z_to_xywh_np(z[j, :n], inst.area_targets, inst.s)) for j in range(len(z))
    ]


def warm_up(placer, samplers, pool, cases) -> float:
    """Compile the refiner for every setting and start the pool; return the seconds."""
    started = time.perf_counter()
    warm = [cases[0]]
    for nfe, steps, draws in SETTINGS:
        z = sample(placer, samplers[nfe], warm, draws, SEED)
        refine(z, warm, draws, steps)
    sync()
    list(pool.imap_unordered(legalize_job, decode_jobs(0, cases[0], z)))
    return time.perf_counter() - started


def run_setting(placer, sampler, pool, cases, steps, draws, layouts, key):
    """Every case through one setting; return per-case rows, times, draw costs and info."""
    rows, times, candidates, info = [], [], [], []
    for i, inst in enumerate(cases):
        t_case = time.perf_counter()
        z = sample(placer, sampler, [inst], draws, SEED + 1 + i)
        sync()
        t_sampled = time.perf_counter()
        z = refine(z, [inst], draws, steps)
        sync()
        t_refined = time.perf_counter()
        jobs = decode_jobs(i, inst, z)
        t_decoded = time.perf_counter()
        results = list(pool.imap_unordered(legalize_job, jobs, chunksize=1))
        t_legalized = time.perf_counter()
        results.sort(key=lambda r: r[1])
        best = min(results, key=lambda r: (r[2][5] < 0.5, r[2][1]))
        if SAVE_LAYOUTS:
            layouts[f"{key}_case{i}"] = best[6]
        rows.append(best[2])
        candidates.append([r[2][1] for r in results])
        info.append([{name: float(v) for name, v in r[5].items()} for r in results])
        times.append(
            [
                t_sampled - t_case,
                t_refined - t_sampled,
                t_decoded - t_refined,
                t_legalized - t_decoded,
                t_legalized - t_case,
                float(np.mean([r[3] for r in results])),
                float(np.mean([r[4] for r in results])),
            ]
        )
    return np.array(rows), np.array(times), candidates, info


def report(rows: np.ndarray, times: np.ndarray, draws: int) -> dict:
    """The official averages of one setting, printed and returned."""
    columns = {name: rows[:, i] for i, name in enumerate(FIELDS)}
    cost, feasible = columns["cost"], columns["feasible"]
    block_counts = columns["block_count"].astype(int).tolist()
    out = {
        "total_exp": float(total_exp(cost.tolist(), block_counts)),
        "cost_mean": float(cost.mean()),
        "cost_median": float(np.median(cost)),
        "hpwl_gap": float(columns["hpwl_gap"].mean()),
        "area_gap": float(columns["area_gap"].mean()),
        "v_rel": float(columns["v_rel"].mean()),
        "feasible": float(feasible.mean()),
    }
    for name in ("grouping", "mib", "boundary"):
        out[f"v_{name}_total"] = int(columns[f"v_{name}"].sum())
    for i, name in enumerate(TIMES):
        values = times[:, i]
        out[name] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "max": float(values.max()),
        }
    print(
        f"  cost (exp(n/12)-weighted total) {out['total_exp']:.4f}"
        f"  mean {out['cost_mean']:.4f}  median {out['cost_median']:.4f}\n"
        f"  hpwl_gap {out['hpwl_gap']:+.4f}  area_gap {out['area_gap']:+.4f}"
        f"  v_rel {out['v_rel']:.4f}  feasible {100 * out['feasible']:.1f}%\n"
        f"  per case {out['case_s']['mean']:.3f}s (sampling {out['sample_s']['mean']:.3f},"
        f" refine {out['refine_s']['mean']:.3f}, legalize {out['legalize_s']['mean']:.3f}"
        f" over {draws} draws)",
        flush=True,
    )
    return out


def main():
    started = time.perf_counter()
    cases = load_validation_set(root=FLOORSET_ROOT, allow_download=ALLOW_DOWNLOAD)
    model = load_model(CHECKPOINT, scale=SCALE, device=DEVICE)
    samplers = {
        nfe: model.sampler(nfe, SAMPLE_SOLVER, SAMPLE_PROJECTIONS)
        for nfe, _, _ in SETTINGS
    }
    pool = mp.get_context("forkserver").Pool(
        WORKERS, initializer=init_worker, initargs=(cases, LEGALIZE_ROUTE, SCORER)
    )
    results, layouts = {}, {}
    print(
        f"{len(cases)} cases; settings {list(SETTINGS)}; {WORKERS} workers", flush=True
    )
    with torch.no_grad():
        placer = model.placer(refiner=None)
        warmup_s = warm_up(placer, samplers, pool, cases)
        print(f"warm-up: {warmup_s:.1f}s", flush=True)
        for nfe, steps, draws in SETTINGS:
            key = f"nfe{nfe}_steps{steps}_n{draws}"
            rows, times, candidates, info = run_setting(
                placer, samplers[nfe], pool, cases, steps, draws, layouts, key
            )
            print(f"\n=== NFE {nfe}, {steps} refiner steps, best of {draws} ===")
            results[key] = {
                "summary": report(rows, times, draws),
                "rows": rows.tolist(),
                "times": times.tolist(),
                "candidates": candidates,
                "legalizer": info,
            }
    pool.close()
    pool.join()
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    if SAVE_LAYOUTS:
        ids = [c.test_id if c.test_id is not None else -1 for c in cases]
        bcount = [c.block_count for c in cases]
        np.savez_compressed(
            SAVE_LAYOUTS, ids=np.array(ids), bcount=np.array(bcount), **layouts
        )
    config = {
        "checkpoint": CHECKPOINT,
        "scale": SCALE,
        "settings": [list(s) for s in SETTINGS],
        "sample_projections": SAMPLE_PROJECTIONS,
        "weights": WEIGHTS,
        "lr": LR,
        "schedule": LR_SCHEDULE,
        "betas": list(BETAS),
        "route": LEGALIZE_ROUTE,
        "workers": WORKERS,
        "seed": SEED,
    }
    report_json = {
        "config": config,
        "warmup_s": warmup_s,
        "fields": list(FIELDS),
        "times": list(TIMES),
        "results": results,
        "total_s": time.perf_counter() - started,
    }
    Path(OUT).write_text(json.dumps(report_json, indent=1))
    print(f"\nwrote {OUT} ({time.perf_counter() - started:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
