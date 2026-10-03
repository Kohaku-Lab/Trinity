"""The full pipeline on the MCNC / GSRC bookshelf cases under the fixed-outline protocol.

Cases: ``SUITE`` (``mcnc`` / ``gsrc``) in ``VARIANT`` (``HARD`` = fixed block shapes,
``SOFT`` = areas with the protocol's aspect bounds), with an outline of area
``(1 + GAMMA)`` x the block area at aspect ``ASPECT`` and the terminals at their file
positions (``MAP_PINS`` maps them onto the outline).

Per setting ``(NFE, refiner steps, draws)``, sampling seed of ``SEEDS`` and case: one sampler
forward of the draws, the closed-form refiner (the six terms plus ``outline``), every draw
legalized in parallel (the LP capped at the outline) and scored with the bookshelf metric
(net HPWL, dead space, fits the outline, overlap-free, shapes kept). The case's result is the
feasible draw with the lowest HPWL. Writes ``OUT`` (JSON: per-case rows and per-setting
summaries) and an ``.npz`` of the kept layouts.

Run::

    kogine run scripts/eval/bookshelf.py --config configs/eval/bookshelf/flagship_gsrc_soft.py
"""

import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import torch

from trinity.floorplan.data import (
    GSRC_SIZES,
    MCNC_NAMES,
    load_gsrc_set,
    load_mcnc_set,
    with_fixed_outline,
)
from trinity.floorplan.legalize import apply_route, scale_pack
from trinity.floorplan.parameterize import z_to_xywh as z_to_xywh_np
from trinity.floorplan.scoring.bookshelf import score_bookshelf
from trinity.floorplan.scoring.cost import use_fast_hpwl
from trinity.floorplan.types import Placement
from trinity.hub import load_model
from trinity.sampling.refine_case import build_refine_case
from trinity.sampling.refine_closed import refine_closed

torch.set_float32_matmul_precision("high")

# The model: a .ckpt, a release directory or a Hugging Face repo id (+ release subfolder).
CHECKPOINT: str = "outputs/train/flagship/checkpoints/last.ckpt"
SCALE: str | None = None

# Protocol.
SUITE: str = "mcnc"  # mcnc | gsrc
VARIANT: str = "HARD"  # HARD | SOFT
NAMES: tuple = ()  # () = every case of the suite (GSRC: block counts)
GAMMA: float = 0.10  # outline white space
ASPECT: float = 1.0  # outline aspect ratio
MAP_PINS: bool = False

# Settings: (NFE, refiner steps, draws) per pass, each repeated for every seed.
SETTINGS: tuple[tuple[int, int, int], ...] = ((8, 400, 16),)
SEEDS: tuple[int, ...] = (0,)
SAMPLE_SOLVER: str = "euler"
SAMPLE_PROJECTIONS: list | None = None  # None = the sampler default, [] = free

# The closed-form refiner, with the protocol's outline term.
WEIGHTS: dict[str, float] = {
    "overlap": 10.0,
    "group": 1.0,
    "mib": 1.0,
    "boundary": 1.0,
    "wl": 1.0,
    "area": 1.0,
    "outline": 1.0,
}
LR: float = 0.001
LR_SCHEDULE: str | dict = "linear"
BETAS: tuple[float, float] = (0.0, 0.99)
CHUNK: int = 5

LEGALIZE_ROUTE: list | None = None  # None = scale_pack with its defaults
WORKERS: int = 16
DEVICE: str = "cuda"
OUT: str = "outputs/eval/bookshelf/flagship_mcnc_hard.json"

# Per-worker state, set by ``init_worker``.
_worker: dict = {}


def init_worker(cases, route) -> None:
    _worker.update(cases=cases, route=route)


def legalize_job(job):
    """``(case, draw, boxes)`` -> ``(case, draw, score dict, legal boxes, ms, info)``."""
    i, j, boxes = job
    inst = _worker["cases"][i]
    started = time.perf_counter()
    with use_fast_hpwl(True):
        legal = apply_route(Placement(boxes.astype(np.float64), inst), _worker["route"])
    ms = 1000 * (time.perf_counter() - started)
    score = score_bookshelf(legal)
    info = scale_pack.last_info()
    return i, j, score.__dict__, legal.xywh.astype(np.float32), ms, info


def sync() -> None:
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def load_cases() -> tuple[list[str], list]:
    """The case names and the instances with their fixed outline."""
    if SUITE == "mcnc":
        names = list(NAMES) if NAMES else list(MCNC_NAMES)
        cases = load_mcnc_set(VARIANT, tuple(names))
    else:
        sizes = tuple(int(n) for n in NAMES) if NAMES else GSRC_SIZES
        names = [f"n{n}" for n in sizes]
        cases = load_gsrc_set(VARIANT, sizes)
    return names, [with_fixed_outline(c, GAMMA, ASPECT, MAP_PINS) for c in cases]


def is_feasible(score: dict) -> bool:
    fits = score["fits_outline"] is None or score["fits_outline"]
    return bool(score["overlap_free"] and score["shapes_kept"] and fits)


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
    if steps <= 0:
        return z
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


def run_case(
    placer, sampler, pool, i, inst, name, steps, draws, seed
) -> tuple[dict, list]:
    """One case through one setting; return its result row and the kept boxes."""
    t_case = time.perf_counter()
    z = sample(placer, sampler, [inst], draws, 1000 * seed + 1 + i)
    sync()
    t_sampled = time.perf_counter()
    z = refine(z, [inst], draws, steps)
    sync()
    t_refined = time.perf_counter()
    zn = z.cpu().numpy().astype(np.float64)
    n = inst.block_count
    jobs = [
        (i, j, z_to_xywh_np(zn[j, :n], inst.area_targets, inst.s)) for j in range(draws)
    ]
    t_decoded = time.perf_counter()
    results = sorted(
        pool.imap_unordered(legalize_job, jobs, chunksize=1), key=lambda r: r[1]
    )
    t_legalized = time.perf_counter()
    feasible = [r for r in results if is_feasible(r[2])]
    best = min(feasible or results, key=lambda r: r[2]["hpwl"])
    row = {
        "case": name,
        "n": n,
        "seed": seed,
        "outline": list(inst.outline),
        "best_draw": best[1],
        "best": best[2],
        "legalized_feasible_rate": float(np.mean([is_feasible(r[2]) for r in results])),
        "hpwl_all": [r[2]["hpwl"] for r in results],
        "fits_all": [bool(r[2]["fits_outline"]) for r in results],
        "dead_space_all": [r[2]["dead_space"] for r in results],
        "times": {
            "sample_s": t_sampled - t_case,
            "refine_s": t_refined - t_sampled,
            "decode_s": t_decoded - t_refined,
            "legalize_s": t_legalized - t_decoded,
            "case_s": t_legalized - t_case,
            "legalize_ms_per_draw": float(np.mean([r[4] for r in results])),
        },
        "legalizer": [{k: float(v) for k, v in r[5].items()} for r in results],
    }
    print(
        f"  [{name:>6} n={n:3d}] seed {seed}: HPWL {best[2]['hpwl']:.1f},"
        f" dead space {best[2]['dead_space']:.3f}, fits {best[2]['fits_outline']},"
        f" {row['times']['case_s']:.2f}s",
        flush=True,
    )
    return row, best[3]


def summarize(rows: list[dict]) -> dict:
    return {
        "hpwl_mean": float(np.mean([r["best"]["hpwl"] for r in rows])),
        "dead_space_mean": float(np.mean([r["best"]["dead_space"] for r in rows])),
        "fits_rate": float(np.mean([bool(r["best"]["fits_outline"]) for r in rows])),
        "feasible_rate": float(np.mean([is_feasible(r["best"]) for r in rows])),
        "case_s_mean": float(np.mean([r["times"]["case_s"] for r in rows])),
    }


def main():
    started = time.perf_counter()
    names, cases = load_cases()
    model = load_model(CHECKPOINT, scale=SCALE, device=DEVICE)
    samplers = {
        nfe: model.sampler(nfe, SAMPLE_SOLVER, SAMPLE_PROJECTIONS)
        for nfe, _, _ in SETTINGS
    }
    pool = mp.get_context("forkserver").Pool(
        WORKERS, initializer=init_worker, initargs=(cases, LEGALIZE_ROUTE)
    )
    results, layouts = {}, {}
    print(f"{SUITE} {VARIANT} gamma {GAMMA} aspect {ASPECT}: {names}", flush=True)
    with torch.no_grad():
        placer = model.placer(refiner=None)
        for nfe, steps, draws in SETTINGS:
            refine(
                sample(placer, samplers[nfe], [cases[0]], draws, 0),
                [cases[0]],
                draws,
                steps,
            )
        sync()
        print(f"warm-up: {time.perf_counter() - started:.1f}s", flush=True)
        for nfe, steps, draws in SETTINGS:
            for seed in SEEDS:
                key = f"nfe{nfe}_steps{steps}_n{draws}_s{seed}"
                rows = []
                for i, (inst, name) in enumerate(zip(cases, names, strict=True)):
                    row, boxes = run_case(
                        placer, samplers[nfe], pool, i, inst, name, steps, draws, seed
                    )
                    rows.append(row)
                    layouts[f"{key}_{name}"] = boxes
                results[key] = {"rows": rows, "summary": summarize(rows)}
                s = results[key]["summary"]
                print(
                    f"=== {key}: HPWL {s['hpwl_mean']:.1f}, dead space"
                    f" {s['dead_space_mean']:.3f}, fits {s['fits_rate']:.2f},"
                    f" feasible {s['feasible_rate']:.2f}",
                    flush=True,
                )
    pool.close()
    pool.join()
    out = Path(OUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out.with_suffix(".npz"), **layouts)
    config = {
        "checkpoint": CHECKPOINT,
        "scale": SCALE,
        "suite": SUITE,
        "variant": VARIANT,
        "names": names,
        "gamma": GAMMA,
        "aspect": ASPECT,
        "map_pins": MAP_PINS,
        "settings": [list(s) for s in SETTINGS],
        "sample_projections": SAMPLE_PROJECTIONS,
        "weights": WEIGHTS,
        "lr": LR,
        "schedule": LR_SCHEDULE,
        "betas": list(BETAS),
        "route": LEGALIZE_ROUTE,
        "workers": WORKERS,
        "seeds": list(SEEDS),
    }
    total_s = time.perf_counter() - started
    out.write_text(
        json.dumps({"config": config, "results": results, "total_s": total_s}, indent=1)
    )
    print(f"wrote {OUT} ({total_s:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
