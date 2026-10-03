"""Per-unit runtime costs on the official 100 cases, one case at a time on an idle machine.

Per case and against its block count:

* the legalizer's milliseconds per call (``LEGALIZE_ROUTE`` on the flagship's refined draws,
  one call at a time in this process), with its LP solves and ladder rung;
* PARSAC's milliseconds per annealing step and SP-SA's milliseconds per evaluation (one short
  single-core run each).

The sampler's per-NFE cost and the refiners' per-step costs come from ``time_scaling.py``.
Output: one JSON with a row per case and the means per block-count bucket.

Run::

    kogine run scripts/profile/unit_costs.py --config configs/profile/unit_costs.py
"""

import json
import time
from pathlib import Path

import numpy as np
import torch

import trinity_baselines.classical  # noqa: F401  (register: parsac / sp_sa solvers)
from trinity.floorplan.data import load_validation_set
from trinity.floorplan.legalize import apply_route, scale_pack
from trinity.floorplan.parameterize import z_to_xywh as z_to_xywh_np
from trinity.floorplan.registry import SOLVER
from trinity.floorplan.registry import build as build_solver
from trinity.floorplan.scoring.cost import use_fast_hpwl
from trinity.floorplan.types import Placement
from trinity.hub import load_model
from trinity.sampling.refine_case import build_refine_case
from trinity.sampling.refine_closed import refine_closed

torch.set_float32_matmul_precision("high")

# The model: a .ckpt, a release directory or a Hugging Face repo id (+ release subfolder).
CHECKPOINT: str = "outputs/train/flagship/checkpoints/last.ckpt"
SCALE: str | None = None
SAMPLE_SOLVER: str = "euler"
SAMPLE_PROJECTIONS: list | None = None  # None = the sampler default, [] = free
NFE: int = 8
STEPS: int = 400
DRAWS: int = 16
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

PARSAC_STEPS: int = 20_000
SPSA_EVALS: int = 2_000
SEED: int = 0
BUCKETS: tuple[tuple[int, int], ...] = (
    (21, 40),
    (41, 60),
    (61, 80),
    (81, 100),
    (101, 120),
)
CASE_LIMIT: int | None = None  # first CASE_LIMIT cases only
DEVICE: str = "cuda"
OUT: str = "outputs/profile/unit_costs.json"

BUCKET_KEYS = (
    "legalize_ms",
    "legalize_ms_median",
    "lp_solves",
    "rung",
    "parsac_ms_per_step",
    "spsa_ms_per_eval",
)


def refined_draws(placer, sampler, inst, seed: int) -> np.ndarray:
    """``DRAWS`` refined latents ``(DRAWS, n, 3)`` of one case."""
    placer.samples = DRAWS
    placer.sampler = sampler
    ctx = placer.build_group_ctx([inst])
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    noise = torch.randn(
        ctx["rows"], ctx["max_n"], 3, device=DEVICE, generator=generator
    )
    z = placer.sample_latents(ctx, noise=noise)
    case = build_refine_case([inst], DRAWS, DEVICE)
    z = refine_closed(
        z,
        case,
        WEIGHTS,
        STEPS,
        LR,
        LR_SCHEDULE,
        betas=BETAS,
        chunk=CHUNK,
        compile_loop=True,
    )[STEPS]
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return z.cpu().numpy().astype(np.float64)[:, : inst.block_count]


def legalizer_costs(inst, z: np.ndarray) -> dict:
    """Milliseconds, LP solves and ladder rung of one legalizer call per draw."""
    ms, solves, rungs = [], [], []
    for zj in z:
        placement = Placement(z_to_xywh_np(zj, inst.area_targets, inst.s), inst)
        started = time.perf_counter()
        with use_fast_hpwl(True):
            apply_route(placement, LEGALIZE_ROUTE)
        ms.append(1000 * (time.perf_counter() - started))
        info = scale_pack.last_info()
        solves.append(info["solves"])
        rungs.append(info["rung"])
    return {
        "legalize_ms": float(np.mean(ms)),
        "legalize_ms_median": float(np.median(ms)),
        "legalize_ms_max": float(np.max(ms)),
        "lp_solves": float(np.mean(solves)),
        "rung": float(np.mean(rungs)),
    }


def bucket_means(rows: list[dict]) -> dict:
    buckets = {}
    for lo, hi in BUCKETS:
        members = [r for r in rows if lo <= r["n"] <= hi]
        if members:
            means = {k: float(np.mean([r[k] for r in members])) for k in BUCKET_KEYS}
            buckets[f"{lo}-{hi}"] = means | {"cases": len(members)}
    return buckets


def main():
    started = time.perf_counter()
    cases = load_validation_set()[: CASE_LIMIT or None]
    model = load_model(CHECKPOINT, scale=SCALE, device=DEVICE)
    sampler = model.sampler(NFE, SAMPLE_SOLVER, SAMPLE_PROJECTIONS)
    parsac = build_solver(
        {"name": "parsac", "units_per_s": 4000.0, "checkpoint_stages": ()}, SOLVER
    )
    spsa = build_solver({"name": "sp_sa", "checkpoint_evals": ()}, SOLVER)
    rows = []
    with torch.no_grad():
        placer = model.placer(refiner=None)
        for i, inst in enumerate(cases):
            z = refined_draws(placer, sampler, inst, SEED + 1 + i)
            row = {"case": i, "id": inst.test_id, "n": inst.block_count}
            row |= legalizer_costs(inst, z)
            parsac_run = parsac.anneal(inst, budget=PARSAC_STEPS, seed=1)
            spsa_run = spsa.anneal(inst, budget=SPSA_EVALS, seed=1)
            row["parsac_ms_per_step"] = 1000 * parsac_run.seconds / parsac_run.steps
            row["spsa_ms_per_eval"] = 1000 * spsa_run.seconds / spsa_run.steps
            rows.append(row)
            print(
                f"  [{i + 1:3d}/{len(cases)}] n={inst.block_count:3d}"
                f" legalize {row['legalize_ms']:7.1f} ms per call"
                f"  PARSAC {row['parsac_ms_per_step']:.4f} ms/step"
                f"  SP-SA {row['spsa_ms_per_eval']:.4f} ms/eval",
                flush=True,
            )
    buckets = bucket_means(rows)
    device_name = torch.cuda.get_device_name(0) if DEVICE == "cuda" else DEVICE
    config = {
        "checkpoint": CHECKPOINT,
        "scale": SCALE,
        "nfe": NFE,
        "steps": STEPS,
        "draws": DRAWS,
        "route": LEGALIZE_ROUTE,
        "parsac_steps": PARSAC_STEPS,
        "spsa_evals": SPSA_EVALS,
        "seed": SEED,
        "device": device_name,
    }
    total_s = time.perf_counter() - started
    report = {"config": config, "rows": rows, "buckets": buckets, "total_s": total_s}
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps(report, indent=1))
    print(f"wrote {OUT} ({total_s:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
