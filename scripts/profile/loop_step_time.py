"""The per-step time of every correction loop (theory T5).

The closed-form refiner and the ported published refiners run ``STEPS`` steps on the first
``rows`` dev cases (padded to their largest block count) for every ``rows`` of ``ROWS``; the
start latent is the reference layout's latent plus Gaussian jitter ``JITTER``. Reported per
loop and row count: the median of ``REPEATS`` timed runs after one warm-up, per step and per
layout, and the peak GPU memory. Output: one JSON.

Run::

    kogine run scripts/profile/loop_step_time.py --config configs/profile/loop_step_time.py
"""

import json
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from trinity.data.splits import load_or_make_splits
from trinity.floorplan.data import LanceFloorplanStore, find_train_lance
from trinity.floorplan.parameterize import xywh_to_z
from trinity.sampling.refine_case import build_refine_case
from trinity.sampling.refine_closed import refine_closed
from trinity_baselines.ports.refine import PORTS

torch.set_float32_matmul_precision("high")

TRAIN_LANCE: str | None = None
DEV_PER_N_K: int = 100
DEV_RANDOM_SIZE: int = 2000
SPLIT_SEED: int = 20090220
N_CASES: int = 2000

ROWS: tuple[int, ...] = (32, 256, 1000)
STEPS: int = 100
REPEATS: int = 5
JITTER: float = 0.05
LOOPS: tuple[str, ...] = ("closed", "chipd_scheduled", "diffplace", "macrodiff")
WEIGHTS: dict[str, float] = {
    "overlap": 10.0,
    "group": 1.0,
    "mib": 1.0,
    "boundary": 1.0,
    "wl": 1.0,
    "area": 1.0,
}
LR: float = 0.001
BETAS: tuple[float, float] = (0.0, 0.99)
CHUNK: int = 5
SEED: int = 20260924
DEVICE: str = "cuda"
OUT: str = "outputs/profile/loop_step_time.json"


def git_sha() -> str:
    try:
        cmd = ["git", "rev-parse", "--short", "HEAD"]
        return subprocess.check_output(cmd, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def dev_cases():
    lance_path = find_train_lance(TRAIN_LANCE)
    store = LanceFloorplanStore(str(lance_path))
    splits = load_or_make_splits(
        store.block_counts,
        lance_path.parent / "splits.json",
        per_n_k=DEV_PER_N_K,
        random_size=DEV_RANDOM_SIZE,
        seed=SPLIT_SEED,
    )
    return store.instances((splits.dev_per_n + splits.dev_random)[:N_CASES])


def loop_runner(loop: str, z0: torch.Tensor, case):
    if loop == "closed":
        return lambda: refine_closed(
            z0,
            case,
            WEIGHTS,
            STEPS,
            LR,
            "linear",
            betas=BETAS,
            chunk=CHUNK,
            compile_loop=True,
        )
    return lambda: PORTS[loop](z0, case, STEPS)


def timed(fn) -> float:
    """Median wall time of ``fn()`` over ``REPEATS`` synchronized calls after one warm-up."""
    fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(REPEATS):
        started = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - started)
    return statistics.median(times)


def jittered_reference(group, case) -> torch.Tensor:
    """The reference latents of ``group``, padded, plus seeded Gaussian jitter."""
    z0 = torch.zeros(len(group), case.token_mask.shape[1], 3, device=DEVICE)
    for i, inst in enumerate(group):
        z = xywh_to_z(inst.gt_positions, inst.area_targets, inst.s)
        z0[i, : inst.block_count] = torch.as_tensor(
            z, dtype=torch.float32, device=DEVICE
        )
    generator = torch.Generator(device=DEVICE).manual_seed(SEED)
    noise = torch.randn(z0.shape, device=DEVICE, generator=generator)
    return z0 + JITTER * noise * case.token_mask[..., None]


def main():
    started = time.perf_counter()
    cases = dev_cases()
    out = {}
    print(f"{len(cases)} cases; rows {list(ROWS)}; {STEPS} steps", flush=True)
    with torch.no_grad():
        for rows in ROWS:
            group = cases[:rows]
            case = build_refine_case(group, 1, DEVICE)
            z0 = jittered_reference(group, case)
            max_n = int(case.token_mask.shape[1])
            for loop in LOOPS:
                torch.cuda.reset_peak_memory_stats()
                seconds = timed(loop_runner(loop, z0, case))
                peak = torch.cuda.max_memory_allocated() / 2**20
                per_step_ms = 1e3 * seconds / STEPS
                out.setdefault(loop, {})[str(rows)] = {
                    "per_step_ms": per_step_ms,
                    "per_step_ms_per_layout": per_step_ms / rows,
                    "peak_mem_mb": peak,
                    "max_n": max_n,
                    "mean_n": float(np.mean([c.block_count for c in group])),
                }
                print(
                    f"  [{loop}] rows {rows:5d} (max n {max_n}):"
                    f" {per_step_ms:8.3f} ms per step, peak {peak:7.0f} MB",
                    flush=True,
                )
            del case, z0
            torch.cuda.empty_cache()
    meta = {
        "git": git_sha(),
        "device": torch.cuda.get_device_name(0),
        "n_cases": len(cases),
        "rows": list(ROWS),
        "steps": STEPS,
        "repeats": REPEATS,
        "jitter": JITTER,
        "weights": WEIGHTS,
        "lr": LR,
        "betas": list(BETAS),
        "seed": SEED,
        "seconds": time.perf_counter() - started,
    }
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps({"meta": meta, "loops": out}, indent=1))
    print(f"wrote {OUT} ({meta['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
