"""Runtime against the block count N at a fixed batch, on one idle GPU.

Measured per N of ``N_LADDER``:

* for every model of ``CHECKPOINTS``: the conditioning build per batch and the sampler's cost
  per NFE (the slope of the sampling time between the two ``NFE_PROBES``);
* the cost per step of the closed-form refiner and of the ported published refiners.

Cases: within the dev range, dev cases with exactly N blocks. Above it, synthetic cases that
tile ``N / SUB_N`` dev cases of ``SUB_N`` blocks in a grid (boxes and preplaced targets
shifted, cluster / MIB ids re-indexed, boundary codes kept), with every pin dropped and as
many cross-tile block-to-block nets added, sampled with probability proportional to the dev
set's net-length histogram and with weights drawn from the dev set's net weights. On a CUDA
out-of-memory the batch is halved and the rows used are recorded. Output: one JSON.

Run::

    kogine run scripts/profile/time_scaling.py --config configs/profile/time_scaling.py
"""

import json
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from trinity.data.splits import dev_split_cases
from trinity.floorplan.types import COL_CLUSTER, COL_MIB, FloorplanInstance
from trinity.hub import load_model
from trinity.sampling.refine_case import build_refine_case
from trinity.sampling.refine_closed import refine_closed
from trinity_baselines.ports.refine import PORTS

torch.set_float32_matmul_precision("high")

# Models: {run name: a .ckpt, a release directory or a Hugging Face repo id}.
CHECKPOINTS: dict[str, str] = {}
SCALES: dict[str, str] = {}  # {run name: release subfolder} for multi-release sources
SAMPLE_PROJECTIONS: list | None = None  # None = the sampler default, [] = free

TRAIN_LANCE: str | None = None
DEV_PER_N_K: int = 100
DEV_RANDOM_SIZE: int = 2000
SPLIT_SEED: int = 20090220

N_LADDER: tuple[int, ...] = (24, 40, 60, 80, 100, 120, 240, 360, 480, 720, 960, 1200)
SUB_N: int = 120  # block count of the tiles of a synthetic case
TILE_GAP: float = 0.1  # gap between tiles, as a fraction of the tile size
HIST_BINS: int = 40
HIST_MAX: float = 2.5  # net length / s collected by the last histogram bin

ROWS: int = 32  # layouts per forward, halved on out-of-memory
DRAWS: int = 1
NFE_PROBES: tuple[int, int] = (8, 32)
REPEATS: int = 5  # timed repetitions per measurement; the median is reported

REFINER_STEPS: int = 100
REFINER_LR: float = 0.001
REFINER_WEIGHTS: dict[str, float] = {
    "overlap": 10.0,
    "group": 1.0,
    "mib": 1.0,
    "boundary": 1.0,
    "wl": 1.0,
    "area": 1.0,
}
PORT_NAMES: tuple[str, ...] = ("chipd_scheduled", "diffplace", "macrodiff")

SEED: int = 20260922
DEVICE: str = "cuda"
OUT: str = "outputs/profile/time_scaling.json"


def git_sha() -> str:
    try:
        cmd = ["git", "rev-parse", "--short", "HEAD"]
        return subprocess.check_output(cmd, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def sync() -> None:
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def timed(fn) -> float:
    """Median wall time of ``fn()`` over ``REPEATS`` synchronized calls after one warm-up."""
    fn()
    sync()
    times = []
    for _ in range(REPEATS):
        started = time.perf_counter()
        fn()
        sync()
        times.append(time.perf_counter() - started)
    return statistics.median(times)


def with_oom_fallback(measure, rows: int):
    """``measure(rows)`` retried at half the rows on CUDA out-of-memory.

    Returns ``(rows used, result, peak MiB)``; ``(0, None, None)`` when even one row fails.
    """
    while rows >= 1:
        try:
            torch.cuda.reset_peak_memory_stats()
            result = measure(rows)
            return rows, result, torch.cuda.max_memory_allocated() / 2**20
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            rows //= 2
    return 0, None, None


def net_stats(cases):
    """The dev set's net-length histogram (Manhattan centre distance / s) and net weights."""
    lengths, weights, nets_per_block = [], [], []
    for case in cases:
        nets_per_block.append(case.b2b.shape[0] / case.block_count)
        if case.b2b.shape[0] == 0:
            continue
        i, j = case.b2b[:, 0].astype(int), case.b2b[:, 1].astype(int)
        cx = case.gt_positions[:, 0] + case.gt_positions[:, 2] / 2
        cy = case.gt_positions[:, 1] + case.gt_positions[:, 3] / 2
        lengths.append((np.abs(cx[i] - cx[j]) + np.abs(cy[i] - cy[j])) / case.s)
        weights.append(case.b2b[:, 2])
    lengths = np.clip(np.concatenate(lengths), 0, HIST_MAX)
    hist, edges = np.histogram(lengths, bins=HIST_BINS, range=(0, HIST_MAX))
    hist = hist.astype(np.float64) / hist.sum()
    return hist, edges, np.concatenate(weights), float(np.mean(nets_per_block))


def tile_case(tiles, hist, edges, weight_pool, rng) -> FloorplanInstance:
    """One synthetic case from the dev cases ``tiles``: pins dropped, cross-tile nets added."""
    grid = int(np.ceil(np.sqrt(len(tiles))))
    tile_w = max(
        float((t.gt_positions[:, 0] + t.gt_positions[:, 2]).max()) for t in tiles
    )
    tile_h = max(
        float((t.gt_positions[:, 1] + t.gt_positions[:, 3]).max()) for t in tiles
    )
    tile_w, tile_h = tile_w * (1 + TILE_GAP), tile_h * (1 + TILE_GAP)
    areas, constraints, boxes, targets, nets = [], [], [], [], []
    offset, mib_base, cluster_base, dropped_pins = 0, 0, 0, 0
    for ti, tile in enumerate(tiles):
        dx, dy = (ti % grid) * tile_w, (ti // grid) * tile_h
        gt = tile.gt_positions.copy()
        gt[:, :2] += (dx, dy)
        target = tile.target_positions.copy()
        target[tile.is_preplaced, :2] += (dx, dy)
        cons = tile.constraints.copy()
        cons[cons[:, COL_MIB] > 0, COL_MIB] += mib_base
        cons[cons[:, COL_CLUSTER] > 0, COL_CLUSTER] += cluster_base
        mib_base += int(tile.constraints[:, COL_MIB].max())
        cluster_base += int(tile.constraints[:, COL_CLUSTER].max())
        b2b = tile.b2b.copy()
        b2b[:, :2] += offset
        areas.append(tile.area_targets)
        constraints.append(cons)
        boxes.append(gt)
        targets.append(target)
        nets.append(b2b)
        dropped_pins += tile.p2b.shape[0]
        offset += tile.block_count
    area = np.concatenate(areas)
    gt = np.concatenate(boxes)
    nets.append(
        cross_tile_nets(tiles, gt, area, dropped_pins, hist, edges, weight_pool, rng)
    )
    return FloorplanInstance(
        block_count=area.shape[0],
        area_targets=area.astype(np.float32),
        constraints=np.concatenate(constraints).astype(np.float32),
        b2b=np.concatenate(nets).astype(np.float32),
        p2b=np.zeros((0, 3), dtype=np.float32),
        pins_pos=np.zeros((0, 2), dtype=np.float32),
        target_positions=np.concatenate(targets).astype(np.float32),
        gt_positions=gt.astype(np.float32),
    )


def cross_tile_nets(
    tiles, gt, area, count, hist, edges, weight_pool, rng
) -> np.ndarray:
    """``count`` block-to-block nets between tiles, lengths following ``hist``."""
    n = area.shape[0]
    s = float(np.sqrt(area.sum()))
    tile_id = np.repeat(np.arange(len(tiles)), [t.block_count for t in tiles])
    cx, cy = gt[:, 0] + gt[:, 2] / 2, gt[:, 1] + gt[:, 3] / 2
    ii, jj = np.triu_indices(n, k=1)
    cross = tile_id[ii] != tile_id[jj]
    ii, jj = ii[cross], jj[cross]
    length = (np.abs(cx[ii] - cx[jj]) + np.abs(cy[ii] - cy[jj])) / s
    p = hist[np.clip(np.digitize(length, edges) - 1, 0, HIST_BINS - 1)]
    m = min(count, ii.shape[0])
    p = p / p.sum() if p.sum() > 0 else None
    pick = rng.choice(ii.shape[0], size=m, replace=False, p=p)
    w = rng.choice(weight_pool, size=m)
    return np.stack([ii[pick], jj[pick], w], axis=1).astype(np.float32)


def ladder_cases(dev, rng, hist, edges, weight_pool) -> dict[int, list]:
    """``{N: ROWS cases}``: dev cases inside the dev range, tiled synthetic cases above."""
    by_n = {}
    for case in dev:
        by_n.setdefault(case.block_count, []).append(case)
    ladder = {}
    for n in N_LADDER:
        if len(by_n.get(n, [])) >= ROWS:
            ladder[n] = by_n[n][:ROWS]
            continue
        pool = by_n[SUB_N]
        ladder[n] = [
            tile_case(
                [
                    pool[i]
                    for i in rng.choice(len(pool), size=n // SUB_N, replace=False)
                ],
                hist,
                edges,
                weight_pool,
                rng,
            )
            for _ in range(ROWS)
        ]
    return ladder


def time_sampling(placer, samplers: dict, group) -> tuple[float, dict[int, float]]:
    """Conditioning time of ``group`` and the sampling time per NFE probe."""
    cond_s = timed(lambda: placer.build_group_ctx(group))
    ctx = placer.build_group_ctx(group)
    shape = (ctx["rows"], ctx["max_n"], 3)
    sample_s = {
        nfe: timed(
            lambda s=sampler: s.sample(placer.backbone, ctx["cond"], shape, DEVICE)
        )
        for nfe, sampler in samplers.items()
    }
    return cond_s, sample_s


def time_model(name: str, source: str, ladder) -> dict:
    """Conditioning and per-NFE sampling time of one model at every ladder point."""
    model = load_model(source, scale=SCALES.get(name), device=DEVICE)
    out = {}
    with torch.no_grad():
        placer = model.placer(samples=DRAWS, refiner=None)
        samplers = {
            nfe: model.sampler(nfe, projections=SAMPLE_PROJECTIONS)
            for nfe in NFE_PROBES
        }

        for n, cases in ladder.items():
            rows, result, peak = with_oom_fallback(
                lambda rows, cases=cases: time_sampling(placer, samplers, cases[:rows]),
                ROWS,
            )
            torch.cuda.empty_cache()
            if rows == 0:
                out[n] = {"rows": 0, "oom": True}
                continue
            cond_s, sample_s = result
            lo, hi = NFE_PROBES
            per_nfe = (sample_s[hi] - sample_s[lo]) / (hi - lo)
            layouts = rows * DRAWS
            out[n] = {
                "rows": layouts,
                "cond_s": cond_s,
                "sample_s": {str(nfe): t for nfe, t in sample_s.items()},
                "per_nfe_ms": 1e3 * per_nfe,
                "per_nfe_ms_per_layout": 1e3 * per_nfe / layouts,
                "peak_mem_mb": peak,
            }
            print(
                f"  [{name}] N={n:5d} rows={layouts:3d} cond {cond_s * 1e3:8.1f} ms"
                f"  per NFE {per_nfe * 1e3:8.2f} ms  peak {peak:7.0f} MB",
                flush=True,
            )
    del model
    torch.cuda.empty_cache()
    return out


def time_refiner(refiner: str, group) -> float:
    """Seconds of ``REFINER_STEPS`` steps of one refiner on a random latent of ``group``."""
    case = build_refine_case(group, DRAWS, DEVICE)
    z0 = torch.randn(len(group) * DRAWS, case.token_mask.shape[1], 3, device=DEVICE)
    if refiner in PORTS:
        return timed(lambda: PORTS[refiner](z0, case, REFINER_STEPS))
    return timed(
        lambda: refine_closed(
            z0,
            case,
            REFINER_WEIGHTS,
            REFINER_STEPS,
            REFINER_LR,
            "linear",
            betas=(0.0, 0.99),
            chunk=5,
            compile_loop=True,
        )
    )


def time_refiners(ladder) -> dict:
    """Per-step time of the closed-form refiner and every port at every ladder point."""
    out = {}
    for n, cases in ladder.items():
        out[n] = {}
        for refiner in ("closed",) + PORT_NAMES:
            rows, seconds, peak = with_oom_fallback(
                lambda rows, cases=cases, refiner=refiner: time_refiner(
                    refiner, cases[:rows]
                ),
                ROWS,
            )
            torch.cuda.empty_cache()
            if rows == 0:
                out[n][refiner] = {"rows": 0, "oom": True}
                continue
            layouts = rows * DRAWS
            per_step_ms = 1e3 * seconds / REFINER_STEPS
            out[n][refiner] = {
                "rows": layouts,
                "steps": REFINER_STEPS,
                "per_step_ms": per_step_ms,
                "per_step_ms_per_layout": per_step_ms / layouts,
                "peak_mem_mb": peak,
            }
            print(
                f"  [refiner {refiner}] N={n:5d} rows={layouts:3d}"
                f" per step {per_step_ms:8.3f} ms  peak {peak:7.0f} MB",
                flush=True,
            )
    return out


def main():
    rng = np.random.default_rng(SEED)
    dev, _, _ = dev_split_cases(TRAIN_LANCE, DEV_PER_N_K, DEV_RANDOM_SIZE, SPLIT_SEED)
    hist, edges, weight_pool, nets_per_block = net_stats(dev)
    ladder = ladder_cases(dev, rng, hist, edges, weight_pool)
    max_dev_n = max(case.block_count for case in dev)
    ladder_stats = {
        n: {
            "synthetic": n > max_dev_n,
            "nets_per_block": float(
                np.mean([c.b2b.shape[0] / c.block_count for c in cs])
            ),
            "pins_per_block": float(
                np.mean([c.p2b.shape[0] / c.block_count for c in cs])
            ),
        }
        for n, cs in ladder.items()
    }
    print(f"dev cases {len(dev)}; nets per block {nets_per_block:.2f}", flush=True)
    models = {
        name: time_model(name, source, ladder) for name, source in CHECKPOINTS.items()
    }
    refiners = time_refiners(ladder)
    device_name = torch.cuda.get_device_name(0) if DEVICE == "cuda" else DEVICE
    meta = {
        "git": git_sha(),
        "device": device_name,
        "rows": ROWS * DRAWS,
        "draws": DRAWS,
        "nfe_probes": list(NFE_PROBES),
        "repeats": REPEATS,
        "refiner_steps": REFINER_STEPS,
        "refiner_lr": REFINER_LR,
        "refiner_weights": REFINER_WEIGHTS,
        "sub_n": SUB_N,
        "tile_gap": TILE_GAP,
        "seed": SEED,
        "length_hist": hist.tolist(),
        "length_edges": edges.tolist(),
        "dev_nets_per_block": nets_per_block,
    }
    report = {
        "meta": meta,
        "ladder": {str(n): v for n, v in ladder_stats.items()},
        "models": {m: {str(n): v for n, v in d.items()} for m, d in models.items()},
        "refiners": {str(n): v for n, v in refiners.items()},
    }
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps(report, indent=1))
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
