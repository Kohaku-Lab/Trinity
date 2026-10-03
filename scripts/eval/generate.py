"""Cache raw diffusion samples of the dev split for every NFE and projection set.

Generation only: no refiner, no legalizer, no scoring; the analysis scripts read the cache.
For every model in ``CHECKPOINTS`` it writes ``OUT_DIR/<run>/<set>_nfe<k>.npz`` per projection
set and NFE, plus ``meta.json``. Every shard starts from the same initial noise per case and
draw, so shards of one run (and of runs sharing ``NOISE_SEED``) are paired.

Shard layout: ``lat`` holds the ``z_s`` latents case-major then draw (case ``i`` draw ``j`` is
rows ``K * sum(bcount[:i]) + j * n_i`` onward), ``bcount`` the block count per case, ``k`` the
draws per case.

Run::

    kogine run scripts/eval/generate.py --config configs/eval/flagship/generate.py
"""

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from trinity.augment_ops import drop_wires
from trinity.data.splits import load_or_make_splits
from trinity.floorplan.data import LanceFloorplanStore, find_train_lance
from trinity.hub import TrinityModel, load_model

torch.set_float32_matmul_precision("high")

# Models: {run name: a .ckpt, a release directory or a Hugging Face repo id}.
CHECKPOINTS: dict[str, str] = {}
SCALES: dict[str, str] = {}  # {run name: release subfolder} for multi-release sources

# Dev split of the train Lance set (None = the project default path).
TRAIN_LANCE: str | None = None
DEV_PER_N_K: int = 100
DEV_RANDOM_SIZE: int = 2000
SPLIT_SEED: int = 20090220
DEV_LIMIT: int = 0  # >0 keeps only the first DEV_LIMIT dev cases
WIRE_DROP: float = 0.0  # fraction of nets and pins dropped from every dev case
WIRE_DROP_SEED: int = 20260829

# Sampling axes: {shard prefix: PROJECTION list (None = the sampler default, [] = free)}.
NFE: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
PROJECTION_SETS: dict[str, list | None] = {"projected": None}
K: int = 4  # draws per case
SAMPLE_SOLVER: str = "euler"
NOISE_SEED: int = 20260826

DEVICE: str = "cuda"
MAX_ROWS: int = 2000  # rows (cases x K) per GPU forward
OUT_DIR: str = "outputs/gen"


def git_sha() -> str:
    try:
        cmd = ["git", "rev-parse", "--short", "HEAD"]
        return subprocess.check_output(cmd, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def dev_cases():
    """The dev-split instances (wires dropped when ``WIRE_DROP > 0``) and the split params."""
    lance_path = find_train_lance(TRAIN_LANCE)
    store = LanceFloorplanStore(str(lance_path))
    splits = load_or_make_splits(
        store.block_counts,
        lance_path.parent / "splits.json",
        per_n_k=DEV_PER_N_K,
        random_size=DEV_RANDOM_SIZE,
        seed=SPLIT_SEED,
    )
    ids = splits.dev_per_n + splits.dev_random
    if DEV_LIMIT > 0:
        ids = ids[:DEV_LIMIT]
    cases = store.instances(ids)
    if WIRE_DROP > 0:
        cases = [
            drop_wires(case, WIRE_DROP, np.random.default_rng(WIRE_DROP_SEED + i))
            for i, case in enumerate(cases)
        ]
    return cases, splits.params


def build_samplers(model: TrinityModel) -> dict[tuple[str, int], object]:
    """One sampler per ``(projection set, NFE)``."""
    return {
        (set_name, nfe): model.sampler(nfe, SAMPLE_SOLVER, projections)
        for set_name, projections in PROJECTION_SETS.items()
        for nfe in NFE
    }


@torch.no_grad()
def sample_all(model: TrinityModel, cases, run: str) -> dict:
    """``{(set, nfe): [per-case (K, n, 3) latents]}`` in the original case order."""
    samplers = build_samplers(model)
    latents = {key: [None] * len(cases) for key in samplers}
    cases_per_chunk = max(1, MAX_ROWS // K)
    order = sorted(range(len(cases)), key=lambda i: cases[i].block_count)
    started = time.perf_counter()
    placer = model.placer(samples=K, refiner=None)
    for chunk, start in enumerate(range(0, len(order), cases_per_chunk)):
        indices = order[start : start + cases_per_chunk]
        group = [cases[i] for i in indices]
        ctx = placer.build_group_ctx(group)
        generator = torch.Generator(device=DEVICE).manual_seed(NOISE_SEED + chunk)
        shape = (ctx["rows"], ctx["max_n"], 3)
        noise = torch.randn(shape, device=DEVICE, generator=generator)
        for key, sampler in samplers.items():
            placer.sampler = sampler
            z = placer.sample_latents(ctx, noise=noise.clone())
            z = z.float().cpu().numpy()
            for gi, case_index in enumerate(indices):
                n = cases[case_index].block_count
                latents[key][case_index] = z[gi * K : (gi + 1) * K, :n]
        done = min(start + cases_per_chunk, len(order))
        elapsed = time.perf_counter() - started
        print(f"  [{run}] {done}/{len(order)} cases  {elapsed:6.1f}s", flush=True)
    return latents


def write_shards(out: Path, latents: dict, cases) -> None:
    bcount = np.array([case.block_count for case in cases], dtype=np.int32)
    for (set_name, nfe), per_case in latents.items():
        flat = np.concatenate([z.reshape(-1, 3) for z in per_case], axis=0)
        np.savez(
            out / f"{set_name}_nfe{nfe}.npz",
            lat=flat.astype(np.float32),
            bcount=bcount,
            k=np.int32(K),
        )


def generate(run: str, source: str, cases, split_params: dict) -> None:
    out = Path(OUT_DIR) / run
    out.mkdir(parents=True, exist_ok=True)
    model = load_model(source, scale=SCALES.get(run), device=DEVICE)
    write_shards(out, sample_all(model, cases, run), cases)
    meta = {
        "run": run,
        "checkpoint": source,
        "scale": SCALES.get(run),
        "git": git_sha(),
        "n_cases": len(cases),
        "k": K,
        "nfe": list(NFE),
        "projection_sets": PROJECTION_SETS,
        "solver": SAMPLE_SOLVER,
        "noise_seed": NOISE_SEED,
        "split": split_params,
        "wire_drop": WIRE_DROP,
        "wire_drop_seed": WIRE_DROP_SEED,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    size_mb = sum(p.stat().st_size for p in out.iterdir()) / 1e6
    print(
        f"  [{run}] wrote {len(NFE) * len(PROJECTION_SETS)} shards, {size_mb:.0f} MB -> {out}"
    )


def main():
    cases, split_params = dev_cases()
    layouts = len(PROJECTION_SETS) * len(NFE) * len(cases) * K
    print(f"dev cases: {len(cases)}  (split {split_params})")
    print(f"{layouts:,} layouts per checkpoint", flush=True)
    for run, source in CHECKPOINTS.items():
        print(f"=== {run} <- {source}", flush=True)
        generate(run, source, cases, split_params)


if __name__ == "__main__":
    main()
