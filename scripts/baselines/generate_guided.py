"""Cache samples of a published placer drawn with its own in-sampler guidance over the dev split.

Uses the dev split, NFE ladder, ``K`` draws, chunking and per-chunk noise of
``scripts/eval/generate.py``, so a guided shard is paired draw for draw with the unguided shard
of the same checkpoint. Sampling runs the ``euler_guided`` sampler
(``trinity_baselines.ports.guidance``). Writes ``<OUT_DIR>/<NAME>/<SHARD>_nfe<f>.npz`` in the
``generate.py`` layout plus ``meta_nfe<list>.json`` with the guidance and its settings. Run::

    kogine run scripts/baselines/generate_guided.py --config configs/baselines/generate_guided_chipdiffusion.py
"""

import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from trinity.data.splits import load_or_make_splits
from trinity.floorplan.data import LanceFloorplanStore, find_train_lance
from trinity.registry import SAMPLER, build
from trinity.sampling.refine_case import build_refine_case
from trinity.training import DiffusionTrainer
from trinity_baselines import ports  # noqa: F401  (register: euler_guided)

torch.set_float32_matmul_precision("high")

NAME: str = ""  # output run name
CKPT: str = ""  # the placer's checkpoint
GUIDANCE: str = "chipd_opt"  # "chipd_opt" | "diffplace" | "macrodiff"
GUIDANCE_KWARGS: dict = {}
PROJECTIONS: list | None = None  # sampler PROJECTION specs; None = the sampler default
SHARD: str = "guided"  # shard file prefix
TRAIN_LANCE: str | None = None  # None = data/floorset_lite.lance
DEV_PER_N_K: int = 100
DEV_RANDOM_SIZE: int = 2000
SPLIT_SEED: int = 20090220
DEV_LIMIT: int = 0  # >0 keeps only the first DEV_LIMIT dev cases
NFE: tuple = (1, 2, 4, 8, 16, 32)
K: int = 4  # draws per case
NOISE_SEED: int = 20260826
DEVICE: str = "cuda"
MAX_ROWS: int = 500  # rows (cases x K) per forward
OUT_DIR: str = "outputs/gen"


def dev_cases():
    """The dev-split instances, their row ids and the split parameters."""
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
    if DEV_LIMIT > 0:
        ids = ids[:DEV_LIMIT]
    return store.instances(ids), ids, splits.params


def git_sha() -> str:
    try:
        command = ["git", "rev-parse", "--short", "HEAD"]
        return subprocess.check_output(command, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def build_samplers(framing) -> dict:
    """One guided Euler sampler per NFE."""
    samplers = {}
    for nfe in NFE:
        spec = {
            "name": "euler_guided",
            "num_steps": nfe,
            "guidance": GUIDANCE,
            "guidance_kwargs": GUIDANCE_KWARGS,
            "projections": PROJECTIONS,
        }
        samplers[nfe] = build(spec, SAMPLER, framing=framing)
    return samplers


@torch.no_grad()
def main():
    cases, _, split_params = dev_cases()
    print(
        f"dev cases: {len(cases)}; {NAME} <- {CKPT}; guidance {GUIDANCE} "
        f"{GUIDANCE_KWARGS}; NFE {list(NFE)} x K={K}",
        flush=True,
    )
    out = Path(OUT_DIR) / NAME
    out.mkdir(parents=True, exist_ok=True)
    model = DiffusionTrainer.load_from_checkpoint(
        CKPT, map_location=DEVICE, strict=False, weights_only=False
    )
    model.eval()
    model.eval_samples = K
    model.eval_max_batch = MAX_ROWS
    model.refiner = None
    if model.hparams.get("latent_param", "s_only") != "s_only":
        raise ValueError("generate_guided expects the identity latent (s_only)")
    samplers = build_samplers(model.framing)
    latents = {nfe: [None] * len(cases) for nfe in NFE}
    per_chunk = max(1, MAX_ROWS // K)
    order = sorted(range(len(cases)), key=lambda i: cases[i].block_count)
    started = time.perf_counter()
    with model._eval_mode():
        placer = model._make_placer()
        for chunk_index, start in enumerate(range(0, len(order), per_chunk)):
            indices = order[start : start + per_chunk]
            group = [cases[i] for i in indices]
            ctx = placer.build_group_ctx(group)
            shape = (ctx["rows"], ctx["max_n"], 3)
            case = build_refine_case(group, K, DEVICE)
            generator = torch.Generator(device=DEVICE).manual_seed(
                NOISE_SEED + chunk_index
            )
            noise = torch.randn(*shape, device=DEVICE, generator=generator)
            for nfe, sampler in samplers.items():
                sampler.case = case
                z = sampler.sample(
                    placer.backbone, ctx["cond"], shape, DEVICE, noise=noise.clone()
                )
                lat = model.param.from_latent(z).float().cpu().numpy()
                for gi, case_index in enumerate(indices):
                    n = cases[case_index].block_count
                    latents[nfe][case_index] = lat[gi * K : (gi + 1) * K, :n]
            done = min(start + per_chunk, len(order))
            elapsed = time.perf_counter() - started
            print(f"  [{NAME}] {done}/{len(order)} cases  {elapsed:7.1f}s", flush=True)

    bcount = np.array([case.block_count for case in cases], dtype=np.int32)
    for nfe, per_case in latents.items():
        flat = np.concatenate([lat.reshape(-1, 3) for lat in per_case], axis=0)
        np.savez(
            out / f"{SHARD}_nfe{nfe}.npz",
            lat=flat.astype(np.float32),
            bcount=bcount,
            k=np.int32(K),
        )
    meta = {
        "run": NAME,
        "ckpt": CKPT,
        "guidance": GUIDANCE,
        "guidance_kwargs": GUIDANCE_KWARGS,
        "projections": PROJECTIONS,
        "git": git_sha(),
        "n_cases": len(cases),
        "k": K,
        "nfe": list(NFE),
        "solver": "euler_guided",
        "noise_seed": NOISE_SEED,
        "ema": True,
        "split": split_params,
        "seconds": time.perf_counter() - started,
        "layout": "lat is case-major then draw; case i draw j is rows "
        "[K*sum(bcount[:i]) + j*n_i : ... + n_i]",
    }
    nfe_tag = "-".join(str(nfe) for nfe in NFE)
    (out / f"meta_nfe{nfe_tag}.json").write_text(json.dumps(meta, indent=2))
    total = sum(os.path.getsize(out / p) for p in os.listdir(out))
    print(f"  [{NAME}] wrote {len(latents)} shards, {total / 1e6:.0f} MB -> {out}")
    print("GEN DONE", flush=True)


if __name__ == "__main__":
    main()
