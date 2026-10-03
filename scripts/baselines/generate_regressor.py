"""Cache the direct regressor's layout of every dev case in the generation-cache format.

The counterpart of ``scripts/eval/generate.py`` for ``RegressorTrainer`` checkpoints: the same
dev split and the same ``.npz`` layout (``lat`` = ``z_s`` rows, ``bcount``, ``k``) plus a
``meta.json``, so ``scripts/eval/score.py`` and ``scripts/eval/dist_metrics.py`` read it as
they read a diffusion cache. One forward per case (``k = 1``); no refiner, no legalizer, no
projections. Run::

    kogine run scripts/baselines/generate_regressor.py --config configs/baselines/generate_reg_z.py
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
from trinity_baselines.training import RegressorTrainer

torch.set_float32_matmul_precision("high")

CKPTS: dict = {}  # {run name: checkpoint path}
TRAIN_LANCE: str | None = None  # None = data/floorset_lite.lance
DEV_PER_N_K: int = 100
DEV_RANDOM_SIZE: int = 2000
SPLIT_SEED: int = 20090220
DEV_LIMIT: int = 0  # >0 keeps only the first DEV_LIMIT dev cases
SHARD: str = "reg"  # the shard file name
DEVICE: str = "cuda"
MAX_ROWS: int = 2000  # cases per forward
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


@torch.no_grad()
def generate(name: str, ckpt: str, cases: list, split_params: dict) -> None:
    """Write ``OUT_DIR/<name>/<SHARD>.npz`` and ``meta.json`` for one checkpoint."""
    out = Path(OUT_DIR) / name
    out.mkdir(parents=True, exist_ok=True)
    model = RegressorTrainer.load_from_checkpoint(
        ckpt, map_location=DEVICE, strict=False, weights_only=False
    )
    model.eval()
    model.eval_max_batch = MAX_ROWS

    order = sorted(range(len(cases)), key=lambda i: cases[i].block_count)
    latents = [None] * len(cases)
    started = time.perf_counter()
    with model._eval_mode():
        placer = model.make_placer()
        for start in range(0, len(order), MAX_ROWS):
            indices = order[start : start + MAX_ROWS]
            group = [cases[i] for i in indices]
            ctx = placer.build_group_ctx(group)
            z = placer.predict_latents(ctx).float().cpu().numpy()
            for gi, case_index in enumerate(indices):
                n = cases[case_index].block_count
                latents[case_index] = z[gi : gi + 1, :n]
            done = min(start + MAX_ROWS, len(order))
            elapsed = time.perf_counter() - started
            print(f"  [{name}] {done}/{len(order)} cases  {elapsed:6.1f}s", flush=True)

    bcount = np.array([case.block_count for case in cases], dtype=np.int32)
    flat = np.concatenate([lat.reshape(-1, 3) for lat in latents], axis=0)
    np.savez(
        out / f"{SHARD}.npz", lat=flat.astype(np.float32), bcount=bcount, k=np.int32(1)
    )
    meta = {
        "run": name,
        "ckpt": ckpt,
        "git": git_sha(),
        "n_cases": len(cases),
        "k": 1,
        "shards": [SHARD],
        "ema": True,
        "split": split_params,
        "layout": "lat is case-major; case i is rows "
        "[sum(bcount[:i]) : sum(bcount[:i]) + n_i]",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    total = sum(os.path.getsize(out / p) for p in os.listdir(out))
    print(f"  [{name}] wrote 1 shard, {total / 1e6:.0f} MB -> {out}", flush=True)


def main():
    cases, _, split_params = dev_cases()
    print(f"dev cases: {len(cases)}  (split {split_params})", flush=True)
    for name, ckpt in CKPTS.items():
        print(f"=== {name} <- {ckpt}", flush=True)
        generate(name, ckpt, cases, split_params)
    print("GEN DONE", flush=True)


if __name__ == "__main__":
    main()
