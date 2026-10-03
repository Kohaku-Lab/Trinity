"""Cache the sampler's starting noise over the
dev split, in the generation-cache format.

The model-free counterpart of ``generate.py``: every dev case and draw gets the same
``N(0, I)`` latent the sampler would start from (same chunking, same ``NOISE_SEED`` per
chunk), written as one shard ``OUT_DIR/<NAME>/<SHARD>.npz``. ``refine.py``,
``legalize.py``, ``score.py`` and ``dist_metrics.py`` read it unchanged; it is the input
of the refiner-only baseline.

Run::

    kogine run scripts/eval/generate_prior.py \\
        --config configs/eval/prior/generate_prior.py
"""

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from trinity.data.splits import dev_split_ids

NAME: str = "prior"
SHARD: str = "free_nfe0"

TRAIN_LANCE: str | None = None
DEV_PER_N_K: int = 100
DEV_RANDOM_SIZE: int = 2000
SPLIT_SEED: int = 20090220
DEV_LIMIT: int = 0  # >0 keeps only the first DEV_LIMIT dev cases

K: int = 4  # draws per case
NOISE_SEED: int = 20260826
DEVICE: str = "cuda"
MAX_ROWS: int = 2000  # rows (cases x K) per chunk
OUT_DIR: str = "outputs/gen"


def git_sha() -> str:
    try:
        cmd = ["git", "rev-parse", "--short", "HEAD"]
        return subprocess.check_output(cmd, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def dev_block_counts():
    """The block count of every dev case, and the split params."""
    store, ids, params = dev_split_ids(
        TRAIN_LANCE, DEV_PER_N_K, DEV_RANDOM_SIZE, SPLIT_SEED, DEV_LIMIT
    )
    block_counts = np.asarray(store.block_counts)[ids].astype(np.int32)
    return block_counts, params


def starting_noise(bcount: np.ndarray) -> list[np.ndarray]:
    """Per case the ``(K, n, 3)`` noise the sampler draws for it, in case order."""
    noise_per_case = [None] * len(bcount)
    cases_per_chunk = max(1, MAX_ROWS // K)
    order = sorted(range(len(bcount)), key=lambda i: bcount[i])
    started = time.perf_counter()
    for chunk, start in enumerate(range(0, len(order), cases_per_chunk)):
        indices = order[start : start + cases_per_chunk]
        max_n = int(max(bcount[i] for i in indices))
        generator = torch.Generator(device=DEVICE).manual_seed(NOISE_SEED + chunk)
        shape = (len(indices) * K, max_n, 3)
        noise = torch.randn(shape, device=DEVICE, generator=generator).cpu().numpy()
        for gi, case_index in enumerate(indices):
            n = int(bcount[case_index])
            noise_per_case[case_index] = noise[gi * K : (gi + 1) * K, :n]
        done = min(start + cases_per_chunk, len(order))
        elapsed = time.perf_counter() - started
        print(f"  [{NAME}] {done}/{len(order)} cases  {elapsed:6.1f}s", flush=True)
    return noise_per_case


def main():
    bcount, split_params = dev_block_counts()
    print(f"dev cases: {len(bcount)}  (split {split_params})", flush=True)
    out = Path(OUT_DIR) / NAME
    out.mkdir(parents=True, exist_ok=True)
    flat = np.concatenate([z.reshape(-1, 3) for z in starting_noise(bcount)], axis=0)
    np.savez(
        out / f"{SHARD}.npz", lat=flat.astype(np.float32), bcount=bcount, k=np.int32(K)
    )
    meta = {
        "run": NAME,
        "ckpt": None,
        "git": git_sha(),
        "n_cases": len(bcount),
        "k": K,
        "shards": [SHARD],
        "noise_seed": NOISE_SEED,
        "split": split_params,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  [{NAME}] wrote {out / f'{SHARD}.npz'}", flush=True)


if __name__ == "__main__":
    main()
