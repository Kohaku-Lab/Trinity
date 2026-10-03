"""The proximal correction against t (Proposition 1).

For every model of ``MODELS``, the raw network output on the rectified-flow interpolant
``x_t = (1 - t) z0 + t x1`` of the reference latent ``z0`` of the first ``N_CASES`` dev cases,
at every ``t`` of ``T_GRID``, is compared with the ``REFERENCE`` model's output. Per case, t
and model it records:

* ``disp_rms`` / ``disp_rms_all`` -- the RMS gap to the reference output over the free
  channels / over every real channel;
* ``disp_along`` / ``disp_ortho`` -- the gap's component along ``g = -grad U_train`` of the
  reference output (free channels) and the RMS remainder; ``gradU_norm`` = ``|g|``;
* ``pred_prox`` -- the proximal prediction ``gamma_t |g|`` with
  ``gamma_t = lambda(t) t^2 / (2 c)`` (``lambda(t) = lambda t`` for a t-weighted model);
* ``U_train`` / ``U_refine`` -- the two constraint energies of the output, and its soft vector.

The output is the EMA backbone's ``x0`` with no projection. Writes ``prox_t.npz`` and
``prox_t.json`` (per model and t the means and medians).

Run::

    kogine run scripts/theory/prox_t.py --config configs/theory/prox_t.py
"""

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from trinity.data.splits import dev_split_ids
from trinity.decode import z_to_xywh
from trinity.floorplan.parameterize import xywh_to_z
from trinity.floorplan.scoring.vector import COLS
from trinity.hub import load_model
from trinity.sampling.refine_case import build_refine_case
from trinity.sampling.refine_constraint import constraint_energy
from trinity.sampling.score_batched import metric_vector_batched

torch.set_float32_matmul_precision("high")

TRAIN_LANCE: str | None = None
DEV_PER_N_K: int = 100
DEV_RANDOM_SIZE: int = 2000
SPLIT_SEED: int = 20090220
N_CASES: int = 2000

T_GRID: tuple[float, ...] = (
    0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1,
    0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
)  # fmt: skip
# {name: (checkpoint, aux weight lambda, whether lambda is multiplied by t)}; a checkpoint
# is a .ckpt, a release directory or a Hugging Face repo id.
MODELS: dict[str, tuple[str, float, bool]] = {
    "flagship": ("outputs/train/flagship/checkpoints/last.ckpt", 0.01, False),
    "no_aux": ("outputs/train/no_aux/checkpoints/last.ckpt", 0.0, False),
}
SCALES: dict[str, str] = {}  # {name: release subfolder} for multi-release sources
REFERENCE: str = "no_aux"
U_TRAIN: dict[str, float] = {
    "overlap": 1.0,
    "group": 1.0,
    "mib": 1.0,
    "boundary": 1.0,
    "wl": 1.0,
    "area": 1.0,
}
U_REFINE: dict[str, float] = {
    "overlap": 10.0,
    "group": 1.0,
    "mib": 1.0,
    "boundary": 1.0,
    "wl": 1.0,
    "area": 1.0,
}
MEAN_ELEMENTS: float | None = None  # 1 / c; None = measured on the train split

NOISE_SEED: int = 20260924
DEVICE: str = "cuda"
MAX_ROWS: int = 256
OUT: str = "outputs/theory/prox_t.json"

KEYS = (
    "disp_rms",
    "disp_rms_all",
    "disp_along",
    "disp_ortho",
    "gradU_norm",
    "pred_prox",
    "U_train",
    "U_refine",
)


def git_sha() -> str:
    try:
        cmd = ["git", "rev-parse", "--short", "HEAD"]
        return subprocess.check_output(cmd, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def dev_cases():
    """The first ``N_CASES`` dev cases, their ids, and 3 x the train split's mean block count."""
    store, dev, _ = dev_split_ids(TRAIN_LANCE, DEV_PER_N_K, DEV_RANDOM_SIZE, SPLIT_SEED)
    block_counts = np.asarray(store.block_counts)
    is_train = np.ones(len(block_counts), bool)
    is_train[np.asarray(dev)] = False
    ids = dev[:N_CASES]
    return store.instances(ids), ids, float(3 * block_counts[is_train].mean())


def reference_latents(cases, max_n: int) -> torch.Tensor:
    z0 = torch.zeros(len(cases), max_n, 3, device=DEVICE)
    for i, inst in enumerate(cases):
        z = xywh_to_z(inst.gt_positions, inst.area_targets, inst.s)
        z0[i, : inst.block_count] = torch.as_tensor(
            z, dtype=torch.float32, device=DEVICE
        )
    return z0


def channel_masks(case) -> tuple[torch.Tensor, torch.Tensor]:
    """``(free, real)`` ``(B, N, 3)``: movable positions and reshapeable rho / real tokens."""
    token = case.token_mask > 0.5
    movable = case.mob_pos > 0.5
    free = torch.stack([movable, movable, case.mob_shape > 0.5], -1) & token[..., None]
    real = token[..., None].expand(-1, -1, 3)
    return free.float(), real.float()


def energy_gradient(z, case, weights) -> torch.Tensor:
    with torch.enable_grad():
        z = z.detach().clone().requires_grad_(True)
        energy = constraint_energy(z, case, weights).sum()
        return torch.autograd.grad(energy, z)[0]


def masked_rms(d2, mask) -> torch.Tensor:
    return (d2 * mask).sum((1, 2)).div(mask.sum((1, 2)).clamp_min(1.0)).sqrt()


def measure_model(name, chunks, cases, mean_elements, store, reference) -> None:
    """Fill ``store`` with every key of model ``name``; the reference model fills ``reference``."""
    source, lam, t_weighted = MODELS[name]
    model = load_model(source, scale=SCALES.get(name), device=DEVICE)
    with torch.no_grad():
        placer = model.placer(samples=1, refiner=None)
        for chunk, indices in enumerate(chunks):
            group = [cases[i] for i in indices]
            ctx = placer.build_group_ctx(group)
            case = build_refine_case(group, 1, DEVICE)
            z0 = model.param.to_latent(reference_latents(group, ctx["max_n"]))
            generator = torch.Generator(device=DEVICE).manual_seed(NOISE_SEED + chunk)
            x1 = torch.randn(z0.shape, device=DEVICE, generator=generator)
            free, real = channel_masks(case)
            n_free = free.sum((1, 2)).clamp_min(1.0)
            token = (case.token_mask > 0.5)[..., None]
            ones = torch.ones(len(indices), device=DEVICE)
            for ti, t in enumerate(T_GRID):
                t_vec = torch.full((len(indices),), float(t), device=DEVICE)
                x_t = (1.0 - t) * z0 + t * x1
                x0_hat = model.param.from_latent(
                    placer.backbone(x_t, t_vec, ctx["cond"])
                )
                x0_hat = x0_hat * token
                if name == REFERENCE:
                    gradient = -energy_gradient(x0_hat, case, U_TRAIN) * free
                    reference[(chunk, ti)] = (x0_hat, gradient)
                x_ref, gradient = reference[(chunk, ti)]
                gap = x0_hat - x_ref
                grad_norm = gradient.flatten(1).norm(dim=1)
                along = (gap * gradient).sum((1, 2)) / grad_norm.clamp_min(1e-12)
                rms = masked_rms(gap * gap, free)
                ortho = (
                    (rms * rms * n_free - along * along).clamp_min(0.0) / n_free
                ).sqrt()
                lam_t = lam * (t if t_weighted else 1.0)
                gamma = lam_t * t * t * mean_elements / 2.0
                values = {
                    "disp_rms": rms,
                    "disp_rms_all": masked_rms(gap * gap, real),
                    "disp_along": along,
                    "disp_ortho": ortho,
                    "gradU_norm": grad_norm,
                    "pred_prox": gamma * grad_norm,
                    "U_train": constraint_energy(x0_hat, case, U_TRAIN),
                    "U_refine": constraint_energy(x0_hat, case, U_REFINE),
                }
                for key, value in values.items():
                    store[f"{key}__{name}"][ti, indices] = value.float().cpu().numpy()
                soft = metric_vector_batched(
                    z_to_xywh(x0_hat, case.area_norm, ones), case
                )
                store[f"soft__{name}"][ti, indices] = soft.float().cpu().numpy()
            print(f"  [{name}] chunk {chunk + 1}/{len(chunks)}", flush=True)
    del model, placer
    torch.cuda.empty_cache()


def summarize(store) -> dict:
    soft_col = COLS.index("soft_cost")
    summary = {}
    for name in MODELS:
        summary[name] = {}
        for ti, t in enumerate(T_GRID):
            row = {
                key: {
                    "mean": float(np.nanmean(store[f"{key}__{name}"][ti])),
                    "median": float(np.nanmedian(store[f"{key}__{name}"][ti])),
                }
                for key in KEYS
            }
            row["soft_cost_mean"] = float(
                np.nanmean(store[f"soft__{name}"][ti, :, soft_col])
            )
            summary[name][str(t)] = row
    return summary


def main():
    started = time.perf_counter()
    cases, ids, mean_elements = dev_cases()
    mean_elements = float(MEAN_ELEMENTS) if MEAN_ELEMENTS else mean_elements
    order = sorted(range(len(cases)), key=lambda i: cases[i].block_count)
    chunks = [order[s : s + MAX_ROWS] for s in range(0, len(order), MAX_ROWS)]
    shape = (len(T_GRID), len(cases))
    store = {
        f"{k}__{m}": np.full(shape, np.nan, np.float32) for m in MODELS for k in KEYS
    }
    for name in MODELS:
        store[f"soft__{name}"] = np.full(shape + (len(COLS),), np.nan, np.float32)
    print(
        f"{len(cases)} cases, {len(T_GRID)} t values, reference {REFERENCE}", flush=True
    )
    reference = {}
    for name in [REFERENCE] + [m for m in MODELS if m != REFERENCE]:
        measure_model(name, chunks, cases, mean_elements, store, reference)
    out = Path(OUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out.with_suffix(".npz"),
        ids=np.asarray(ids),
        bcount=np.asarray([c.block_count for c in cases]),
        t=np.asarray(T_GRID, np.float32),
        **store,
    )
    meta = {
        "git": git_sha(),
        "models": {
            m: {"checkpoint": v[0], "lambda": v[1], "t_weighted": v[2]}
            for m, v in MODELS.items()
        },
        "reference": REFERENCE,
        "n_cases": len(cases),
        "t_grid": list(T_GRID),
        "noise_seed": NOISE_SEED,
        "max_rows": MAX_ROWS,
        "u_train": U_TRAIN,
        "u_refine": U_REFINE,
        "cols": list(COLS),
        "c": 1.0 / mean_elements,
        "mean_elements_per_case": mean_elements,
        "seconds": time.perf_counter() - started,
    }
    out.write_text(json.dumps({"meta": meta, "summary": summarize(store)}, indent=1))
    print(f"wrote {OUT} ({meta['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
