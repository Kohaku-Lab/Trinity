"""Reference layouts through every correction loop (Proposition 2(ii)).

Every dev case's reference layout (its ground-truth latent) is the start of each loop of
``LOOPS``: the ported published refiners at their published settings and the closed-form
refiner at the paper weights and at the training weights. Every snapshot is measured:

* ``disp_rms`` / ``disp_max`` -- RMS and largest centre displacement of the movable blocks
  (units of ``s``); ``disp_preplaced`` -- the displacement of the preplaced blocks (a check);
* ``U_train`` / ``U_refine`` -- the two constraint energies; the soft vector; the six term
  values at weight 1.

Also the reference's own rule violations (cluster pairs with a gap, coded sides off their
edge). Writes ``reference_loops.npz`` (per-case arrays) and ``reference_loops.json`` (per loop
and step the means and medians, and the share of cases whose soft cost rose).

Run::

    kogine run scripts/theory/reference_loops.py --config configs/theory/reference_loops.py
"""

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from trinity.data.splits import dev_split_cases
from trinity.decode import z_to_xywh
from trinity.floorplan.parameterize import xywh_to_z
from trinity.floorplan.scoring.vector import COLS
from trinity.losses import constraint as C
from trinity.sampling.refine_case import build_refine_case
from trinity.sampling.refine_closed import TERM_NAMES, refine_closed
from trinity.sampling.refine_constraint import TERMS, constraint_energy
from trinity.sampling.score_batched import metric_vector_batched
from trinity_baselines.ports.refine import PORTS

torch.set_float32_matmul_precision("high")

TRAIN_LANCE: str | None = None
DEV_PER_N_K: int = 100
DEV_RANDOM_SIZE: int = 2000
SPLIT_SEED: int = 20090220
DEV_LIMIT: int = 0  # >0 keeps only the first DEV_LIMIT dev cases

# {loop name: spec}; "port" loops are keys of trinity_baselines.ports.refine.PORTS.
LOOPS: dict[str, dict] = {
    "chipd_scheduled": {
        "kind": "port",
        "steps": 5000,
        "snapshots": (0, 10, 25, 50, 100, 200, 500, 1000, 2000, 5000),
    },
    "diffplace": {
        "kind": "port",
        "steps": 500,
        "snapshots": (0, 10, 25, 50, 100, 200, 500),
    },
    "macrodiff": {
        "kind": "port",
        "steps": 500,
        "snapshots": (0, 10, 25, 50, 100, 200, 500),
    },
    "closed": {
        "kind": "closed",
        "steps": 400,
        "snapshots": (0, 10, 25, 50, 100, 200, 400),
        "weights": {
            "overlap": 10.0,
            "group": 1.0,
            "mib": 1.0,
            "boundary": 1.0,
            "wl": 1.0,
            "area": 1.0,
        },
    },
    "closed_train": {
        "kind": "closed",
        "steps": 400,
        "snapshots": (0, 10, 25, 50, 100, 200, 400),
        "weights": {
            "overlap": 1.0,
            "group": 1.0,
            "mib": 1.0,
            "boundary": 1.0,
            "wl": 1.0,
            "area": 1.0,
        },
    },
}
LR: float = 0.001
LR_SCHEDULE: str | dict = "linear"
BETAS: tuple[float, float] = (0.0, 0.99)
CHUNK: int = 5
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
DEVICE: str = "cuda"
MAX_ROWS: int = 2000
OUT: str = "outputs/theory/reference_loops.json"

KEYS = ("disp_rms", "disp_max", "disp_preplaced", "U_train", "U_refine")


def git_sha() -> str:
    try:
        cmd = ["git", "rev-parse", "--short", "HEAD"]
        return subprocess.check_output(cmd, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def reference_latents(cases, max_n: int) -> torch.Tensor:
    z = torch.zeros(len(cases), max_n, 3, device=DEVICE)
    for i, inst in enumerate(cases):
        latent = xywh_to_z(inst.gt_positions, inst.area_targets, inst.s)
        z[i, : inst.block_count] = torch.as_tensor(
            latent, dtype=torch.float32, device=DEVICE
        )
    return z


def run_loop(name: str, spec: dict, z0: torch.Tensor, case) -> dict[int, torch.Tensor]:
    snapshots = tuple(spec["snapshots"])
    if spec["kind"] == "port":
        return PORTS[name](z0, case, spec["steps"], snapshots=snapshots)
    return refine_closed(
        z0,
        case,
        spec["weights"],
        spec["steps"],
        LR,
        LR_SCHEDULE,
        betas=BETAS,
        chunk=CHUNK,
        compile_loop=True,
        snapshots=snapshots,
    )


def violations(g, case):
    """Per case: same-cluster pairs, those with a gap, coded sides, those off their edge."""
    gap = C.gap_matrix(g)
    cluster = case.cluster_id
    member = (case.token_mask > 0.5) & (cluster > 0)
    same = (
        (cluster[:, :, None] == cluster[:, None, :])
        & member[:, :, None]
        & member[:, None, :]
    )
    same = torch.triu(same, diagonal=1)
    pairs = same.sum((1, 2)).to(g.dtype)
    bad_pairs = (same & (gap > 1e-6)).sum((1, 2)).to(g.dtype)
    x, y, xr, yt = C._edges(g)
    x_min, y_min, x_max, y_max = C.bbox(g, case.token_mask)
    code = case.boundary_code.long()
    real = case.token_mask > 0.5
    sides, bad_sides = torch.zeros_like(pairs), torch.zeros_like(pairs)
    side_distance = (
        (1, x - x_min[:, None]),
        (2, x_max[:, None] - xr),
        (4, y_max[:, None] - yt),
        (8, y - y_min[:, None]),
    )
    for bit, distance in side_distance:
        coded = ((code & bit) > 0) & real
        sides += coded.sum(1).to(g.dtype)
        bad_sides += (coded & (distance > 1e-6)).sum(1).to(g.dtype)
    return pairs, bad_pairs, sides, bad_sides


def check_reference(group, ids, g_star, soft0) -> None:
    """Assert the reference latents decode back to the ground truth with zero gaps."""
    for gi, inst in enumerate(group):
        n, s = inst.block_count, float(inst.s)
        back = (g_star[gi, :n] * s).cpu().numpy()
        error = np.abs(back - inst.gt_positions).max()
        assert np.allclose(back, inst.gt_positions, rtol=1e-4, atol=1e-3 * s), (
            ids[gi],
            error,
        )
    for column in ("hpwl_gap", "area_gap"):
        assert float(soft0[:, COLS.index(column)].abs().max()) < 1e-4, column


def measure_chunk(indices, cases, ids, store, soft0, viol, timing) -> None:
    group = [cases[i] for i in indices]
    case = build_refine_case(group, 1, DEVICE)
    z_star = reference_latents(group, case.token_mask.shape[1])
    ones = torch.ones(len(indices), device=DEVICE)
    g_star = z_to_xywh(z_star, case.area_norm, ones)
    s0 = metric_vector_batched(g_star, case)
    check_reference(group, [ids[i] for i in indices], g_star, s0)
    soft0[indices] = s0.cpu().numpy()
    viol[indices] = torch.stack(violations(g_star, case), 1).cpu().numpy()
    movable = ((case.token_mask > 0.5) & (case.mob_pos > 0.5)).float()
    preplaced = ((case.token_mask > 0.5) & (case.mob_pos < 0.5)).float()
    centre_star = g_star[..., :2] + g_star[..., 2:] / 2
    n_cases = len(cases)
    for name, spec in LOOPS.items():
        torch.cuda.synchronize()
        started = time.perf_counter()
        snapshots = run_loop(name, spec, z_star, case)
        torch.cuda.synchronize()
        timing[name] = timing.get(name, 0.0) + time.perf_counter() - started
        for step, z in snapshots.items():
            g = z_to_xywh(z, case.area_norm, ones)
            d = ((g[..., :2] + g[..., 2:] / 2) - centre_star).norm(dim=-1)
            values = {
                "disp_rms": (
                    (d * d * movable).sum(1) / movable.sum(1).clamp_min(1.0)
                ).sqrt(),
                "disp_max": (d * movable).max(1).values,
                "disp_preplaced": (d * preplaced).max(1).values,
                "U_train": constraint_energy(z, case, U_TRAIN),
                "U_refine": constraint_energy(z, case, U_REFINE),
                "soft": metric_vector_batched(g, case),
                "terms": torch.stack([TERMS[t](g, case) for t in TERM_NAMES], 1),
            }
            for key, value in values.items():
                array_key = f"{key}__{name}__{step}"
                if array_key not in store:
                    store[array_key] = np.zeros(
                        (n_cases,) + tuple(value.shape[1:]), np.float32
                    )
                store[array_key][indices] = value.float().cpu().numpy()


def summarize(store, soft0, timing) -> dict:
    soft_col = COLS.index("soft_cost")
    summary = {}
    for name, spec in LOOPS.items():
        per_step = {}
        for step in sorted(set(spec["snapshots"]) | {spec["steps"]}):
            entry = {
                key: {
                    "mean": float(np.mean(store[f"{key}__{name}__{step}"])),
                    "median": float(np.median(store[f"{key}__{name}__{step}"])),
                }
                for key in KEYS
            }
            for key in ("soft", "terms"):
                values = store[f"{key}__{name}__{step}"]
                entry[key] = {
                    "mean": values.mean(0).tolist(),
                    "median": np.median(values, 0).tolist(),
                }
            soft = store[f"soft__{name}__{step}"]
            entry["share_soft_cost_up"] = float(
                np.mean(soft[:, soft_col] > soft0[:, soft_col] + 1e-9)
            )
            per_step[str(step)] = entry
        summary[name] = {
            "steps": spec["steps"],
            "snapshots": list(spec["snapshots"]),
            "weights": spec.get("weights"),
            "seconds": timing[name],
            "per_step": per_step,
        }
        last = per_step[str(spec["steps"])]
        print(
            f"  {name}: final disp_rms {last['disp_rms']['mean']:.4f},"
            f" soft cost {last['soft']['mean'][soft_col]:.4f}"
            f" (reference {soft0[:, soft_col].mean():.4f}),"
            f" rose on {100 * last['share_soft_cost_up']:.1f}% of cases",
            flush=True,
        )
    return summary


def reference_report(soft0, viol) -> dict:
    return {
        "soft": {
            "mean": soft0.mean(0).tolist(),
            "median": np.median(soft0, 0).tolist(),
        },
        "share_cases_with_cluster_gap": float(np.mean(viol[:, 1] > 0)),
        "share_cluster_pairs_with_gap": float(
            viol[:, 1].sum() / max(viol[:, 0].sum(), 1.0)
        ),
        "share_cases_with_boundary_miss": float(np.mean(viol[:, 3] > 0)),
        "share_coded_sides_off_edge": float(
            viol[:, 3].sum() / max(viol[:, 2].sum(), 1.0)
        ),
        "cluster_pairs": float(viol[:, 0].sum()),
        "coded_sides": float(viol[:, 2].sum()),
    }


def main():
    started = time.perf_counter()
    cases, ids, _ = dev_split_cases(
        TRAIN_LANCE, DEV_PER_N_K, DEV_RANDOM_SIZE, SPLIT_SEED, DEV_LIMIT
    )
    order = sorted(range(len(cases)), key=lambda i: cases[i].block_count)
    store, timing = {}, {}
    soft0 = np.zeros((len(cases), len(COLS)), np.float32)
    viol = np.zeros((len(cases), 4), np.float32)
    print(f"{len(cases)} dev cases; loops {list(LOOPS)}", flush=True)
    with torch.no_grad():
        for start in range(0, len(order), MAX_ROWS):
            indices = order[start : start + MAX_ROWS]
            measure_chunk(indices, cases, ids, store, soft0, viol, timing)
            done = min(start + MAX_ROWS, len(order))
            elapsed = time.perf_counter() - started
            print(f"  {done}/{len(order)} cases  {elapsed:6.0f}s", flush=True)
    summary = summarize(store, soft0, timing)
    out = Path(OUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out.with_suffix(".npz"),
        ids=np.asarray(ids),
        bcount=np.asarray([c.block_count for c in cases]),
        soft__reference=soft0,
        violations__reference=viol,
        **store,
    )
    meta = {
        "git": git_sha(),
        "n_cases": len(cases),
        "loops": {
            name: {k: list(v) if isinstance(v, tuple) else v for k, v in spec.items()}
            for name, spec in LOOPS.items()
        },
        "lr": LR,
        "lr_schedule": LR_SCHEDULE,
        "betas": list(BETAS),
        "u_train": U_TRAIN,
        "u_refine": U_REFINE,
        "cols": list(COLS),
        "term_names": list(TERM_NAMES),
        "keys": list(KEYS),
        "max_rows": MAX_ROWS,
        "seconds": time.perf_counter() - started,
    }
    report = {
        "meta": meta,
        "reference": reference_report(soft0, viol),
        "loops": summary,
    }
    out.write_text(json.dumps(report, indent=1))
    print(f"wrote {OUT} ({meta['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
