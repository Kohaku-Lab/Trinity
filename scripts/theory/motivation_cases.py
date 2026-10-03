"""The motivation figure's snapshots: one raw draw through a published loop and
through the closed-form refiner, on cases chosen by a stated rule.

Selection: from the ported loops' legalization caches (``legalize.py`` outputs with
``SAVE_LAYOUTS``), per case the change of ``group_gap``, ``boundary_dist`` and
``fixed_dist`` from the raw draw 0 to the loop's final step, and the raw / final
``overlap_ratio``. Candidates are the cases with at least ``MIN_CLUSTERS`` clusters of
at least ``MIN_CLUSTER_SIZE`` blocks and at least one preplaced block, in each
block-count band of ``BANDS``; the chosen case of a band is the candidate at the median
of the ChipDiffusion loop's ``group_gap`` increase.

Each chosen case runs every row of ``ROWS`` (a raw run's draw 0 through one loop) and
records the snapshots' layouts (units of ``s``), soft vectors and ``U_refine``. Writes
``motivation_cases.npz`` and ``motivation_cases.json`` (the candidate table, the choices
with their ranks, the per-snapshot soft vectors).

Run::

    kogine run scripts/theory/motivation_cases.py \\
        --config configs/theory/motivation_cases.py
"""

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from trinity.conditioning.features import build_b2b_dense, pin_edges
from trinity.data.splits import dev_split_ids
from trinity.decode import z_to_xywh
from trinity.floorplan.scoring.vector import COLS
from trinity.sampling.refine_case import build_refine_case
from trinity.sampling.refine_closed import refine_closed
from trinity.sampling.refine_constraint import constraint_energy
from trinity.sampling.score_batched import metric_vector_batched
from trinity_baselines.ports.refine import PORTS

torch.set_float32_matmul_precision("high")

TRAIN_LANCE: str | None = None
DEV_PER_N_K: int = 100
DEV_RANDOM_SIZE: int = 2000
SPLIT_SEED: int = 20090220

LEGALIZE_DIR: str = "outputs/eval/legalize"
GEN_DIR: str = "outputs/gen"
NFE: int = 32
RAW_SHARD: str = "free_nfe32"
# {port: (legalize cache stem, final step)}; each
# cache is the port's own model's NFE-32 grid.
PORT_CACHES: dict[str, tuple[str, int]] = {
    "chipd_scheduled": ("baseline-chipdiffusion_port_chipd", 5000),
    "macrodiff": ("baseline-macrodiff_port_macrodiff", 500),
    "diffplace": ("baseline-diffplace_port_diffplace", 500),
}
# {source tag: generation run} of the raw draws.
RAW_RUNS: dict[str, str] = {
    "chipd": "baseline-chipdiffusion",
    "macro": "baseline-macrodiff",
    "trinity": "flagship",
}
BANDS: tuple[tuple[int, int], ...] = ((50, 70), (25, 35), (100, 120))
MIN_CLUSTERS: int = 2
MIN_CLUSTER_SIZE: int = 3
# {row name: (raw source tag, loop, steps,
# snapshots)}; loop "closed" = the paper refiner.
ROWS: dict[str, tuple[str, str, int, tuple[int, ...]]] = {
    "chipd_raw__chipd_loop": ("chipd", "chipd_scheduled", 5000, (0, 50, 500, 5000)),
    "chipd_raw__ours": ("chipd", "closed", 400, (0, 10, 100, 400)),
    "trinity_raw__ours": ("trinity", "closed", 400, (0, 10, 100, 400)),
    "macro_raw__macro_loop": ("macro", "macrodiff", 500, (0, 10, 100, 500)),
}
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
DEVICE: str = "cuda"
OUT: str = "outputs/theory/motivation_cases.json"


def git_sha() -> str:
    try:
        cmd = ["git", "rev-parse", "--short", "HEAD"]
        return subprocess.check_output(cmd, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def case_rows(array, case_index: int, bcount, draws: int):
    """Draw 0's rows of one case in a case-major-then-draw flat array."""
    offset = int(np.sum(bcount[:case_index].astype(np.int64)) * draws)
    return array[offset : offset + int(bcount[case_index])]


def raw_draw0(run: str, case_index: int) -> torch.Tensor:
    """Draw 0 of one case in a generation cache, as a ``(1, n, 3)`` tensor."""
    data = np.load(Path(GEN_DIR) / run / f"{RAW_SHARD}.npz")
    z = case_rows(data["lat"], case_index, data["bcount"], int(data["k"]))
    return torch.as_tensor(z.astype(np.float32), device=DEVICE)[None]


def candidate_table(store, ids):
    """Per case the loops' changes from the port
    caches and the cluster / preplaced counts."""
    caches = {}
    for port, (stem, final) in PORT_CACHES.items():
        data = np.load(Path(LEGALIZE_DIR) / f"{stem}.npz")
        keys = (
            "ids",
            "bcount",
            "draws",
            f"nfe{NFE}_steps0_soft",
            f"nfe{NFE}_steps{final}_soft",
        )
        caches[port] = {key: data[key] for key in keys}
    reference = caches["chipd_scheduled"]
    assert np.array_equal(
        reference["ids"], np.asarray(ids)
    ), "cache ids != dev split ids"
    col = COLS.index
    rows = []
    instances = store.instances(list(ids))
    for i, inst in enumerate(instances):
        cluster = inst.cluster_id
        sizes = [int((cluster == c).sum()) for c in np.unique(cluster[cluster > 0])]
        row = {
            "index": i,
            "id": int(ids[i]),
            "n": int(reference["bcount"][i]),
            "clusters_ge_min": int(sum(s >= MIN_CLUSTER_SIZE for s in sizes)),
            "preplaced": int(inst.is_preplaced.sum()),
        }
        for port, (_stem, final) in PORT_CACHES.items():
            s0 = caches[port][f"nfe{NFE}_steps0_soft"][i, 0]
            s1 = caches[port][f"nfe{NFE}_steps{final}_soft"][i, 0]
            row[port] = {
                "d_group_gap": float(s1[col("group_gap")] - s0[col("group_gap")]),
                "d_boundary_dist": float(
                    s1[col("boundary_dist")] - s0[col("boundary_dist")]
                ),
                "d_fixed_dist": float(s1[col("fixed_dist")] - s0[col("fixed_dist")]),
                "overlap_raw": float(s0[col("overlap_ratio")]),
                "overlap_final": float(s1[col("overlap_ratio")]),
            }
        rows.append(row)
    return rows, reference["bcount"], int(reference["draws"]), instances


def band_candidates(rows, lo: int, hi: int) -> list[dict]:
    return [
        r
        for r in rows
        if lo <= r["n"] <= hi
        and r["clusters_ge_min"] >= MIN_CLUSTERS
        and r["preplaced"] >= 1
    ]


def choose(rows) -> list[dict]:
    """Per band the candidate at the median ChipDiffusion ``group_gap`` increase."""
    chosen = []
    for lo, hi in BANDS:
        candidates = band_candidates(rows, lo, hi)
        candidates.sort(key=lambda r: r["chipd_scheduled"]["d_group_gap"])
        rank = len(candidates) // 2
        pick = candidates[rank]
        chosen.append(
            {
                "band": [lo, hi],
                "candidates": len(candidates),
                "rank_of_choice": rank,
                "index": pick["index"],
                "id": pick["id"],
                "n": pick["n"],
                "chipd_d_group_gap": pick["chipd_scheduled"]["d_group_gap"],
                "candidate_ids_by_rank": [c["id"] for c in candidates],
            }
        )
    return chosen


def instance_arrays(prefix: str, inst) -> dict:
    s = float(inst.s)
    pin_xy, pin_w, pin_block = pin_edges(inst)
    return {
        f"{prefix}_n": np.int64(inst.block_count),
        f"{prefix}_gt_xywh": (inst.gt_positions / s).astype(np.float32),
        f"{prefix}_cluster_id": inst.cluster_id.astype(np.int64),
        f"{prefix}_mib_id": inst.mib_id.astype(np.int64),
        f"{prefix}_boundary_code": inst.boundary_code.astype(np.int64),
        f"{prefix}_is_preplaced": inst.is_preplaced.astype(bool),
        f"{prefix}_is_fixed": inst.is_fixed.astype(bool),
        f"{prefix}_adjacency": build_b2b_dense(inst).astype(np.float32),
        f"{prefix}_pin_xy": (np.asarray(pin_xy, np.float32) / s).reshape(-1, 2),
        f"{prefix}_pin_block": np.asarray(pin_block, np.int64),
        f"{prefix}_pin_w": np.asarray(pin_w, np.float32),
    }


def run_row(loop: str, steps: int, snapshots, z0, case) -> dict[int, torch.Tensor]:
    if loop == "closed":
        return refine_closed(
            z0,
            case,
            WEIGHTS,
            steps,
            LR,
            LR_SCHEDULE,
            betas=BETAS,
            chunk=CHUNK,
            compile_loop=False,
            snapshots=tuple(snapshots),
        )
    return PORTS[loop](z0, case, steps, snapshots=tuple(snapshots))


def main():
    started = time.perf_counter()
    store, ids, _ = dev_split_ids(TRAIN_LANCE, DEV_PER_N_K, DEV_RANDOM_SIZE, SPLIT_SEED)
    rows, bcount, draws, instances = candidate_table(store, ids)
    chosen = choose(rows)
    text = {
        "selection_rule": {
            "bands": [list(b) for b in BANDS],
            "min_clusters": MIN_CLUSTERS,
            "min_cluster_size": MIN_CLUSTER_SIZE,
            "preplaced": ">= 1",
            "choice": (
                "the median ChipDiffusion group_gap increase among the candidates"
            ),
        },
        "candidates": {f"{lo}-{hi}": band_candidates(rows, lo, hi) for lo, hi in BANDS},
        "chosen": chosen,
        "rows": {},
    }
    arrays = {}
    col = COLS.index
    with torch.no_grad():
        for ci, choice in enumerate(chosen):
            i, inst = choice["index"], instances[choice["index"]]
            case = build_refine_case([inst], 1, DEVICE)
            arrays[f"c{ci}_id"] = np.int64(choice["id"])
            arrays |= instance_arrays(f"c{ci}", inst)
            text["rows"][f"c{ci}"] = {"id": choice["id"], "n": inst.block_count}
            ones = torch.ones(1, device=DEVICE)
            for row_name, (source, loop, steps, snapshots) in ROWS.items():
                z0 = raw_draw0(RAW_RUNS[source], i)
                if row_name == "chipd_raw__chipd_loop":
                    stem = PORT_CACHES["chipd_scheduled"][0]
                    cache = np.load(Path(LEGALIZE_DIR) / f"{stem}.npz")
                    cached = case_rows(cache[f"nfe{NFE}_steps0_lat"], i, bcount, draws)
                    assert np.allclose(
                        z0[0].cpu().numpy(), cached, rtol=1e-4, atol=1e-4
                    )
                out = run_row(loop, steps, snapshots, z0, case)
                step_list = sorted(out)
                boxes = [z_to_xywh(out[s], case.area_norm, ones) for s in step_list]
                xywh = np.stack([b[0].cpu().numpy() for b in boxes])
                soft = np.stack(
                    [metric_vector_batched(b, case)[0].cpu().numpy() for b in boxes]
                )
                energy = [
                    float(constraint_energy(out[s], case, WEIGHTS)[0])
                    for s in step_list
                ]
                energy = np.array(energy, np.float32)
                prefix = f"c{ci}_{row_name}"
                arrays[f"{prefix}_steps"] = np.asarray(step_list, np.int64)
                arrays[f"{prefix}_xywh"] = xywh.astype(np.float32)
                arrays[f"{prefix}_soft"] = soft.astype(np.float32)
                arrays[f"{prefix}_U_refine"] = energy
                text["rows"][f"c{ci}"][row_name] = {
                    "steps": step_list,
                    "soft": {
                        str(s): dict(zip(COLS, map(float, soft[j]), strict=True))
                        for j, s in enumerate(step_list)
                    },
                    "U_refine": energy.tolist(),
                }
                trace = " -> ".join(
                    f"[{s}] overlap {soft[j][col('overlap_ratio')]:.4f}"
                    f" soft {soft[j][col('soft_cost')]:.3f}"
                    for j, s in enumerate(step_list)
                )
                print(f"  c{ci} {row_name}: {trace}", flush=True)
    out = Path(OUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out.with_suffix(".npz"), **arrays)
    text["meta"] = {
        "git": git_sha(),
        "rows": {
            name: {"start": v[0], "loop": v[1], "steps": v[2], "snapshots": list(v[3])}
            for name, v in ROWS.items()
        },
        "weights": WEIGHTS,
        "lr": LR,
        "lr_schedule": LR_SCHEDULE,
        "betas": list(BETAS),
        "cols": list(COLS),
        "seconds": time.perf_counter() - started,
    }
    out.write_text(json.dumps(text, indent=1))
    print(f"wrote {OUT} ({text['meta']['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
