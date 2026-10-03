"""The cost-versus-time curve of a classical solver: anneal a case set at several budgets and
seeds, snapshot the layout along every run, and score every snapshot.

Per snapshot the record holds the soft metric vector of the raw layout, the hard cost of the
layout after ``LEGALIZE_ROUTE`` (for bookshelf cases: the net HPWL and the outline fit), the
annealing seconds so far (scoring excluded) and the legalizer's milliseconds. Hard metrics are
only ever taken on legalized layouts.

``CASES`` selects the case set:

* ``{"source": "official"}`` -- the 100 FloorSet validation cases;
* ``{"source": "dev", "per_n": k, "n_list": [...]}`` -- the first ``k`` dev cases per block count;
* ``{"source": "dev", "limit": m}`` -- the first ``m`` dev cases;
* ``{"source": "dev", "spread": m}`` -- ``m`` dev cases at a constant stride by block count;
* ``{"source": "bookshelf", "suite": "mcnc" | "gsrc", ...}`` -- MCNC / GSRC under a fixed outline
  (``variant``, ``gamma``, ``aspect``, ``map_pins``, ``names`` / ``sizes``).

``INIT`` warm-starts every run from stored legalized boxes, either a ``scripts/eval/legalize.py``
run saved with ``SAVE_LAYOUTS`` (``{"npz", "nfe", "steps", "draw"}``, dev cases) or an ``.npz``
keyed per case (``{"npz", "key"}`` with ``{name}`` / ``{index}`` placeholders).

Every job runs in its own process (PARSAC keeps a per-process wirelength normalization) with at
most ``WORKERS`` jobs in flight. A job that raises, or is still running
``JOB_TIMEOUT_S + JOB_TIMEOUT_PER_STEP x budget`` seconds after submission, is skipped and listed
in ``meta["failed_jobs"]``. ``RESUME`` names a previous run's JSON over the same task list whose
finished jobs are reused. Writes one JSON (records and per-budget summaries) and one ``.npz``
(key ``job<i>``: every snapshot's boxes, ``(snapshots, n, 4)``). Run::

    kogine run scripts/baselines/anneal_curve.py --config configs/baselines/anneal_parsac.py
"""

import inspect
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

import trinity_baselines.classical  # noqa: F401  (register: parsac, sp_sa)
from trinity.data.splits import dev_split_ids
from trinity.floorplan.data import (
    GSRC_SIZES,
    MCNC_NAMES,
    LanceFloorplanStore,
    find_train_lance,
    load_gsrc_set,
    load_mcnc_set,
    load_validation_set,
    with_fixed_outline,
)
from trinity.floorplan.legalize import apply_route
from trinity.floorplan.registry import SOLVER, build
from trinity.floorplan.scoring import validate
from trinity.floorplan.scoring.bookshelf import orient_targets, score_bookshelf
from trinity.floorplan.scoring.cost import use_fast_hpwl
from trinity.floorplan.scoring.total import total_exp
from trinity.floorplan.scoring.vector import COLS, metric_vector
from trinity.floorplan.types import Placement
from trinity_baselines.classical.parsac import engine, engine_available

TRAIN_LANCE: str | None = None  # None = the project default path
DEV_PER_N_K: int = 100
DEV_RANDOM_SIZE: int = 2000
SPLIT_SEED: int = 20090220
CASES: dict = {"source": "dev", "per_n": 1, "n_list": [30, 60, 90, 120]}
SOLVER_SPEC: dict = {
    "name": "parsac",
    "units_per_s": 4000.0,
    "checkpoint_stages": tuple(range(0, 100, 5)),
}
BUDGETS: tuple = (10_000, 100_000)
SEEDS: tuple = (1,)
INIT: dict | None = None  # warm-start layouts (see the module docstring)
LEGALIZE_ROUTE: list = [
    {"name": "scale_pack", "relation": "restored", "reshape": True, "trust": 1.3}
]
SCORER: str = "full_fast"
WORKERS: int = 24
POOL_CONTEXT: str = "forkserver"
JOB_TIMEOUT_S: float = 120.0
JOB_TIMEOUT_PER_STEP: float = 3e-3
RESUME: str | None = None  # a previous run's JSON over the same task list
OUT: str = "outputs/classical/anneal_curve.json"

# Per-worker state, set by the pool initializer.
_worker = {}


def dev_ids_and_counts():
    """The dev-split row ids and their block counts."""
    store, ids, _ = dev_split_ids(TRAIN_LANCE, DEV_PER_N_K, DEV_RANDOM_SIZE, SPLIT_SEED)
    return ids, np.asarray(store.block_counts)[ids]


def bookshelf_cases():
    """The MCNC / GSRC cases of ``CASES`` under the fixed-outline protocol, with labels."""
    variant = CASES.get("variant", "HARD")
    if CASES["suite"] == "mcnc":
        names = list(CASES.get("names", MCNC_NAMES))
        cases = load_mcnc_set(variant, tuple(names))
    else:
        sizes = tuple(CASES.get("sizes", GSRC_SIZES))
        cases = load_gsrc_set(variant, sizes)
        names = [f"n{n}" for n in sizes]
    gamma = float(CASES.get("gamma", 0.10))
    aspect = float(CASES.get("aspect", 1.0))
    map_pins = bool(CASES.get("map_pins", False))
    cases = [with_fixed_outline(case, gamma, aspect, map_pins) for case in cases]
    labels = [f"{CASES['suite']}_{variant}_{name}" for name in names]
    return labels, cases


def select_cases():
    """``(labels, loaders, cases)``: a label and a ``(kind, key)`` loader per case.

    ``cases`` is the in-memory case list for the ``"official"`` loaders, ``None`` for dev cases
    (the workers read those from the Lance set).
    """
    if CASES["source"] == "official":
        cases = load_validation_set()
        labels = [f"official_{case.block_count}" for case in cases]
        return labels, [("official", i) for i in range(len(cases))], cases
    if CASES["source"] == "bookshelf":
        labels, cases = bookshelf_cases()
        return labels, [("official", i) for i in range(len(cases))], cases
    ids, counts = dev_ids_and_counts()
    if "limit" in CASES:
        chosen = list(range(int(CASES["limit"])))
    elif "spread" in CASES:
        order = sorted(range(len(ids)), key=lambda i: counts[i])
        stride = max(1, len(order) // int(CASES["spread"]))
        chosen = order[::stride][: int(CASES["spread"])]
    else:
        chosen = []
        for n in CASES["n_list"]:
            chosen += [i for i in range(len(ids)) if counts[i] == n][
                : int(CASES["per_n"])
            ]
    labels = [f"dev{i}_id{ids[i]}_n{counts[i]}" for i in chosen]
    loaders = [("dev", (i, int(ids[i]))) for i in chosen]
    return labels, loaders, None


def init_layouts(loaders, labels):
    """The warm-start boxes per case key, or ``None`` without ``INIT``."""
    if INIT is None:
        return None
    data = np.load(INIT["npz"])
    if "key" in INIT:
        layouts = {}
        for (_, key), label in zip(loaders, labels, strict=True):
            npz_key = INIT["key"].format(name=label.split("_")[-1], index=key)
            layouts[key] = data[npz_key].astype(np.float64)
        return layouts
    ids, bcount, draws = data["ids"], data["bcount"], int(data["draws"])
    boxes = data[f"nfe{INIT['nfe']}_steps{INIT['steps']}_xywh"]
    offsets = np.concatenate([[0], np.cumsum(bcount.astype(np.int64) * draws)])
    layouts = {}
    for kind, key in loaders:
        if kind != "dev":
            raise ValueError("INIT layouts are stored by dev index; use a dev case set")
        i, row = key
        if int(ids[i]) != row:
            raise ValueError(
                f"dev index {i} maps to id {ids[i]} in the npz, expected {row}"
            )
        n = int(bcount[i])
        start = offsets[i] + INIT["draw"] * n
        layouts[i] = boxes[start : start + n].astype(np.float64)
    return layouts


def normalize_key(key):
    """A record key as stored in memory (JSON turns tuples into lists)."""
    return tuple(key) if isinstance(key, list) else key


def resume(tasks, labels, loaders, records, arrays) -> set:
    """Copy the finished jobs of ``RESUME`` into ``records`` / ``arrays``; return their indices."""
    if not RESUME:
        return set()
    previous = json.loads(Path(RESUME).read_text())
    boxes = np.load(Path(RESUME).with_suffix(".npz"))
    by_task = {
        (r["kind"], normalize_key(r["key"]), r["seed"], r["budget"]): r
        for r in previous["records"]
    }
    done = set()
    for j, kind, key, seed, budget in tasks:
        record = by_task.get((kind, key, seed, budget))
        if record is None or f"job{record['job']}" not in boxes:
            continue
        arrays[f"job{j}"] = boxes[f"job{record['job']}"]
        record = dict(record, job=j, label=labels[loaders.index((kind, key))])
        records.append(record)
        done.add(j)
    print(f"resumed {len(done)} jobs from {RESUME}; {len(tasks) - len(done)} to run")
    return done


def init_worker(cases, route, scorer, spec, init, train_lance):
    _worker.update(
        cases=cases,
        route=route,
        scorer=scorer,
        spec=spec,
        init=init,
        train_lance=train_lance,
        store=None,
    )


def load_instance(kind, key):
    """The instance of one loader, inside a worker."""
    if kind == "official":
        return _worker["cases"][key]
    if _worker["store"] is None:
        lance_path = str(find_train_lance(_worker["train_lance"]))
        _worker["store"] = LanceFloorplanStore(lance_path)
    return _worker["store"].instances([key[1]])[0]


def score_bookshelf_snapshot(inst, xywh) -> dict:
    """A bookshelf snapshot: the legalized layout's net HPWL and its feasibility."""
    oriented = orient_targets(inst, xywh)
    placement = Placement(xywh=xywh.astype(np.float64), instance=oriented)
    started = time.perf_counter()
    with use_fast_hpwl(True):
        legalized = score_bookshelf(apply_route(placement, _worker["route"]))
    ms = 1000 * (time.perf_counter() - started)
    fits = legalized.fits_outline is None or legalized.fits_outline
    feasible = bool(legalized.overlap_free and legalized.shapes_kept and fits)
    nan = float("nan")
    return {
        "soft": [nan] * len(COLS),
        "cost": legalized.hpwl,
        "feasible": feasible,
        "hpwl_gap": nan,
        "area_gap": nan,
        "v_rel": nan,
        "legalize_ms": ms,
        "bookshelf": {"legalized": legalized.__dict__},
    }


def score_snapshot(inst, xywh) -> dict:
    """The soft vector of the raw layout and the hard cost of the legalized layout."""
    if inst.nets is not None:
        return score_bookshelf_snapshot(inst, xywh)
    soft = metric_vector(xywh.astype(np.float64), inst)
    placement = Placement(xywh=xywh.astype(np.float64), instance=inst)
    with use_fast_hpwl(True):
        started = time.perf_counter()
        legalized = apply_route(placement, _worker["route"])
        score = validate(legalized, _worker["scorer"]).score
        ms = 1000 * (time.perf_counter() - started)
    return {
        "soft": [float(v) for v in soft],
        "cost": float(score.cost),
        "feasible": bool(score.feasible),
        "hpwl_gap": float(score.hpwl_gap),
        "area_gap": float(score.area_gap),
        "v_rel": float(score.v_rel),
        "legalize_ms": ms,
    }


def run_job(task):
    """Anneal one ``(job, kind, key, seed, budget)`` task; return ``(record, snapshot boxes)``."""
    j, kind, key, seed, budget = task
    inst = load_instance(kind, key)
    solver = build(dict(_worker["spec"]), SOLVER)
    params = inspect.signature(solver.anneal).parameters
    kwargs = {}
    if _worker["init"] is not None and "init" in params:
        init_key = key[0] if isinstance(key, tuple) else key
        kwargs["init"] = Placement(xywh=_worker["init"][init_key], instance=inst)
    if inst.outline is not None and "outline" in params:
        kwargs["outline"] = inst.outline
    result = solver.anneal(inst, budget=budget, seed=seed, **kwargs)
    curve = [
        {
            "label": cp.label,
            "steps": cp.steps,
            "seconds": cp.seconds,
            "solver_cost": cp.solver_cost,
            **score_snapshot(inst, cp.xywh),
        }
        for cp in result.checkpoints
    ]
    boxes = np.stack([cp.xywh for cp in result.checkpoints]).astype(np.float32)
    info = {k: v for k, v in result.info.items() if k != "engine_cost"}
    record = {
        "job": j,
        "kind": kind,
        "key": key,
        "n": inst.block_count,
        "seed": seed,
        "budget": budget,
        "seconds": result.seconds,
        "steps": result.steps,
        "info": info,
        "curve": curve,
    }
    return record, boxes


def summarize(records) -> dict:
    """Per budget and snapshot label: mean legalized cost, feasibility, soft cost and seconds."""
    soft_col = COLS.index("soft_cost")
    out = {}
    for budget in sorted({r["budget"] for r in records}):
        runs = [r for r in records if r["budget"] == budget]
        labels = [cp["label"] for cp in runs[0]["curve"]]
        per_label = {}
        for k, label in enumerate(labels):
            points = [(r["curve"][k], r["n"]) for r in runs if k < len(r["curve"])]
            cost = np.array([p["cost"] for p, _ in points])
            per_label[label] = {
                "steps": int(np.mean([p["steps"] for p, _ in points])),
                "seconds": float(np.mean([p["seconds"] for p, _ in points])),
                "cost": float(cost.mean()),
                "cost_median": float(np.median(cost)),
                "feasible": float(np.mean([p["feasible"] for p, _ in points])),
                "total_exp": float(total_exp(cost.tolist(), [n for _, n in points])),
                "soft_cost": float(np.mean([p["soft"][soft_col] for p, _ in points])),
                "jobs": len(points),
            }
        out[str(budget)] = per_label
    return out


def describe(task) -> str:
    j, kind, key, seed, budget = task
    return f"job {j} ({kind} {key} seed {seed} budget {budget})"


def run_pool(tasks, labels, loaders, cases, init, records, arrays, failed, total):
    """Run ``tasks`` with at most ``WORKERS`` in flight, appending finished records."""
    soft_col = COLS.index("soft_cost")
    started = time.perf_counter()
    init_args = (cases, LEGALIZE_ROUTE, SCORER, SOLVER_SPEC, init, TRAIN_LANCE)
    context = mp.get_context(POOL_CONTEXT)
    processes = min(WORKERS, max(len(tasks), 1))
    with context.Pool(
        processes, initializer=init_worker, initargs=init_args, maxtasksperchild=1
    ) as pool:
        queue = list(tasks)
        pending = {}
        while pending or queue:
            while queue and len(pending) < WORKERS:
                task = queue.pop(0)
                handle = pool.apply_async(run_job, (task,))
                pending[task[0]] = (handle, time.perf_counter(), task)
            for j in list(pending):
                handle, submitted, task = pending[j]
                if not handle.ready():
                    timeout = JOB_TIMEOUT_S + JOB_TIMEOUT_PER_STEP * task[4]
                    if time.perf_counter() - submitted > timeout:
                        failed.append(task)
                        del pending[j]
                        print(f"  {describe(task)} timed out; skipped", flush=True)
                    continue
                del pending[j]
                try:
                    record, boxes = handle.get()
                except Exception as exc:  # noqa: BLE001
                    failed.append(task)
                    print(f"  {describe(task)} failed: {exc!r}; skipped", flush=True)
                    continue
                loader = (record["kind"], normalize_key(record["key"]))
                record["label"] = labels[loaders.index(loader)]
                records.append(record)
                arrays[f"job{record['job']}"] = boxes
                last = record["curve"][-1]
                print(
                    f"  [{len(records)}/{total}] {record['label']:<22} "
                    f"seed {record['seed']} budget {record['budget']:>9d} "
                    f"{record['seconds']:8.1f}s  soft {last['soft'][soft_col]:.3f}  "
                    f"legalized {last['cost']:.3f} feas {last['feasible']}  "
                    f"({len(record['curve'])} snapshots)  "
                    f"[{time.perf_counter() - started:6.0f}s]",
                    flush=True,
                )
            time.sleep(0.2)


def write_atomic(path: Path, write) -> None:
    """Write through a temporary file, then rename it onto ``path``."""
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    tmp.replace(path)


def main():
    started = time.perf_counter()
    labels, loaders, cases = select_cases()
    init = init_layouts(loaders, labels)
    combos = [
        (loader, seed, budget)
        for budget in sorted(BUDGETS, reverse=True)
        for loader in loaders
        for seed in SEEDS
    ]
    tasks = [
        (j, kind, key, seed, budget)
        for j, ((kind, key), seed, budget) in enumerate(combos)
    ]
    if SOLVER_SPEC["name"] == "parsac":
        if not engine_available():
            raise RuntimeError(
                "PARSAC source tree not found; run scripts/baselines/build_parsac.sh "
                "or set TRINITY_PARSAC"
            )
        engine()
    print(
        f"{len(labels)} cases x {len(SEEDS)} seeds x budgets {list(BUDGETS)} = "
        f"{len(tasks)} jobs on {WORKERS} workers; solver {SOLVER_SPEC}",
        flush=True,
    )
    records, arrays, failed = [], {}, []
    done_before = resume(tasks, labels, loaders, records, arrays)
    todo = [task for task in tasks if task[0] not in done_before]
    run_pool(todo, labels, loaders, cases, init, records, arrays, failed, len(tasks))

    records.sort(key=lambda r: r["job"])
    summary = summarize(records)
    for budget, per_label in summary.items():
        final = per_label["final"]
        print(
            f"  budget {budget:>9}: legalized cost {final['cost']:.4f} "
            f"(median {final['cost_median']:.4f}, total {final['total_exp']:.4f}), "
            f"feasible {final['feasible']:.3f}, soft {final['soft_cost']:.4f}, "
            f"{final['seconds']:.1f} s per run",
            flush=True,
        )
    out = Path(OUT)
    out.parent.mkdir(parents=True, exist_ok=True)

    def write_arrays(tmp: Path) -> None:
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **arrays)

    write_atomic(out.with_suffix(".npz"), write_arrays)
    meta = {
        "cases": CASES,
        "n_cases": len(labels),
        "solver": SOLVER_SPEC,
        "budgets": list(BUDGETS),
        "seeds": list(SEEDS),
        "init": INIT,
        "legalize_route": LEGALIZE_ROUTE,
        "scorer": SCORER,
        "workers": WORKERS,
        "cols": list(COLS),
        "seconds": time.perf_counter() - started,
        "failed_jobs": [
            {"job": t[0], "kind": t[1], "key": t[2], "seed": t[3], "budget": t[4]}
            for t in failed
        ],
    }
    payload = json.dumps(
        {"meta": meta, "summary": summary, "records": records}, indent=1
    )
    write_atomic(out, lambda tmp: tmp.write_text(payload))
    print(f"wrote {OUT} in {meta['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
