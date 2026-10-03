"""The ``scale_pack`` legalizer: expansion + constraint-graph packing.

Stages:

* (A) restore -- preplaced blocks to their target box, fixed-shape blocks to their
  target shape, soft blocks to their target area, all-soft MIB groups to one shared
  shape;
* (B) expand -- free centers moved outward from the layout center by the smallest factor
  (capped) at which no pair overlaps;
* (C) relations -- per pair, the axis with the larger gap and the order along it,
  rewritten so no block sits between an abutting pair or outside a boundary-coded block;
* (D) one packing LP over positions (shapes fixed): the relations, preplaced blocks
  pinned, cluster members abutting along a spanning tree, boundary-coded edges on the
  outline, minimizing the bounding box plus a displacement term;
* (E) a fixed ladder when the LP is infeasible: abutment and boundary as penalties; then
  the pinned pairs the LP cannot hold flipped to the other axis (up to ``FLIP_ROUNDS``);
  then free blocks above the preplaced blocks of those pairs; finally above every
  preplaced block, feasible by construction;
* (F) abutting pairs snapped to exact contact;
* (H, optional) the aspect of every soft non-MIB block re-optimized under the same
  relations, kept only when the checker scores it better (with a fixed outline: repeated
  up to ``RESHAPE_ROUNDS`` times while it gains a fit or a smaller bounding box).

The stages live in :mod:`.pack_relations` and :mod:`.pack_lp`; this module runs the
ladder. ``last_info()`` reports the latest call's expansion factor, rung, solves, time
and whether the aspect pass was kept.
"""

import time

import numpy as np

from trinity.floorplan.geometry import bounding_box, overlapping_pairs
from trinity.floorplan.legalize.pack_lp import (
    PIN_PENALTY,
    pinned_code_blocked,
    solve_lp,
    solve_reshape_lp,
)
from trinity.floorplan.legalize.pack_relations import (
    MAX_EXPAND,
    abut_tree,
    beyond,
    clear_between,
    clear_outside,
    count_between,
    expand,
    flip_pairs,
    relations,
    restore,
    snap_contacts,
)
from trinity.floorplan.registry import LEGALIZER
from trinity.floorplan.scoring.feasibility import check_feasibility
from trinity.floorplan.scoring.validate import validate
from trinity.floorplan.types import Placement

__all__ = [
    "MU",
    "PENALTY",
    "PIN_PENALTY",
    "CONTACT",
    "MAX_EXPAND",
    "FLIP_ROUNDS",
    "RESHAPE_ROUNDS",
    "RUNG_NAMES",
    "last_info",
    "restore",
    "expand",
    "relations",
    "clear_between",
    "clear_outside",
    "count_between",
    "abut_tree",
    "pinned_code_blocked",
    "solve_lp",
    "solve_reshape_lp",
    "flip_pairs",
    "beyond",
    "snap_contacts",
    "reshape_pass",
    "scale_pack_legalize",
]

MU = 0.5
PENALTY = 10.0
CONTACT = 1e-3
FLIP_ROUNDS = 6
RESHAPE_ROUNDS = 4
RUNG_NAMES = ("hard", "soft", "flip", "beyond_conflict", "beyond")
_LAST = {"lam": 1.0, "rung": 0, "solves": 0, "ms": 0.0, "reshaped": 0}


def last_info() -> dict:
    """``{"lam", "rung", "solves", "ms",
    "reshaped"}`` of the latest ``scale_pack`` call.

    ``rung`` 0 = no LP was needed; ``reshaped`` counts the kept aspect passes.
    """
    return dict(_LAST)


def _outline_better(inst, cand: np.ndarray, out: np.ndarray, box) -> bool:
    """Whether feasible ``cand`` fits the outline ``box`` better than ``out``.

    Better = it fits where ``out`` does not, or
    both fit equally and its bbox is smaller.
    """
    if not check_feasibility(Placement(cand, inst)).is_feasible():
        return False

    def fits(xywh):
        x0, y0, x1, y1 = bounding_box(xywh)
        return bool(x1 - x0 <= box[0] * (1 + 1e-9) and y1 - y0 <= box[1] * (1 + 1e-9))

    def area(xywh):
        x0, y0, x1, y1 = bounding_box(xywh)
        return (x1 - x0) * (y1 - y0)

    cand_fits, out_fits = fits(cand), fits(out)
    return cand_fits > out_fits or (
        cand_fits == out_fits and area(cand) < area(out) * (1 - 1e-6)
    )


def _attempt(
    inst,
    restored,
    expanded,
    rel,
    amode,
    bmode,
    mu,
    penalty,
    contact,
    pin_soft=False,
    rewrite=True,
    outline=None,
):
    """One packing solve: abutment tree, relation rewrite, LP, overlap check.

    Returns ``(layout or None, abutment edges, violated pinned pairs, relations used)``.
    """
    abut = abut_tree(inst, restored, rel, contact) if amode != "none" else []
    if abut and rewrite:
        rel = clear_between(rel, abut, expanded, inst.is_preplaced)
    out, violated = solve_lp(
        inst, restored, rel, abut, amode, bmode, mu, penalty, contact, pin_soft, outline
    )
    if out is not None and not pin_soft and overlapping_pairs(out):
        out = None
    return out, abut, violated, rel


def reshape_pass(
    inst,
    out: np.ndarray,
    rel: dict,
    abut: list,
    mu: float,
    penalty: float,
    contact: float,
    trust: float,
    outline=None,
):
    """Stage H on a legal layout; returns the reshaped layout or ``None``.

    The positions are re-packed with the new shapes when the exact-area projection left
    an overlap.
    """
    cand = solve_reshape_lp(inst, out, rel, abut, mu, penalty, contact, trust, outline)
    if cand is None:
        return None
    if overlapping_pairs(cand):
        cand, _ = solve_lp(
            inst, cand, rel, abut, "soft", "soft", mu, penalty, contact, outline=outline
        )
        if cand is None or overlapping_pairs(cand):
            return None
    return snap_contacts(cand, abut)


@LEGALIZER.register("scale_pack")
def scale_pack_legalize(
    placement: Placement,
    mu: float = MU,
    penalty: float = PENALTY,
    contact: float = CONTACT,
    cap: float = MAX_EXPAND,
    relation: str = "restored",
    reshape: bool = True,
    trust: float = 1.3,
    outline: bool = True,
) -> Placement:
    """Legalize one layout by expansion + constraint-graph packing (module docstring).

    ``mu`` weights the displacement term, ``penalty`` the relaxed abutment / boundary
    terms, ``contact`` is the perpendicular overlap kept on every abutting pair, ``cap``
    bounds the expansion factor. ``relation`` reads the pair relations from the
    ``restored`` or the ``expanded`` layout. ``reshape`` enables stage H (``trust``
    bounds the change per side). ``outline`` caps the LP at the instance's fixed outline
    (when it has one) on every rung but the last. Returns the restored layout when no
    rung solves.
    """
    t0 = time.perf_counter()
    inst = placement.instance
    restored = restore(inst, placement.xywh)
    fallback = Placement(restored, inst)
    if restored.shape[0] <= 1:
        _LAST.update(
            lam=1.0, rung=0, solves=0, ms=1000 * (time.perf_counter() - t0), reshaped=0
        )
        return fallback

    pinned = inst.is_preplaced
    box = inst.outline if (outline and inst.outline is not None) else None
    expanded, factor = expand(restored, ~pinned, cap)
    ref = expanded if relation == "expanded" else restored
    base = relations(ref, pinned)
    rel = clear_outside(base, inst, ref)
    solves = 0
    conflict = set()

    def attempt(r, amode, bmode, **kwargs):
        return _attempt(
            inst, restored, ref, r, amode, bmode, mu, penalty, contact, **kwargs
        )

    def flip_ladder():
        """Flip the pinned pairs a slack solve
        violates to the other axis, then re-solve."""
        nonlocal solves
        current = rel
        for _ in range(FLIP_ROUNDS):
            probe, _, violated, _ = attempt(
                current, "soft", "soft", pin_soft=True, outline=box
            )
            solves += 1
            if probe is None or not violated:
                return None, [], current
            conflict.update(k[0] if pinned[k[0]] else k[1] for k in violated)
            current = flip_pairs(current, violated, probe)
            got, got_abut, _, got_rel = attempt(current, "soft", "soft", outline=box)
            solves += 1
            if got is not None:
                return got, got_abut, got_rel
        return None, [], current

    def step(r, amode, bmode, rewrite=True, capped=True):
        got, got_abut, _, got_rel = attempt(
            r, amode, bmode, rewrite=rewrite, outline=box if capped else None
        )
        return got, got_abut, got_rel

    ladder = (
        lambda: step(rel, "hard", "hard"),
        lambda: step(rel, "soft", "soft"),
        flip_ladder,
        lambda: step(beyond(base, pinned, conflict), "soft", "soft", rewrite=False),
        lambda: step(beyond(base, pinned), "soft", "soft", rewrite=False, capped=False),
    )
    out, abut, rung, rel_used = None, [], 0, rel
    for k, rung_step in enumerate(ladder, start=1):
        out, abut, rel_used = rung_step()
        solves += 1
        if out is not None:
            rung = k
            break

    reshaped = 0
    if out is not None:
        out = snap_contacts(out, abut)
        reshape_args = (mu, penalty, contact, trust, box)
        if reshape and box is not None:
            for _ in range(RESHAPE_ROUNDS):
                cand = reshape_pass(inst, out, rel_used, abut, *reshape_args)
                solves += 2
                if cand is None or not _outline_better(inst, cand, out, box):
                    break
                out, reshaped = cand, reshaped + 1
        elif reshape and (
            np.any(inst.cluster_id > 0) or np.any(inst.boundary_code != 0)
        ):
            cand = reshape_pass(inst, out, rel_used, abut, *reshape_args)
            solves += 2
            if cand is not None:
                before = validate(Placement(out, inst), "full_fast").score
                after = validate(Placement(cand, inst), "full_fast").score
                if after.feasible and after.cost < before.cost:
                    out, reshaped = cand, 1

    elapsed_ms = 1000 * (time.perf_counter() - t0)
    _LAST.update(
        lam=float(factor), rung=rung, solves=solves, ms=elapsed_ms, reshaped=reshaped
    )
    return Placement(out, inst) if out is not None else fallback
