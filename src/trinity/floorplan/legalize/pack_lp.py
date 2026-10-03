"""The packing LPs of ``scale_pack`` (GLOP): positions under pair relations, and reshape.

Both LPs minimize the linearized bounding-box area ``h0 (R - L) + w0 (T - B)`` (over the
starting box ``w0 x h0``) plus a displacement term ``mu / (n s)`` per coordinate, subject to
the relations of :mod:`.pack_relations` (``pos[lo] + size[lo] <= pos[hi]``). Preplaced
blocks are fixed. Abutment and boundary codes enter as hard rows, as penalties
(``penalty / s`` per unit), or not at all.
"""

import numpy as np
from ortools.linear_solver import pywraplp

from trinity.floorplan.geometry import bounding_box

PIN_PENALTY = 1000.0
TIME_LIMIT_MS = 2000


def pinned_code_blocked(anchor: np.ndarray, pins: np.ndarray, i: int, bit: int) -> bool:
    """Whether another preplaced block lies beyond preplaced block ``i`` on its coded side."""
    others = pins[pins != i]
    if others.size == 0:
        return False
    if bit == 1:
        return bool(np.any(anchor[others, 0] < anchor[i, 0] - 1e-9))
    if bit == 2:
        right = anchor[others, 0] + anchor[others, 2]
        return bool(np.any(right > anchor[i, 0] + anchor[i, 2] + 1e-9))
    if bit == 8:
        return bool(np.any(anchor[others, 1] < anchor[i, 1] - 1e-9))
    top = anchor[others, 1] + anchor[others, 3]
    return bool(np.any(top > anchor[i, 1] + anchor[i, 3] + 1e-9))


class _PackingLP:
    """A GLOP model with the shared variables, rows and objective of both packing LPs."""

    def __init__(self, inst, anchor: np.ndarray) -> None:
        self.n = len(anchor)
        self.anchor = anchor
        self.pinned = inst.is_preplaced
        self.s = max(float(inst.s), 1e-12)
        x0, y0, x1, y1 = bounding_box(anchor)
        self.box = (x0, y0, x1, y1)
        self.w0, self.h0 = max(x1 - x0, 1e-12), max(y1 - y0, 1e-12)
        self.span = 20.0 * max(self.w0, self.h0, self.s) + float(
            np.abs(anchor[:, 2:]).sum()
        )
        self.solver = pywraplp.Solver.CreateSolver("GLOP")
        self.ok = self.solver is not None
        if self.ok:
            self.inf = self.solver.infinity()

    def var(self, lo: float, hi: float):
        return self.solver.NumVar(float(lo), float(hi), "")

    def position_vars(self):
        """``X``, ``Y``: fixed for preplaced blocks, free within the span otherwise."""
        x0, y0, x1, y1 = self.box
        X, Y = [], []
        for i in range(self.n):
            if self.pinned[i]:
                X.append(self.var(self.anchor[i, 0], self.anchor[i, 0]))
            else:
                X.append(self.var(x0 - self.span, x1 + self.span))
        for i in range(self.n):
            if self.pinned[i]:
                Y.append(self.var(self.anchor[i, 1], self.anchor[i, 1]))
            else:
                Y.append(self.var(y0 - self.span, y1 + self.span))
        return X, Y

    def outline_vars(self):
        """``L, R, B, T`` and the bbox-area objective over them."""
        x0, y0, x1, y1 = self.box
        L = self.var(x0 - self.span, x1 + self.span)
        R = self.var(x0 - self.span, x1 + self.span)
        B = self.var(y0 - self.span, y1 + self.span)
        T = self.var(y0 - self.span, y1 + self.span)
        self.obj = self.solver.Objective()
        self.obj.SetMinimization()
        area = self.w0 * self.h0
        self.obj.SetCoefficient(R, self.h0 / area)
        self.obj.SetCoefficient(L, -self.h0 / area)
        self.obj.SetCoefficient(T, self.w0 / area)
        self.obj.SetCoefficient(B, -self.w0 / area)
        return L, R, B, T

    def row(self, lo: float, hi: float, terms) -> None:
        """The constraint ``lo <= sum(coef * var) <= hi``."""
        constraint = self.solver.Constraint(lo, hi)
        for var, coef in terms:
            constraint.SetCoefficient(var, coef)

    def absdev(self, terms, target: float, weight: float) -> None:
        """Add ``weight * |sum(coef * var) - target|`` to the objective."""
        t = self.solver.NumVar(0.0, self.inf, "")
        self.row(-target, self.inf, [(t, 1.0)] + [(v, -c) for v, c in terms])
        self.row(target, self.inf, [(t, 1.0)] + [(v, c) for v, c in terms])
        self.obj.SetCoefficient(t, weight)

    def slack(self, weight: float):
        """A nonnegative variable with cost ``weight``."""
        t = self.solver.NumVar(0.0, self.inf, "")
        self.obj.SetCoefficient(t, weight)
        return t

    def cap_outline(self, L, R, B, T, outline) -> None:
        if outline is not None:
            self.row(-self.inf, float(outline[0]), [(R, 1.0), (L, -1.0)])
            self.row(-self.inf, float(outline[1]), [(T, 1.0), (B, -1.0)])

    def solve(self) -> bool:
        self.solver.SetTimeLimit(TIME_LIMIT_MS)
        status = self.solver.Solve()
        return status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE)


def solve_lp(
    inst,
    anchor: np.ndarray,
    rel: dict,
    abut: list,
    amode: str,
    bmode: str,
    mu: float,
    penalty: float,
    contact: float,
    pin_soft: bool = False,
    outline=None,
):
    """Stage D: the packing LP over positions with shapes fixed.

    ``amode`` / ``bmode`` in ``hard`` | ``soft`` | ``none`` set the abutment and boundary
    terms; ``pin_soft`` turns the relations of (free, preplaced) pairs into penalized
    slacks; ``outline = (W, H)`` caps the outline. Returns ``(layout or None, the pinned
    pairs whose relation the solution violates)``.
    """
    lp = _PackingLP(inst, anchor)
    if not lp.ok:
        return None, []
    n, s, inf, pinned = lp.n, lp.s, lp.inf, lp.pinned
    w, h = anchor[:, 2], anchor[:, 3]
    codes = inst.boundary_code
    X, Y = lp.position_vars()
    L, R, B, T = lp.outline_vars()

    for i in range(n):
        lp.row(-inf, 0.0, [(L, 1.0), (X[i], -1.0)])
        lp.row(float(w[i]), inf, [(R, 1.0), (X[i], -1.0)])
        lp.row(-inf, 0.0, [(B, 1.0), (Y[i], -1.0)])
        lp.row(float(h[i]), inf, [(T, 1.0), (Y[i], -1.0)])
        if not pinned[i]:
            lp.absdev([(X[i], 1.0)], float(anchor[i, 0]), mu / (n * s))
            lp.absdev([(Y[i], 1.0)], float(anchor[i, 1]), mu / (n * s))
    lp.cap_outline(L, R, B, T, outline)

    pos, size = (X, Y), (w, h)
    pin_slacks = []
    for key, (axis, lo, hi) in rel.items():
        terms = [(pos[axis][lo], 1.0), (pos[axis][hi], -1.0)]
        if pin_soft and pinned[lo] != pinned[hi]:
            t = lp.slack(PIN_PENALTY / s)
            terms.append((t, -1.0))
            pin_slacks.append((key, t))
        lp.row(-inf, -float(size[axis][lo]), terms)

    if amode != "none":
        for axis, lo, hi in abut:
            p, q = pos[axis], 1 - axis
            gap = [(p[lo], 1.0), (p[hi], -1.0)]
            lo_first = [(pos[q][lo], 1.0), (pos[q][hi], -1.0)]
            hi_first = [(pos[q][hi], 1.0), (pos[q][lo], -1.0)]
            if amode == "hard":
                lp.row(-float(size[axis][lo]), -float(size[axis][lo]), gap)
                lp.row(contact - float(size[q][lo]), inf, lo_first)
                lp.row(contact - float(size[q][hi]), inf, hi_first)
            else:
                lp.absdev(gap, -float(size[axis][lo]), penalty / s)
                lo_first.append((lp.slack(penalty / s), 1.0))
                lp.row(contact - float(size[q][lo]), inf, lo_first)
                hi_first.append((lp.slack(penalty / s), 1.0))
                lp.row(contact - float(size[q][hi]), inf, hi_first)

    if bmode != "none":
        pins = np.nonzero(pinned)[0]
        for i in np.nonzero(codes)[0]:
            code = int(codes[i])
            sides = (
                (1, X[i], L, 0.0),
                (2, X[i], R, -float(w[i])),
                (8, Y[i], B, 0.0),
                (4, Y[i], T, -float(h[i])),
            )
            for bit, var, edge, offset in sides:
                if not code & bit:
                    continue
                if pinned[i] and pinned_code_blocked(anchor, pins, i, bit):
                    continue
                if bmode == "hard":
                    lp.row(offset, offset, [(var, 1.0), (edge, -1.0)])
                else:
                    lp.absdev([(var, 1.0), (edge, -1.0)], offset, penalty / s)

    if not lp.solve():
        return None, []
    out = anchor.copy()
    for i in range(n):
        out[i, 0], out[i, 1] = X[i].solution_value(), Y[i].solution_value()
        if pinned[i]:
            out[i, :2] = anchor[i, :2]
    if not np.all(np.isfinite(out)):
        return None, []
    violated = [key for key, t in pin_slacks if t.solution_value() > 1e-6]
    return out, violated


def _width_bounds(inst, anchor: np.ndarray, free_shape: np.ndarray, trust: float):
    """Per-block width bounds: ``w / trust .. w * trust``, within the aspect bounds."""
    w = anchor[:, 2]
    w_lo, w_hi = w / trust, w * trust
    if inst.aspect_bounds is not None:
        lo, hi = inst.aspect_bounds[:, 0], inst.aspect_bounds[:, 1]
        bounded = free_shape & (lo > 0) & (hi > 0)
        area = np.maximum(inst.area_targets, 1e-12)
        w_lo = np.where(
            bounded, np.maximum(w_lo, np.sqrt(area * np.maximum(lo, 1e-12))), w_lo
        )
        w_hi = np.where(
            bounded, np.minimum(w_hi, np.sqrt(area * np.maximum(hi, 1e-12))), w_hi
        )
        w_lo, w_hi = np.minimum(w_lo, w_hi), np.maximum(w_lo, w_hi)
    return w_lo, w_hi


def solve_reshape_lp(
    inst,
    anchor: np.ndarray,
    rel: dict,
    abut: list,
    mu: float,
    penalty: float,
    contact: float,
    trust: float,
    outline=None,
):
    """Stage H: the packing LP with the shapes of the soft non-MIB blocks free.

    Each free width / height moves within a factor ``trust`` (and the aspect bounds), area
    kept by the tangent ``h0 W + w0 H = 2A``; abutment and boundary are penalties and the
    outline is capped at ``outline``. The solved shapes are projected back to exact area.
    Returns the layout or ``None``.
    """
    lp = _PackingLP(inst, anchor)
    if not lp.ok:
        return None
    n, s, inf, pinned = lp.n, lp.s, lp.inf, lp.pinned
    w, h = anchor[:, 2], anchor[:, 3]
    codes = inst.boundary_code
    free_shape = inst.is_soft & (inst.mib_id == 0) & (inst.area_targets > 0)
    X, Y = lp.position_vars()
    w_lo, w_hi = _width_bounds(inst, anchor, free_shape, trust)
    W = [
        lp.var(w_lo[i], w_hi[i]) if free_shape[i] else lp.var(w[i], w[i])
        for i in range(n)
    ]
    H = [
        lp.var(h[i] / trust, h[i] * trust) if free_shape[i] else lp.var(h[i], h[i])
        for i in range(n)
    ]
    L, R, B, T = lp.outline_vars()

    for i in range(n):
        lp.row(-inf, 0.0, [(L, 1.0), (X[i], -1.0)])
        lp.row(0.0, inf, [(R, 1.0), (X[i], -1.0), (W[i], -1.0)])
        lp.row(-inf, 0.0, [(B, 1.0), (Y[i], -1.0)])
        lp.row(0.0, inf, [(T, 1.0), (Y[i], -1.0), (H[i], -1.0)])
        if free_shape[i]:
            area = 2.0 * float(inst.area_targets[i])
            lp.row(area, area, [(W[i], float(h[i])), (H[i], float(w[i]))])
        if not pinned[i]:
            cx = float(anchor[i, 0] + anchor[i, 2] / 2)
            cy = float(anchor[i, 1] + anchor[i, 3] / 2)
            lp.absdev([(X[i], 1.0), (W[i], 0.5)], cx, mu / (n * s))
            lp.absdev([(Y[i], 1.0), (H[i], 0.5)], cy, mu / (n * s))
    lp.cap_outline(L, R, B, T, outline)

    pos, size = (X, Y), (W, H)
    for axis, lo, hi in rel.values():
        lp.row(
            -inf,
            0.0,
            [(pos[axis][lo], 1.0), (size[axis][lo], 1.0), (pos[axis][hi], -1.0)],
        )
    for axis, lo, hi in abut:
        q = 1 - axis
        gap = [(pos[axis][lo], 1.0), (size[axis][lo], 1.0), (pos[axis][hi], -1.0)]
        lp.absdev(gap, 0.0, penalty / s)
        lo_first = [(pos[q][lo], 1.0), (size[q][lo], 1.0), (pos[q][hi], -1.0)]
        lp.row(contact, inf, lo_first + [(lp.slack(penalty / s), 1.0)])
        hi_first = [(pos[q][hi], 1.0), (size[q][hi], 1.0), (pos[q][lo], -1.0)]
        lp.row(contact, inf, hi_first + [(lp.slack(penalty / s), 1.0)])

    pins = np.nonzero(pinned)[0]
    for i in np.nonzero(codes)[0]:
        code = int(codes[i])
        sides = (
            (1, [(X[i], 1.0), (L, -1.0)]),
            (2, [(X[i], 1.0), (W[i], 1.0), (R, -1.0)]),
            (8, [(Y[i], 1.0), (B, -1.0)]),
            (4, [(Y[i], 1.0), (H[i], 1.0), (T, -1.0)]),
        )
        for bit, terms in sides:
            if code & bit and not (
                pinned[i] and pinned_code_blocked(anchor, pins, i, bit)
            ):
                lp.absdev(terms, 0.0, penalty / s)

    if not lp.solve():
        return None
    out = anchor.copy()
    for i in range(n):
        if pinned[i]:
            continue
        out[i, 0], out[i, 1] = X[i].solution_value(), Y[i].solution_value()
        if free_shape[i]:
            width = max(W[i].solution_value(), 1e-12)
            out[i, 2], out[i, 3] = width, float(inst.area_targets[i]) / width
    return out if np.all(np.isfinite(out)) else None
