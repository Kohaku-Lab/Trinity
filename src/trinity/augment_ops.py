"""Training augmentations: an ordered list of registered ``AUGMENT_OP`` instance ops.

Every op maps a :class:`FloorplanInstance` to a new one; the dataset applies the list, then
encodes the result. The rigid ops (``rot90``, ``flip``, ``shift``) move blocks, pins and
boundary codes together; ``cond_dropout`` drops constraint annotations; ``wire_dropout``
drops nets.

Config form::

    AUGMENT = [
        {"rot90": {"choices": [0, 1, 2, 3]}},
        {"flip": {"p": 0.5}},
        {"shift": {"std": 0.1}},
    ]
"""

from dataclasses import replace

import numpy as np

from trinity.floorplan.augment import transform_instance
from trinity.floorplan.scoring.cost import METRIC_B2B_WL, METRIC_P2B_WL, hpwl_fast
from trinity.floorplan.types import (
    COL_BOUNDARY,
    COL_CLUSTER,
    COL_FIXED,
    COL_MIB,
    COL_PREPLACED,
    Placement,
)
from trinity.registry import AUGMENT_OP, build


@AUGMENT_OP.register("rot90")
class Rot90Op:
    """A random rotation by ``k * 90`` degrees; ``choices`` are the allowed ``k``."""

    def __init__(self, choices=(0, 1, 2, 3)) -> None:
        self.choices = tuple(choices)

    def apply(self, inst, rng):
        k = int(rng.choice(self.choices))
        return transform_instance(inst, rot=k) if k else inst


@AUGMENT_OP.register("flip")
class FlipOp:
    """An x mirror applied with probability ``p``."""

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def apply(self, inst, rng):
        return transform_instance(inst, flip=True) if rng.random() < self.p else inst


@AUGMENT_OP.register("shift")
class ShiftOp:
    """A Gaussian translation; ``std`` is in ``/ s`` units."""

    def __init__(self, std: float = 0.1) -> None:
        self.std = std

    def apply(self, inst, rng):
        dx = float(rng.normal(0.0, self.std)) * inst.s
        dy = float(rng.normal(0.0, self.std)) * inst.s
        return transform_instance(inst, shift=(dx, dy))


@AUGMENT_OP.register("cond_dropout")
class CondDropoutOp:
    """Zero constraint columns at per-group probabilities.

    Receives a drop probability per group -- ``boundary``, ``cluster``, ``mib``, ``fixed``,
    ``preplaced`` -- and ``cluster_per_group`` / ``mib_per_group`` (per group id, or the whole
    column). Dropping ``fixed`` / ``preplaced`` also nulls the affected ``target_positions``.
    Returns the instance with the edited ``constraints``.
    """

    def __init__(
        self,
        boundary=0.0,
        cluster=0.0,
        mib=0.0,
        fixed=0.0,
        preplaced=0.0,
        cluster_per_group=True,
        mib_per_group=True,
    ) -> None:
        self.p = {
            "boundary": boundary,
            "cluster": cluster,
            "mib": mib,
            "fixed": fixed,
            "preplaced": preplaced,
        }
        self.cluster_per_group = cluster_per_group
        self.mib_per_group = mib_per_group

    def apply(self, inst, rng):
        if not any(self.p.values()):
            return inst
        cons = inst.constraints.copy()
        targets = inst.target_positions.copy()

        if self.p["boundary"] and rng.random() < self.p["boundary"]:
            cons[:, COL_BOUNDARY] = 0.0
        self._drop_groups(
            cons, COL_CLUSTER, self.p["cluster"], self.cluster_per_group, rng
        )
        self._drop_groups(cons, COL_MIB, self.p["mib"], self.mib_per_group, rng)
        for name, col in (("fixed", COL_FIXED), ("preplaced", COL_PREPLACED)):
            if self.p[name] and rng.random() < self.p[name]:
                dropped = cons[:, col] != 0
                cons[dropped, col] = 0.0
                targets[dropped] = -1.0

        unchanged = not (cons != inst.constraints).any()
        if unchanged and (targets == inst.target_positions).all():
            return inst
        return replace(inst, constraints=cons, target_positions=targets)

    @staticmethod
    def _drop_groups(cons, col, prob, per_group, rng):
        """Zero the group-id column ``col`` with probability ``prob``.

        Per group id when ``per_group``, else the whole column at once.
        """
        if not prob:
            return
        ids = cons[:, col]
        groups = np.unique(ids[ids > 0])
        if groups.size == 0:
            return
        if not per_group:
            if rng.random() < prob:
                cons[:, col] = 0.0
            return
        for g in groups:
            if rng.random() < prob:
                cons[ids == g, col] = 0.0


def drop_wires(inst, p: float, rng):
    """Drop each b2b net and p2b pin of ``inst`` independently with probability ``p``.

    Returns a new instance; the GT HPWL metric entry is recomputed on the reduced netlist.
    """
    if p <= 0.0:
        return inst
    b2b = inst.b2b[rng.random(inst.b2b.shape[0]) >= p]
    p2b = inst.p2b[rng.random(inst.p2b.shape[0]) >= p]
    out = replace(inst, b2b=b2b, p2b=p2b)
    if (
        inst.metrics is not None
        and inst.gt_positions is not None
        and len(inst.metrics) >= 8
    ):
        m = np.array(inst.metrics, dtype=np.float64)
        m[METRIC_B2B_WL] = hpwl_fast(Placement(xywh=inst.gt_positions, instance=out))
        m[METRIC_P2B_WL] = 0.0
        out = replace(out, metrics=m)
    return out


@AUGMENT_OP.register("wire_dropout")
class WireDropoutOp:
    """Drop b2b nets and p2b pins at a per-instance rate drawn from a three-branch mixture.

    With probability ``keep`` the rate is 0, with probability ``packing`` it is drawn from
    ``packing_range``, otherwise from ``Beta(*beta)``; see :func:`drop_wires`.
    """

    def __init__(
        self,
        keep: float = 0.5,
        beta: tuple[float, float] = (1.0, 3.0),
        packing: float = 0.1,
        packing_range: tuple[float, float] = (0.9, 1.0),
    ) -> None:
        if keep + packing > 1.0:
            raise ValueError(f"keep ({keep}) + packing ({packing}) must be <= 1")
        self.keep = keep
        self.beta = tuple(beta)
        self.packing = packing
        self.packing_range = tuple(packing_range)

    def rate(self, rng) -> float:
        """Draw this instance's dropout rate from the mixture."""
        u = rng.random()
        if u < self.keep:
            return 0.0
        if u < self.keep + self.packing:
            return float(rng.uniform(*self.packing_range))
        return float(rng.beta(*self.beta))

    def apply(self, inst, rng):
        return drop_wires(inst, self.rate(rng), rng)


def _normalize_entry(entry):
    """A list entry as a ``{"name": ..., **kwargs}`` spec.

    Accepts ``"flip"``, ``{"name": "flip", "p": 0.5}`` and ``{"flip": {"p": 0.5}}``.
    """
    if isinstance(entry, str):
        return {"name": entry}
    if "name" in entry:
        return entry
    ((name, kw),) = entry.items()
    return {"name": name, **(kw or {})}


class AugmentPipeline:
    """The built ops of an ``AUGMENT`` list, applied in order (empty = identity)."""

    def __init__(self, ops: list) -> None:
        self.ops = ops

    def apply(self, inst, rng):
        for op in self.ops:
            inst = op.apply(inst, rng)
        return inst


def build_pipeline(spec) -> "AugmentPipeline":
    """Build the :class:`AugmentPipeline` of an ``AUGMENT`` spec.

    ``spec``: ``None`` / ``False`` (no ops) | ``True`` (rot90 + flip + shift) | a list of op
    entries (see :func:`_normalize_entry`).
    """
    if not spec:
        spec = []
    elif spec is True:
        spec = [{"rot90": {}}, {"flip": {}}, {"shift": {}}]
    if not isinstance(spec, list):
        raise TypeError(
            f"AUGMENT must be None / True / a list of ops, got {type(spec).__name__}: {spec!r}. "
            'Use e.g. [{"rot90": {"choices": [0,1,2,3]}}, {"flip": {"p": 0.5}}, {"shift": {"std": 0.1}}]'
        )
    ops = [build(_normalize_entry(e), AUGMENT_OP) for e in spec]
    return AugmentPipeline(ops)
