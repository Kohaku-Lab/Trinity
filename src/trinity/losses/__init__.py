"""Loss terms: the diffusion regression family and the auxiliary constraint terms.

Importing this package registers, in :data:`trinity.registry.LOSS`:

* the regression losses ``denoise`` / ``l2``, ``l1``, ``huber``, ``charbonnier``,
  ``pseudo_huber`` (:mod:`.diffusion`);
* the per-constraint aux terms ``overlap``, ``group``, ``mib``, ``boundary``, ``wl``, ``area``
  (:mod:`.constraint`, ``docs/physics.md``);
* the ``ref_*`` aux terms of the FloorSet reference recipe (:mod:`.aux`).
"""

from trinity.losses import (
    aux,  # noqa: F401  (register: ref_* terms)
    constraint,  # noqa: F401  (register: per-constraint aux terms)
    diffusion,  # noqa: F401  (register: regression losses)
)
from trinity.losses.base import LossContext, LossTerm

__all__ = ["LossContext", "LossTerm"]
