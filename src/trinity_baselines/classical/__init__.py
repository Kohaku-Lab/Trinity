"""Classical (non-learned) placers, registered under ``trinity.floorplan.registry.SOLVER``.

* :mod:`~trinity_baselines.classical.sp_sa` -- BBOPlace-Bench's sequence-pair simulated
  annealing (``sp_sa``).
* :mod:`~trinity_baselines.classical.parsac` -- PARSAC's constraints-aware B*-tree simulated
  annealing (``parsac``).
* :mod:`~trinity_baselines.classical.anneal` -- the snapshot and result records both return.
"""

from trinity_baselines.classical import (  # noqa: F401  (register: parsac, sp_sa)
    parsac,
    sp_sa,
)
from trinity_baselines.classical.anneal import AnnealResult, Checkpoint

__all__ = ["AnnealResult", "Checkpoint"]
