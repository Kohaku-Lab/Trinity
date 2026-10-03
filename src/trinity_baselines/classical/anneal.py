"""Records shared by the annealing solvers: a layout
snapshot along a run and the run's result."""

from dataclasses import dataclass, field

import numpy as np

from trinity.floorplan.types import Placement


@dataclass
class Checkpoint:
    """The annealer's layout after ``steps`` moves and ``seconds`` of annealing time."""

    steps: int
    seconds: float
    xywh: np.ndarray
    solver_cost: float
    label: str = ""


@dataclass
class AnnealResult:
    """The final layout of a run with the snapshots taken along the way."""

    placement: Placement
    seconds: float
    steps: int
    checkpoints: list[Checkpoint] = field(default_factory=list)
    info: dict = field(default_factory=dict)
