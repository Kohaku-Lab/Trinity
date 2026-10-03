"""The floorplanning problem: FloorSet and bookshelf
data, geometry, scoring, legalization, viz.

Model-agnostic. Importing the package leaves the registries
(:data:`LEGALIZER`, :data:`RENDERER`, :data:`SCORER`, :data:`JITTER`) populated with the
built-in components so ``build(spec, REGISTRY)`` works immediately.
"""

from trinity.floorplan import (
    data,  # noqa: F401  (public submodule)
    jitter,  # noqa: F401  (register: jitter modes)
    legalize,  # noqa: F401  (register: legalizer stages)
    scoring,  # noqa: F401  (register: full, stub scorers)
    viz,  # noqa: F401  (register: placement renderer)
)
from trinity.floorplan.registry import (
    JITTER,
    LEGALIZER,
    RENDERER,
    SCORER,
    Registry,
    build,
)
from trinity.floorplan.types import FloorplanInstance, Placement

__all__ = [
    "build",
    "Registry",
    "LEGALIZER",
    "RENDERER",
    "SCORER",
    "JITTER",
    "FloorplanInstance",
    "Placement",
    "data",
    "scoring",
    "jitter",
    "legalize",
    "viz",
]
