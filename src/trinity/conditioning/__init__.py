"""Per-block conditioning of a case: features, adjacency, anchors and the graph PE.

Importing this package registers the ``GRAPH_PE`` builders (``none``, ``rwpe``,
``spectral_draw``).
"""

from trinity.conditioning import graph_pe  # noqa: F401  (register: graph PE builders)
from trinity.conditioning.anchors import build_anchors
from trinity.conditioning.features import (
    DEFAULT_COND_OPTIONS,
    FEATURE_DIM,
    CondOptions,
    build_adjacency,
    build_b2b_dense,
    build_conditioning,
    build_features,
    pin_targets,
)

__all__ = [
    "FEATURE_DIM",
    "CondOptions",
    "DEFAULT_COND_OPTIONS",
    "build_adjacency",
    "build_b2b_dense",
    "build_anchors",
    "build_conditioning",
    "build_features",
    "pin_targets",
]
