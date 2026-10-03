"""Visualization: render placements to images.

Importing this package registers the ``placement`` renderer in
:data:`trinity.floorplan.registry.RENDERER`.
"""

from trinity.floorplan.viz.placement import (
    render_placement,
)  # noqa: F401  (register: placement)

__all__ = ["render_placement"]
