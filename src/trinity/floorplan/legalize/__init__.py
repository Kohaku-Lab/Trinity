"""Legalization: the ``scale_pack`` legalizer and the route / portfolio runner.

* ``scale_pack`` -- the ladder (stages in ``pack_relations`` and ``pack_lp``);
* ``route`` -- routes, portfolios and :func:`legalize`.

Importing this package registers ``scale_pack`` in
:data:`trinity.floorplan.registry.LEGALIZER`.
"""

from trinity.floorplan.legalize import scale_pack  # noqa: F401  (register: scale_pack)
from trinity.floorplan.legalize.route import (
    DEFAULT_PORTFOLIO,
    DEFAULT_ROUTE,
    LegalizeResult,
    apply_route,
    apply_stage,
    legalize,
    with_stage_options,
)

__all__ = [
    "DEFAULT_PORTFOLIO",
    "DEFAULT_ROUTE",
    "LegalizeResult",
    "apply_route",
    "apply_stage",
    "legalize",
    "with_stage_options",
]
