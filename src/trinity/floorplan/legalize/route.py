"""Legalization routes (stage lists applied left to right) and the best-of over routes.

A *route* is a list of stage specs (a registry key or a ``{"name": ..., **options}`` dict)
resolved in :data:`trinity.floorplan.registry.LEGALIZER`; a *portfolio* is a list of
routes. :func:`legalize` applies every route of a portfolio, scores each output and returns
the best (feasible first, then the lowest cost).
"""

from dataclasses import dataclass

from trinity.floorplan.registry import LEGALIZER, SCORER, resolve
from trinity.floorplan.scoring.cost import CaseScore
from trinity.floorplan.types import Placement

DEFAULT_ROUTE: list = ["scale_pack"]
DEFAULT_PORTFOLIO: list[list] = [DEFAULT_ROUTE]


@dataclass
class LegalizeResult:
    """The chosen legalized layout, its score and the route that produced it."""

    placement: Placement
    score: CaseScore
    route: list


def apply_stage(placement: Placement, stage_spec) -> Placement:
    """Apply one stage spec (registry key or ``{"name": ..., **options}``) to ``placement``."""
    if isinstance(stage_spec, dict):
        opts = dict(stage_spec)
        return resolve(opts.pop("name"), LEGALIZER)(placement, **opts)
    return resolve(stage_spec, LEGALIZER)(placement)


def apply_route(placement: Placement, route: list | None = None) -> Placement:
    """Apply the stages of ``route`` (default :data:`DEFAULT_ROUTE`) left to right."""
    current = placement
    for stage_spec in DEFAULT_ROUTE if route is None else route:
        current = apply_stage(current, stage_spec)
    return current


def with_stage_options(spec, stage_name: str, options: dict):
    """A copy of a route or portfolio with ``options`` merged into every ``stage_name`` stage.

    An option already set on the stage wins.
    """

    def rewrite(node):
        if isinstance(node, list):
            return [rewrite(child) for child in node]
        if isinstance(node, dict):
            return {**options, **node} if node.get("name") == stage_name else dict(node)
        return (
            {"name": stage_name, **options} if node == stage_name and options else node
        )

    return rewrite(spec)


def legalize(
    placement: Placement, portfolio: list[list] | None = None, scorer: str = "full"
) -> LegalizeResult:
    """Apply every route of ``portfolio`` (default :data:`DEFAULT_PORTFOLIO`) and keep the best.

    Each output is scored with ``scorer``; feasible outputs come first, then the lowest cost.
    """
    score_fn = resolve(scorer, SCORER)
    results = []
    for route in DEFAULT_PORTFOLIO if portfolio is None else portfolio:
        out = apply_route(placement, route)
        results.append(LegalizeResult(placement=out, score=score_fn(out), route=route))
    feasible = [r for r in results if r.score.feasible]
    return min(feasible or results, key=lambda r: r.score.cost)
