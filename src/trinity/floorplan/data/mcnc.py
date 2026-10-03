"""Load the MCNC floorplanning benchmark as :class:`FloorplanInstance` cases.

MCNC (apte 9, xerox 10, hp 11, ami33 33, ami49 49 blocks, with terminals) in the
Bookshelf form of the UMich mirror (:mod:`trinity.floorplan.data.bookshelf`). ``HARD``
gives fixed block shapes, ``SOFT`` areas with an aspect range.
"""

from pathlib import Path

from trinity.floorplan.data.bookshelf import parse_bookshelf_case
from trinity.floorplan.data.paths import MCNC_NAMES, find_mcnc_root
from trinity.floorplan.types import FloorplanInstance

__all__ = ["MCNC_NAMES", "load_mcnc_case", "load_mcnc_set", "mcnc_case_paths"]


def mcnc_case_paths(
    root: Path, name: str, variant: str = "HARD"
) -> tuple[Path, Path, Path]:
    """The ``(blocks, pl, nets)`` paths of one MCNC
    case (``variant`` = ``HARD`` / ``SOFT``)."""
    base = root / variant / name
    return (
        base.with_suffix(".blocks"),
        base.with_suffix(".pl"),
        base.with_suffix(".nets"),
    )


def load_mcnc_case(
    name: str,
    variant: str = "HARD",
    root: str | Path | None = None,
    allow_download: bool = True,
    hard_shapes: bool | None = None,
) -> FloorplanInstance:
    """Load one MCNC case by name.

    Hard blocks become fixed-shape blocks when
    ``hard_shapes`` (default: ``variant == "HARD"``).
    """
    override = None if root is None else str(root)
    mcnc_root = find_mcnc_root(override=override, allow_download=allow_download)
    hard = (variant == "HARD") if hard_shapes is None else hard_shapes
    return parse_bookshelf_case(
        *mcnc_case_paths(mcnc_root, name, variant), hard_shapes=hard
    )


def load_mcnc_set(
    variant: str = "HARD",
    names: tuple[str, ...] = MCNC_NAMES,
    root: str | Path | None = None,
    allow_download: bool = True,
    hard_shapes: bool | None = None,
) -> list[FloorplanInstance]:
    """Load the MCNC cases ``names`` (default all five)."""
    return [
        load_mcnc_case(n, variant, root, allow_download, hard_shapes) for n in names
    ]
