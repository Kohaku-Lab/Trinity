"""Load the GSRC floorplanning benchmark as :class:`FloorplanInstance` cases.

GSRC (``n10`` .. ``n300``, 10 to 300 blocks) in the Bookshelf ``.blocks`` / ``.pl`` /
``.nets`` format (:mod:`trinity.floorplan.data.bookshelf`). The HARD variant gives fixed
block shapes and an overlap-free reference layout (kept as ``gt_positions``); the SOFT
variant gives areas with an aspect range.
"""

from pathlib import Path

from trinity.floorplan.data.bookshelf import (
    ground_truth_placement,
    parse_bookshelf_case,
)
from trinity.floorplan.data.paths import GSRC_SIZES, find_gsrc_root
from trinity.floorplan.types import FloorplanInstance

__all__ = [
    "GSRC_SIZES",
    "load_gsrc_case",
    "load_gsrc_set",
    "gsrc_case_paths",
    "ground_truth_placement",
]


def gsrc_case_paths(
    root: Path, n_blocks: int, variant: str = "HARD"
) -> tuple[Path, Path, Path]:
    """The ``(blocks, pl, nets)`` paths of one GSRC case (``variant`` = ``HARD`` / ``SOFT``)."""
    base = root / variant / f"n{n_blocks}"
    return (
        base.with_suffix(".blocks"),
        base.with_suffix(".pl"),
        base.with_suffix(".nets"),
    )


def load_gsrc_case(
    n_blocks: int,
    variant: str = "HARD",
    root: str | Path | None = None,
    allow_download: bool = True,
    hard_shapes: bool | None = None,
) -> FloorplanInstance:
    """Load the GSRC case of ``n_blocks`` blocks.

    Hard blocks become fixed-shape blocks when ``hard_shapes`` (default: ``variant == "HARD"``).
    """
    override = None if root is None else str(root)
    gsrc_root = find_gsrc_root(override=override, allow_download=allow_download)
    hard = (variant == "HARD") if hard_shapes is None else hard_shapes
    paths = gsrc_case_paths(gsrc_root, n_blocks, variant)
    return parse_bookshelf_case(*paths, hard_shapes=hard)


def load_gsrc_set(
    variant: str = "HARD",
    sizes: tuple[int, ...] = GSRC_SIZES,
    root: str | Path | None = None,
    allow_download: bool = True,
    hard_shapes: bool | None = None,
) -> list[FloorplanInstance]:
    """Load the GSRC cases of the block counts ``sizes`` (default all six)."""
    return [
        load_gsrc_case(n, variant, root, allow_download, hard_shapes) for n in sizes
    ]
