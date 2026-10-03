"""Data access: locate and download the datasets, parse them into ``FloorplanInstance``.

* ``paths`` -- dataset locations and downloads;
* ``floorset`` / ``lance_store`` -- the FloorSet validation set and the 1M train set;
* ``bookshelf`` / ``gsrc`` / ``mcnc`` -- the
  Bookshelf reader and the GSRC / MCNC suites.
"""

from trinity.floorplan.data.bookshelf import (
    fixed_outline,
    parse_bookshelf_case,
    with_fixed_outline,
)
from trinity.floorplan.data.floorset import (
    ground_truth_placement,
    load_validation_case,
    load_validation_set,
    parse_layout,
)
from trinity.floorplan.data.gsrc import (
    GSRC_SIZES,
    load_gsrc_case,
    load_gsrc_set,
)
from trinity.floorplan.data.lance_store import LANCE_DIRNAME, LanceFloorplanStore
from trinity.floorplan.data.mcnc import (
    MCNC_NAMES,
    load_mcnc_case,
    load_mcnc_set,
)
from trinity.floorplan.data.paths import (
    download_train_raw,
    find_floorset_root,
    find_gsrc_root,
    find_mcnc_root,
    find_train_lance,
)

__all__ = [
    "find_floorset_root",
    "find_gsrc_root",
    "find_mcnc_root",
    "find_train_lance",
    "MCNC_NAMES",
    "load_mcnc_case",
    "load_mcnc_set",
    "fixed_outline",
    "with_fixed_outline",
    "download_train_raw",
    "ground_truth_placement",
    "load_validation_case",
    "load_validation_set",
    "parse_layout",
    "GSRC_SIZES",
    "load_gsrc_case",
    "load_gsrc_set",
    "parse_bookshelf_case",
    "LanceFloorplanStore",
    "LANCE_DIRNAME",
]
