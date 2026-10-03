"""Random-access reader over the transcoded FloorSet-Lite Lance dataset.

The 1M training layouts are transcoded once (``scripts/data/transcode_floorset.py``) into
one Lance dataset of fixed and flat ``List`` columns. ``ds.take(row_ids)`` reads only those
rows (memory-mapped), and :meth:`LanceFloorplanStore.instance` reshapes the flat columns
back into a :class:`FloorplanInstance`. The stored ``fp_sol`` is ``(n, 4) = (w, h, x, y)``.
"""

from pathlib import Path

import lance
import numpy as np

from trinity.floorplan.data.floorset import target_positions_from_gt
from trinity.floorplan.types import FloorplanInstance, Placement

LANCE_DIRNAME = "floorset_lite.lance"


class LanceFloorplanStore:
    """Read :class:`FloorplanInstance` rows of a Lance dataset by row id."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self.ds = lance.dataset(self.path)
        self._block_counts: np.ndarray | None = None

    def __len__(self) -> int:
        return self.ds.count_rows()

    @property
    def block_counts(self) -> np.ndarray:
        """The block count ``n`` of every row (read once, cached)."""
        if self._block_counts is None:
            column = self.ds.to_table(columns=["n"]).column("n").to_numpy()
            self._block_counts = column.astype(np.int64)
        return self._block_counts

    def _row_to_instance(self, row: dict) -> FloorplanInstance:
        n = int(row["n"])
        area = np.asarray(row["area"], dtype=np.float32)
        constraints = np.asarray(row["constraints"], dtype=np.float32).reshape(n, 5)
        fp_sol = np.asarray(row["fp_sol"], dtype=np.float32).reshape(n, 4)
        b2b = np.asarray(row["b2b"], dtype=np.float32).reshape(int(row["n_b2b"]), 3)
        p2b = np.asarray(row["p2b"], dtype=np.float32).reshape(int(row["n_p2b"]), 3)
        pins = np.asarray(row["pins"], dtype=np.float32).reshape(int(row["n_pins"]), 2)
        metrics = np.asarray(row["metrics"], dtype=np.float32)

        # (w, h, x, y) -> (x, y, w, h)
        gt_positions = np.stack(
            [fp_sol[:, 2], fp_sol[:, 3], fp_sol[:, 0], fp_sol[:, 1]], axis=1
        ).astype(np.float32)

        return FloorplanInstance(
            block_count=n,
            area_targets=area,
            constraints=constraints,
            b2b=b2b,
            p2b=p2b,
            pins_pos=pins,
            target_positions=target_positions_from_gt(constraints, gt_positions),
            gt_positions=gt_positions,
            metrics=metrics,
        )

    def instance(self, idx: int) -> FloorplanInstance:
        """The case at row ``idx``."""
        row = self.ds.take([idx]).to_pylist()[0]
        return self._row_to_instance(row)

    def instances(self, indices: list[int]) -> list[FloorplanInstance]:
        """The cases at rows ``indices``, in order."""
        rows = self.ds.take(indices).to_pylist()
        return [self._row_to_instance(r) for r in rows]

    def ground_truth_placement(self, idx: int) -> Placement:
        """The ground-truth layout of row ``idx``."""
        inst = self.instance(idx)
        return Placement(xywh=inst.gt_positions.copy(), instance=inst)
