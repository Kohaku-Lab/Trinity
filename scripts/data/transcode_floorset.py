"""Transcode the raw FloorSet-Lite ``.th`` files into one Lance dataset.

The 1M-layout train set ships as ``floorset_lite/worker_*/layouts_*.th`` files, each a
7-column list of 112 layouts of one block count. This script writes them into a single Lance
dataset (``data/floorset_lite.lance``): fixed fields as scalars or fixed-size lists, the
ragged per-block / per-edge / per-pin arrays as flat ``list<float>`` columns plus their
counts. The boxes ``fp_sol = [w, h, x, y]`` are stored as-is; the latent and the
conditioning are derived at load time.

With ``MIB_FIX`` (the default, and the set every paper model trained on) the MIB column is
rebuilt from the ground-truth shapes: the MIB group becomes every block whose ground-truth
``(w, h)`` equals the most frequent ``(w, h)`` of the layout (ties: the lexicographically
smallest), so all members of the group share one shape. ``FIX_EXISTING`` applies the same
rebuild to an already transcoded Lance set instead of the raw files.

Download the raw tree first (``trinity.floorplan.data.download_train_raw``), then::

    kogine run scripts/data/transcode_floorset.py
    kogine run scripts/data/transcode_floorset.py --set MIB_FIX=False --set OUT=data/raw.lance
    kogine run scripts/data/transcode_floorset.py --set FIX_EXISTING=data/floorset_lite.lance
"""

import glob
import os
import shutil
import time
from collections import Counter

import lance
import numpy as np
import pyarrow as pa
import torch

# Directory holding ``worker_*/layouts_*.th``.
SRC: str = "data/floorset_lite"
OUT: str = "data/floorset_lite_mibfix.lance"
# Rebuild the MIB groups from the ground-truth shapes.
MIB_FIX: bool = True
# An existing Lance set to rewrite with the MIB fix into OUT (None: transcode SRC).
FIX_EXISTING: str | None = None
# Raw files per Lance write batch.
FLUSH_FILES: int = 200
# Fragments of the final dataset.
SHARDS: int = 5
# Only re-shard an existing ``OUT``.
COMPACT_ONLY: bool = False

LAYOUTS_PER_FILE = 112
GB = 1 << 30
# Column of the MIB id inside the five constraint columns.
MIB_COLUMN = 2

SCHEMA = pa.schema(
    [
        ("n", pa.int32()),
        ("n_pins", pa.int32()),
        ("n_b2b", pa.int32()),
        ("n_p2b", pa.int32()),
        ("area", pa.list_(pa.float32())),  # [n]
        ("constraints", pa.list_(pa.float32())),  # [n * 5] -> (n, 5)
        ("fp_sol", pa.list_(pa.float32())),  # [n * 4] -> (n, 4) = (w, h, x, y)
        ("b2b", pa.list_(pa.float32())),  # [n_b2b * 3]
        ("p2b", pa.list_(pa.float32())),  # [n_p2b * 3]
        ("pins", pa.list_(pa.float32())),  # [n_pins * 2]
        ("metrics", pa.list_(pa.float32(), 8)),
    ]
)


def _flat(tensor) -> list[float]:
    return tensor.contiguous().view(-1).tolist()


def fix_mib(constraints: list[float], fp_sol: list[float], n: int) -> list[float]:
    """The flat ``(n, 5)`` constraints with the MIB column rebuilt from the shapes.

    The new MIB group (id 1) is every block whose ``(w, h)`` in ``fp_sol`` equals the most
    frequent shape (ties: the lexicographically smallest); no group when no shape repeats.
    """
    table = np.asarray(constraints, dtype=np.float32).reshape(n, 5).copy()
    shapes = [
        tuple(box) for box in np.asarray(fp_sol, dtype=np.float32).reshape(n, 4)[:, :2]
    ]
    counts = Counter(shapes)
    top = max(counts.values())
    table[:, MIB_COLUMN] = 0.0
    if top > 1:
        shape = min(s for s, count in counts.items() if count == top)
        members = [i for i, s in enumerate(shapes) if s == shape]
        table[members, MIB_COLUMN] = 1.0
    return table.reshape(-1).tolist()


def _layout_row(columns, i: int) -> dict:
    """One Lance row from layout ``i`` of a raw file's columns."""
    raw = columns[0][i]  # (n, 6): area, then the five constraint columns
    n = int((raw[:, 0] != -1).sum())
    raw = raw[:n]
    b2b, p2b, pins = columns[1][i], columns[2][i], columns[3][i]
    row = {
        "n": n,
        "n_pins": int(pins.shape[0]),
        "n_b2b": int(b2b.shape[0]),
        "n_p2b": int(p2b.shape[0]),
        "area": _flat(raw[:, 0]),
        "constraints": _flat(raw[:, 1:6]),
        "fp_sol": _flat(columns[5][i][:n]),
        "b2b": _flat(b2b),
        "p2b": _flat(p2b),
        "pins": _flat(pins),
        "metrics": columns[6][i].view(-1)[:8].tolist(),
    }
    if MIB_FIX:
        row["constraints"] = fix_mib(row["constraints"], row["fp_sol"], n)
    return row


def _file_rows(path: str) -> list[dict]:
    columns = torch.load(path, weights_only=False)
    return [_layout_row(columns, i) for i in range(len(columns[0]))]


def _fixed_batches(source: str):
    """The record batches of the Lance set ``source`` with the MIB column rebuilt."""
    for batch in lance.dataset(source).to_batches(batch_size=8192):
        rows = batch.to_pylist()
        for row in rows:
            row["constraints"] = fix_mib(row["constraints"], row["fp_sol"], row["n"])
        yield pa.RecordBatch.from_pylist(rows, schema=SCHEMA)


def fix_existing(source: str, out: str) -> None:
    """Write the Lance set ``source`` to ``out`` with the MIB column rebuilt."""
    if os.path.exists(out):
        raise SystemExit(f"{out} exists; remove it to write it again")
    print(f"rebuilding the MIB groups of {source} -> {out}", flush=True)
    reader = pa.RecordBatchReader.from_batches(SCHEMA, _fixed_batches(source))
    lance.write_dataset(reader, out, mode="create")


def compact(out: str, shards: int, max_bytes_per_file: int = 5 * GB) -> None:
    """Rewrite the dataset at ``out`` into ``shards`` balanced fragments, streaming."""
    dataset = lance.dataset(out)
    rows = dataset.count_rows()
    target_rows = max(1, rows // shards)
    tmp = out + ".rewriting"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    print(
        f"compacting {out}: {len(dataset.get_fragments())} fragments -> {shards}",
        flush=True,
    )
    reader = pa.RecordBatchReader.from_batches(
        dataset.schema, dataset.to_batches(batch_size=8192)
    )
    lance.write_dataset(
        reader,
        tmp,
        mode="create",
        max_rows_per_file=target_rows,
        max_bytes_per_file=max_bytes_per_file,
    )
    if lance.dataset(tmp).count_rows() != rows:
        raise RuntimeError("row count changed while re-sharding")
    shutil.rmtree(out)
    os.rename(tmp, out)
    final = lance.dataset(out)
    print(f"sharded: {len(final.get_fragments())} fragments, {final.count_rows()} rows")


def main() -> None:
    if COMPACT_ONLY:
        compact(OUT, SHARDS)
        return
    if FIX_EXISTING is not None:
        fix_existing(FIX_EXISTING, OUT)
        compact(OUT, SHARDS)
        return

    files = sorted(glob.glob(os.path.join(SRC, "worker_*", "layouts_*.th")))
    if not files:
        raise SystemExit(f"no layouts_*.th under {SRC!r}")
    if os.path.exists(OUT):
        raise SystemExit(f"{OUT} exists; remove it to transcode again")
    print(
        f"transcoding {len(files)} files (~{len(files) * LAYOUTS_PER_FILE} layouts) -> {OUT}"
    )

    start = time.perf_counter()
    written = 0
    buffer: list[dict] = []
    mode = "create"
    for index, path in enumerate(files, start=1):
        buffer.extend(_file_rows(path))
        if index % FLUSH_FILES == 0 or index == len(files):
            lance.write_dataset(
                pa.Table.from_pylist(buffer, schema=SCHEMA), OUT, mode=mode
            )
            mode = "append"
            written += len(buffer)
            buffer = []
            elapsed = time.perf_counter() - start
            print(
                f"  {index:5d}/{len(files)} files  {written:7d} rows  {elapsed:6.1f}s",
                flush=True,
            )

    print(f"written {written} rows in {time.perf_counter() - start:.1f}s")
    compact(OUT, SHARDS)


if __name__ == "__main__":
    main()
