# Data

All data resolves through `trinity.floorplan.data.paths`: an explicit argument, then an
environment variable, then the project data directory `data/` (`TRINITY_DATA` moves it).
Missing evaluation sets are downloaded into `data/` on first use.

| dataset | variable | used for |
|---|---|---|
| FloorSet validation (`LiteTensorDataTest/`, 100 cases) | `TRINITY_FLOORSET` | the official 100-case evaluation, quick tests |
| FloorSet train, transcoded (`floorset_lite_mibfix.lance`) | `TRINITY_TRAIN_LANCE` | training and the 12,000-case dev split |
| GSRC (`gsrc/{HARD,SOFT}/`) | `TRINITY_GSRC` | the bookshelf protocol |
| MCNC (`mcnc/{HARD,SOFT}/`) | `TRINITY_MCNC` | the bookshelf protocol |

## The training set

The 1M-layout FloorSet-Lite training set ships as 9,000 `.th` files. Training reads one Lance
dataset built from them once:

```python
from trinity.floorplan.data import download_train_raw

download_train_raw()  # data/floorset_lite/ (~6.6 GB)
```

```bash
kogine run scripts/data/transcode_floorset.py   # -> data/floorset_lite_mibfix.lance
```

The transcode stores every field of a layout (areas, the five constraint columns, the
ground-truth boxes, the netlist, the pins, the metrics); the latent and the conditioning are
derived at load time.

**MIB groups.** With `MIB_FIX = True` (the default, and the set every model of the paper was
trained on) the MIB column is rebuilt from the ground-truth shapes: the MIB group becomes
every block whose ground-truth `(w, h)` is the most frequent shape of the layout (ties: the
lexicographically smallest), so the members of a group share one shape in the target.
`MIB_FIX = False` keeps the column as shipped; `FIX_EXISTING` applies the rebuild to an
already transcoded set.

## Splits

`trinity.data.splits` holds out a dev split from the train set: 100 layouts per block count
(21–120, 10,000) plus 2,000 uniformly random layouts, 12,000 in all, with seed `20090220`.
The split is written next to the Lance set (`splits.json`) on first use and read back after,
so training and every evaluation see the same 12,000 cases.

## Instances

Every loader returns a `trinity.floorplan.types.FloorplanInstance`: block areas, the
constraint columns (fixed, preplaced, MIB, cluster, boundary), the netlist (block-to-block
and pin-to-block edges), the pins, the ground-truth placement when present, and an optional
fixed outline (bookshelf protocol). A `Placement` pairs an instance with boxes
`(x, y, w, h)`.
