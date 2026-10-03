"""Backbones of published learned placers, ported behind ``BASELINE_BACKBONE``.

Each module copies the defining parts of a published placer's network with the smallest changes
needed to read the Trinity conditioning (a padded ``(B, N)`` batch with a per-row adjacency
instead of one shared PyG graph). Provenance is recorded per file.

* :mod:`flowplace_attgnn` -- the FlowPlace / ChipDiffusion ``AttGNN`` (``flowplace_attgnn``).
* :mod:`diffplace_vgnn` -- the DiffPlace ``VectorGNNV2Global`` (``diffplace_vgnn``).
* :mod:`macrodiff_hetero` -- the MacroDiff+ ``MacroPlacer``, hetero GNN + token U-Net
  (``macrodiff_hetero``).
"""

from trinity_baselines.ported import (  # noqa: F401  (register the backbones)
    diffplace_vgnn,
    flowplace_attgnn,
    macrodiff_hetero,
)
