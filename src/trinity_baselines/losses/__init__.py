"""Losses of the direct regressor: the ``trinity.registry.LOSS`` terms, re-exported.

The regressor selects its regression term (``denoise`` / ``l2`` / ``l1`` / ``huber`` /
``charbonnier`` / ``pseudo_huber``) and the ``ref_*`` aux terms by name, as a diffusion config
does. The trainer fills the shared :class:`~trinity.losses.base.LossContext` with ``pred`` = the
predicted layout, ``target`` = the ground-truth layout, a uniform weight and ``t = 0``.
"""

from trinity.losses import LossContext, LossTerm
from trinity.registry import LOSS

__all__ = ["LossContext", "LossTerm", "LOSS"]
