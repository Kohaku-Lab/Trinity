"""Diffusion / flow regression losses: a weighted distance between ``pred`` and ``target``.

Each distance (L2, L1, Huber, Charbonnier, pseudo-Huber) is a registered ``LossTerm``;
``denoise`` is the masked L2. They share, through :class:`_RegressionLoss`, the per-sample
weight ``w(t)``, the token mask (padding excluded), an optional anchor down-weight
(``anchor_downweight``, default 0) and an optional cosine term.
"""

import torch
import torch.nn.functional as F

from trinity.losses.base import LossContext, LossTerm
from trinity.registry import LOSS


def _bcast_t(wt: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Reshape a per-sample ``(B,)`` weight to broadcast against ``ref`` ``(B, ...)``."""
    return wt.reshape(wt.shape[0], *([1] * (ref.ndim - 1)))


class _RegressionLoss(LossTerm):
    """Weighted ``distance(pred, target)`` over the masked coordinates, plus optional cosine.

    Subclasses set ``name`` and implement ``_distance(pred, target)`` (per element).
    ``reduction``:

    * ``"per_sample"`` -- each sample normalized by its own coordinate weight, weighted by
      ``w(t)``, then averaged over the batch;
    * ``"global"`` -- one ``sum(err * w) / sum(w)`` over the batch, ``w(t)`` folded in.

    ``cosine_weight > 0`` adds ``1 - cos(pred, target)`` weighted by ``w(t)``.
    """

    def __init__(
        self,
        weight: float = 1.0,
        cosine_weight: float = 0.0,
        anchor_downweight: float = 0.0,
        reduction: str = "per_sample",
    ) -> None:
        self.weight = weight
        self.cosine_weight = cosine_weight
        self.anchor_downweight = anchor_downweight
        self.reduction = reduction

    def _distance(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _coord_weight(self, ctx: LossContext, like: torch.Tensor) -> torch.Tensor:
        """Per-coordinate weight: 1, minus ``anchor_downweight`` on anchors, 0 on padding."""
        w = torch.ones_like(like)
        if ctx.anchor_mask is not None:
            w = w - self.anchor_downweight * ctx.anchor_mask
        if ctx.token_mask is not None:
            w = w * ctx.token_mask.unsqueeze(-1)
        return w

    def _reduce(
        self, err: torch.Tensor, cw: torch.Tensor, wt: torch.Tensor
    ) -> torch.Tensor:
        if self.reduction == "global":
            wcw = cw * _bcast_t(wt, cw)
            return (err * wcw).sum() / wcw.sum().clamp_min(1.0)
        per_sample = (err * cw).flatten(1).sum(dim=1) / cw.flatten(1).sum(
            dim=1
        ).clamp_min(1.0)
        return (per_sample * wt).mean()

    def __call__(self, ctx: LossContext) -> tuple[torch.Tensor, dict]:
        err = self._distance(ctx.pred, ctx.target)
        cw = self._coord_weight(ctx, err)
        main = self._reduce(err, cw, ctx.weight)
        loss = self.weight * main
        logs = {self.name: main.detach()}
        if self.cosine_weight > 0:
            cos = 1.0 - F.cosine_similarity(
                ctx.pred.flatten(1), ctx.target.flatten(1), dim=1
            )
            cosine = (cos * ctx.weight).mean()
            loss = loss + self.cosine_weight * cosine
            logs["cosine"] = cosine.detach()
        return loss, logs


@LOSS.register("l2")
@LOSS.register("denoise")
class L2Loss(_RegressionLoss):
    """Squared error (registered as ``l2`` and ``denoise``)."""

    name = "l2"

    def _distance(self, pred, target):
        return F.mse_loss(pred, target, reduction="none")


@LOSS.register("l1")
class L1Loss(_RegressionLoss):
    """Absolute error."""

    name = "l1"

    def _distance(self, pred, target):
        return F.l1_loss(pred, target, reduction="none")


@LOSS.register("huber")
class HuberLoss(_RegressionLoss):
    """Huber: L2 within ``delta`` of zero, L1 beyond."""

    name = "huber"

    def __init__(
        self, weight=1.0, cosine_weight=0.0, anchor_downweight=0.0, delta: float = 1.0
    ) -> None:
        super().__init__(weight, cosine_weight, anchor_downweight)
        self.delta = delta

    def _distance(self, pred, target):
        return F.huber_loss(pred, target, reduction="none", delta=self.delta)


@LOSS.register("charbonnier")
class CharbonnierLoss(_RegressionLoss):
    """Charbonnier: ``sqrt((pred - target)^2 + eps^2)``."""

    name = "charbonnier"

    def __init__(
        self, weight=1.0, cosine_weight=0.0, anchor_downweight=0.0, eps: float = 1e-3
    ) -> None:
        super().__init__(weight, cosine_weight, anchor_downweight)
        self.eps = eps

    def _distance(self, pred, target):
        return torch.sqrt((pred - target).pow(2) + self.eps**2)


@LOSS.register("pseudo_huber")
class PseudoHuberLoss(_RegressionLoss):
    """Pseudo-Huber: ``c * (sqrt((d / c)^2 + 1) - 1)``."""

    name = "pseudo_huber"

    def __init__(
        self, weight=1.0, cosine_weight=0.0, anchor_downweight=0.0, c: float = 1.0
    ) -> None:
        super().__init__(weight, cosine_weight, anchor_downweight)
        self.c = c

    def _distance(self, pred, target):
        d = pred - target
        return self.c * (torch.sqrt((d / self.c).pow(2) + 1.0) - 1.0)
