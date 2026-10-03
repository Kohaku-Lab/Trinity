"""Output heads of the direct regressor: the prediction space and its decode to a layout.

* ``z`` (``LatentHead``) -- predicts the latent ``(cx/s, cy/s, rho)``, decoded with
  ``z_to_xywh`` (``w * h == area`` by construction).
* ``xywh`` (``BoxHead``) -- predicts the raw normalized box ``(cx/s, cy/s, w/s, h/s)``; the
  area is only encouraged by the area aux term.

A head declares its ``out_dim`` (the backbone's ``latent_dim``), builds its regression target
from the ``/s`` latent ``z_s``, decodes a prediction to a normalized ``(x, y, w, h)``, and maps a
prediction back to ``z_s`` for the refiner and legalizer.
"""

import torch

from trinity.decode import RHO_CLAMP
from trinity.decode import z_to_xywh as z_to_xywh_from_latent
from trinity_baselines.registry import HEAD


class RegressionHead:
    """Map between the backbone prediction space and normalized ``(x,y,w,h)``.

    ``out_dim`` is the backbone's output width. ``target_from_zs`` builds the regression
    target (and the anchor latent) from the canonical ``/s`` latent ``z_s = (cx/s,cy/s,rho)``.
    ``to_xywh`` decodes a ``(B,N,out_dim)`` prediction to a normalized ``(B,N,4)`` box (the
    aux-loss / refine space: scale 1, area in ``/s^2`` units).
    """

    out_dim: int = 3

    def target_from_zs(
        self, z_s: torch.Tensor, area_norm: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    def to_xywh(self, pred: torch.Tensor, area_norm: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def to_zs(self, pred: torch.Tensor, area_norm: torch.Tensor) -> torch.Tensor:
        """Map a prediction to the ``/s`` latent ``z_s = (cx/s, cy/s, rho)``."""
        raise NotImplementedError

    def anchor(
        self,
        anchor_zs: torch.Tensor,
        anchor_mask: torch.Tensor,
        area_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The anchor ``(z_s value, 3-D mask)`` in the head's ``out_dim`` channels."""
        value = self.target_from_zs(anchor_zs, area_norm)
        return value, self.expand_mask(anchor_mask)

    def expand_mask(self, anchor_mask: torch.Tensor) -> torch.Tensor:
        """Expand the canonical ``(cx, cy, rho)`` clamp mask to this head's channels."""
        raise NotImplementedError


@HEAD.register("z")
class LatentHead(RegressionHead):
    """``z``: predict the area-preserving latent ``(cx/s, cy/s, rho)`` (identity target)."""

    out_dim = 3

    def target_from_zs(
        self, z_s: torch.Tensor, area_norm: torch.Tensor
    ) -> torch.Tensor:
        return z_s

    def to_xywh(self, pred: torch.Tensor, area_norm: torch.Tensor) -> torch.Tensor:
        ones = area_norm.new_ones(area_norm.shape[0])
        return z_to_xywh_from_latent(pred, area_norm, ones)

    def to_zs(self, pred: torch.Tensor, area_norm: torch.Tensor) -> torch.Tensor:
        return pred

    def expand_mask(self, anchor_mask: torch.Tensor) -> torch.Tensor:
        return anchor_mask


@HEAD.register("xywh")
class BoxHead(RegressionHead):
    """``xywh``: predict the raw normalized box ``(cx/s, cy/s, w/s, h/s)``.

    The target takes the centers from ``z_s`` and ``w/s = sqrt(area_norm) e^{rho/2}``,
    ``h/s = sqrt(area_norm) e^{-rho/2}``; the decode is the corner shift ``x = cx - w/2``.
    """

    out_dim = 4

    def target_from_zs(
        self, z_s: torch.Tensor, area_norm: torch.Tensor
    ) -> torch.Tensor:
        rho = z_s[..., 2].clamp(-RHO_CLAMP, RHO_CLAMP)
        root_a = area_norm.clamp_min(0).sqrt()
        w = root_a * torch.exp(rho / 2)
        h = root_a * torch.exp(-rho / 2)
        return torch.stack([z_s[..., 0], z_s[..., 1], w, h], dim=-1)

    def to_xywh(self, pred: torch.Tensor, area_norm: torch.Tensor) -> torch.Tensor:
        cx, cy, w, h = pred[..., 0], pred[..., 1], pred[..., 2], pred[..., 3]
        w = w.clamp_min(0.0)
        h = h.clamp_min(0.0)
        x = cx - w / 2
        y = cy - h / 2
        return torch.stack([x, y, w, h], dim=-1)

    def to_zs(self, pred: torch.Tensor, area_norm: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        w = pred[..., 2].clamp_min(eps)
        h = pred[..., 3].clamp_min(eps)
        rho = torch.log(w / h).clamp(-RHO_CLAMP, RHO_CLAMP)
        return torch.stack([pred[..., 0], pred[..., 1], rho], dim=-1)

    def expand_mask(self, anchor_mask: torch.Tensor) -> torch.Tensor:
        # [cx, cy, rho] -> [cx, cy, w, h]: the rho channel drives both size channels.
        cx, cy, rho = anchor_mask[..., 0], anchor_mask[..., 1], anchor_mask[..., 2]
        return torch.stack([cx, cy, rho, rho], dim=-1)
