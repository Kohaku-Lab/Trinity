"""The regression placer: forward -> refine -> legalize -> select (no sampler).

The deterministic counterpart of :class:`trinity.solver.DiffusionPlacer`: the conditioning,
refiner, decoding, legalization and selection are inherited; only ``sample_latents`` is
replaced by one regressor forward. A regressor yields one layout per case, so ``samples = 1``.
The head maps the prediction to the ``z_s`` latent the refiner and legalizer read.
"""

import torch

from trinity.registry import PROJECTION, build
from trinity.solver import DiffusionPlacer


class RegressionPlacer(DiffusionPlacer):
    """A trained :class:`~trinity_baselines.models.DirectRegressor` + refiner + legalizer.

    Constructed like ``DiffusionPlacer`` without a ``sampler``. ``head`` maps the network
    output to ``z_s``; ``projections`` is a list of ``PROJECTION`` specs applied to the
    prediction (none by default).
    """

    def __init__(
        self, regressor, refiner=None, *, head, projections=None, **kwargs
    ) -> None:
        super().__init__(regressor, sampler=None, refiner=refiner, **kwargs)
        self.head = head
        self.projections = [build(spec, PROJECTION) for spec in (projections or [])]

    def predict_latents(self, ctx: dict) -> torch.Tensor:
        """One regressor forward mapped to the ``z_s`` latent, projections applied."""
        z = self.head.to_zs(self.backbone(ctx["cond"]), ctx["area"])
        for projection in self.projections:
            z = projection(z, ctx["cond"])
        return z

    @torch.no_grad()
    def sample_latents(self, ctx: dict, noise=None) -> torch.Tensor:
        """The regressor's prediction (``noise`` is ignored)."""
        return self.predict_latents(ctx)

    @torch.no_grad()
    def predict_group(self, group):
        """Forward only (no refiner): normalized ``(rows, max_n, 4)`` boxes, area in ``/s^2``."""
        ctx = self.build_group_ctx(group)
        z = self.predict_latents(ctx)
        target = self.head.target_from_zs(z, ctx["area"])
        return self.head.to_xywh(target, ctx["area"]).cpu().numpy()
