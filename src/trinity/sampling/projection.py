"""Sampler-state projections: maps ``(z, cond) -> z`` applied after every sampler step.

Registered under ``PROJECTION`` and selected by name::

    SAMPLE_PROJECTIONS = ["anchor_clamp", "mib_group_mean"]  # the sampler default
    SAMPLE_PROJECTIONS = []                                  # free sampling

* ``anchor_clamp`` -- writes the anchored coordinates (preplaced positions and shapes,
  fixed shapes, the shape of an MIB group with a known member);
* ``mib_group_mean`` -- gives every all-soft MIB group its members' mean shape.
"""

import torch

from trinity.registry import PROJECTION


@PROJECTION.register("anchor_clamp")
class AnchorClamp:
    """Overwrite anchored coordinates with their known values."""

    def __call__(self, z: torch.Tensor, cond) -> torch.Tensor:
        """Return ``z`` with ``cond.anchor_z`` substituted where ``cond.anchor_mask`` is 1."""
        if cond.anchor_mask is None or cond.anchor_z is None:
            return z
        return torch.where(cond.anchor_mask > 0.5, cond.anchor_z, z)


def project_mib_group_rho(z: torch.Tensor, group: torch.Tensor) -> torch.Tensor:
    """Replace each MIB member's shape channels with its group's mean, by scatter-add averaging.

    ``z`` is ``(B, N, latent)``; channels ``0:2`` are the position and are left untouched, channels
    ``2:`` are the shape. ``group`` is ``(B, N)`` group ids with 0 meaning "not a member". Returns a
    tensor of the same shape.
    """
    member = group > 0
    if not bool(member.any()):
        return z
    shape = z[..., 2:]
    n_groups = int(group.max().item()) + 1
    idx_c = group.clamp_min(0)[..., None].expand_as(shape)
    weight = member.to(shape.dtype)[..., None]
    sums = torch.zeros(
        shape.shape[0],
        n_groups,
        shape.shape[-1],
        dtype=shape.dtype,
        device=shape.device,
    )
    sums.scatter_add_(1, idx_c, shape * weight)
    counts = torch.zeros_like(sums).scatter_add_(1, idx_c, weight.expand_as(shape))
    mean = (sums / counts.clamp_min(1.0)).gather(1, idx_c)
    return torch.cat([z[..., :2], torch.where(member[..., None], mean, shape)], dim=-1)


@PROJECTION.register("mib_group_mean")
class MIBGroupMean:
    """Make all-soft MIB group members share one shape, at the group's own mean."""

    def __call__(self, z: torch.Tensor, cond) -> torch.Tensor:
        """Return ``z`` with the shape channels of each ``cond.mib_soft_group`` member averaged."""
        if cond.mib_soft_group is None:
            return z
        return project_mib_group_rho(z, cond.mib_soft_group)
