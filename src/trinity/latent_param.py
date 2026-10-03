"""Latent parameterizations: a bijection between ``z_s`` and what the network predicts.

Everything outside the network (dataset, losses, refiner, legalizer, scoring, anchors)
works in ``z_s = (cx/s, cy/s, rho)``. A ``LatentParam`` maps ``z_s`` to the network's
latent (``to_latent``) and back (``from_latent``) at the network boundary.

* ``s_only`` -- identity: the network predicts ``z_s``.
"""

from trinity.registry import LATENT_PARAM


class LatentParam:
    """Bijection ``z_s <-> z_latent``."""

    def to_latent(self, z_s):
        raise NotImplementedError

    def from_latent(self, z):
        raise NotImplementedError


@LATENT_PARAM.register("s_only")
class IdentityLatentParam(LatentParam):
    """Identity: the network predicts ``z_s``."""

    def to_latent(self, z_s):
        return z_s

    def from_latent(self, z):
        return z
