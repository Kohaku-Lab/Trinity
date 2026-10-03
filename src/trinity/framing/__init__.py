"""Diffusion / flow framings and training timestep samplers.

Importing this package registers the framings (``ddpm_x0``, ``ddpm_eps``, ``ddpm_v``,
``ddpm_cosine_x0``, ``ddpm_cosine_eps``, ``rectified_flow``, ``gvp``) and the time samplers
(``uniform``, ``logit_normal``).
"""

from trinity.framing import (
    standard,  # noqa: F401  (register: framings)
    time_sampler,  # noqa: F401  (register: time samplers)
)
from trinity.framing.base import Coeffs, Framing
from trinity.framing.general import GeneralFraming

__all__ = ["Coeffs", "Framing", "GeneralFraming"]
