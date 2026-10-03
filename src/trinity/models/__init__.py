"""The denoiser backbone, its block, its components and the architecture presets.

Importing this package registers the components
(norm, mlp, attention, time conditioning, graph mix).
"""

from trinity.models import components  # noqa: F401  (register: model components)
from trinity.models.backbone import DenoiserCond, SetTransformerDenoiser
from trinity.models.presets import PRESETS, DenoiserArchConfig, get_preset

__all__ = [
    "DenoiserCond",
    "SetTransformerDenoiser",
    "DenoiserArchConfig",
    "PRESETS",
    "get_preset",
]
