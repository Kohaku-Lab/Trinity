"""The direct regressor and its output heads (registers the ``z`` / ``xywh`` heads)."""

from trinity_baselines.models import head  # noqa: F401  (register: z / xywh heads)
from trinity_baselines.models.head import BoxHead, LatentHead, RegressionHead
from trinity_baselines.models.regressor import DirectRegressor

__all__ = [
    "DirectRegressor",
    "RegressionHead",
    "LatentHead",
    "BoxHead",
]
