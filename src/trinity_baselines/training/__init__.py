"""The direct-regressor trainer, and the validation
callback it shares with the diffusion trainer."""

from trinity.training import ValidationCallback
from trinity_baselines.training.trainer import RegressorTrainer

__all__ = ["RegressorTrainer", "ValidationCallback"]
