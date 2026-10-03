"""The Lightning trainer for the diffusion placer + the validation callback."""

from trinity.training.callbacks import ValidationCallback
from trinity.training.trainer import DiffusionTrainer

__all__ = ["DiffusionTrainer", "ValidationCallback"]
