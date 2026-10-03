"""Exponential moving average of a module's parameters.

An ``nn.Module`` holding the shadow weights (saved in checkpoints). :meth:`EMAModule.update`
runs once per optimizer step; :meth:`EMAModule.use_ema` swaps the shadow weights into the
live module for a block and restores the training weights on exit.
"""

import contextlib

import torch
import torch.nn as nn


class EMAModule(nn.Module):
    """Decay-averaged shadow of ``model``'s parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        super().__init__()
        self.decay = decay
        self._names = [name for name, _ in model.named_parameters()]
        self.shadow = nn.ParameterList(
            [
                nn.Parameter(p.detach().clone(), requires_grad=False)
                for p in model.parameters()
            ]
        )

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """``shadow = decay * shadow + (1 - decay) * params``."""
        shadow = list(self.shadow)
        params = [p.detach() for p in model.parameters()]
        if len(shadow) != len(params):
            raise ValueError(
                f"EMA holds {len(shadow)} tensors, the model has {len(params)}"
            )
        torch._foreach_mul_(shadow, self.decay)
        torch._foreach_add_(shadow, params, alpha=1 - self.decay)

    @contextlib.contextmanager
    def use_ema(self, model: nn.Module):
        """Swap EMA weights into ``model`` for the duration of the context."""
        backup = [p.detach().clone() for p in model.parameters()]
        with torch.no_grad():
            for p, s in zip(model.parameters(), self.shadow, strict=True):
                p.copy_(s)
        try:
            yield
        finally:
            with torch.no_grad():
                for p, b in zip(model.parameters(), backup, strict=True):
                    p.copy_(b)
