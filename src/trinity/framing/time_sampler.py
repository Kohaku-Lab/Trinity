"""Training timestep samplers ``t ~ p(t)``: ``uniform`` and ``logit_normal``."""

import torch

from trinity.registry import TIME_SAMPLER


@TIME_SAMPLER.register("uniform")
class UniformSampler:
    """``t ~ U(t_min, t_max)``."""

    def __init__(self, t_min: float = 0.0, t_max: float = 1.0) -> None:
        self.t_min = t_min
        self.t_max = t_max

    def __call__(self, n: int, device) -> torch.Tensor:
        return torch.rand(n, device=device) * (self.t_max - self.t_min) + self.t_min


@TIME_SAMPLER.register("logit_normal")
class LogitNormalSampler:
    """``t = sigmoid(N(mean, std))``."""

    def __init__(self, mean: float = 0.0, std: float = 1.0) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, n: int, device) -> torch.Tensor:
        return torch.sigmoid(torch.randn(n, device=device) * self.std + self.mean)
