"""Total score over a set of cases (lower is better), under three weightings."""

import numpy as np


def total_exp(costs: list[float], block_counts: list[int]) -> float:
    """``sum(cost * exp(n/12)) / sum(exp(n/12))``."""
    weights = np.exp(np.asarray(block_counts, dtype=np.float64) / 12.0)
    return float(np.sum(np.asarray(costs) * weights) / np.sum(weights))


def total_blockcount(costs: list[float], block_counts: list[int]) -> float:
    """``sum(cost * n) / sum(n)``."""
    n = np.asarray(block_counts, dtype=np.float64)
    if n.sum() == 0:
        return float(np.mean(costs))
    return float(np.sum(np.asarray(costs) * n) / n.sum())


def total_mean(costs: list[float], block_counts: list[int] | None = None) -> float:
    """Unweighted mean cost (``block_counts`` accepted for a uniform call signature)."""
    return float(np.mean(costs)) if costs else 0.0
