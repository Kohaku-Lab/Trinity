"""Distribution-level comparisons between two descriptor sets: Fréchet distance and kernel MMD.

* :func:`frechet` -- ``||mu_a - mu_b||^2 + Tr(S_a + S_b - 2 (S_a S_b)^{1/2})`` on Gaussian fits,
  returned as ``(total, mean_term, cov_term)``.
* :func:`mmd2` -- the unbiased RBF-kernel MMD² estimator; :func:`median_bandwidth` gives the
  median pairwise distance of a reference set.
* :func:`mmd_permutation_test` -- p-value for ``MMD²(A, ref) < MMD²(B, ref)`` under label
  permutation of A and B.
"""

import numpy as np
from scipy import linalg

FRECHET_JITTER = 1e-6


def frechet(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Fréchet distance between the Gaussian fits of rows ``a`` and rows ``b``.

    ``FRECHET_JITTER * I`` is added to both covariances; returns ``(total, mean, cov)``.
    """
    mu_a, mu_b = a.mean(0), b.mean(0)
    eye = FRECHET_JITTER * np.eye(a.shape[1])
    s_a, s_b = np.cov(a, rowvar=False) + eye, np.cov(b, rowvar=False) + eye
    mean_term = float(((mu_a - mu_b) ** 2).sum())
    covmean = np.asarray(linalg.sqrtm(s_a @ s_b)).real
    cov_term = float(np.trace(s_a) + np.trace(s_b) - 2 * np.trace(covmean))
    return mean_term + cov_term, mean_term, cov_term


def _sqdist(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xx = (x**2).sum(1)[:, None]
    yy = (y**2).sum(1)[None, :]
    return np.maximum(xx + yy - 2 * x @ y.T, 0.0)


def median_bandwidth(ref: np.ndarray, max_n: int = 4000, seed: int = 0) -> float:
    """Median pairwise Euclidean distance of (a subsample of) ``ref``."""
    rng = np.random.default_rng(seed)
    r = ref[rng.choice(ref.shape[0], min(max_n, ref.shape[0]), replace=False)]
    d = np.sqrt(_sqdist(r, r))
    return float(np.median(d[np.triu_indices(r.shape[0], 1)]))


def _kernel_sums(x: np.ndarray, y: np.ndarray, bw: float) -> tuple[float, float, float]:
    g = 1.0 / (2 * bw * bw)
    kxx = np.exp(-g * _sqdist(x, x))
    kyy = np.exp(-g * _sqdist(y, y))
    kxy = np.exp(-g * _sqdist(x, y))
    n, m = x.shape[0], y.shape[0]
    sxx = (kxx.sum() - np.trace(kxx)) / (n * (n - 1))
    syy = (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
    sxy = kxy.mean()
    return sxx, syy, sxy


def mmd2(x: np.ndarray, y: np.ndarray, bw: float) -> float:
    """Unbiased RBF MMD² between rows ``x`` and rows ``y`` at bandwidth ``bw``."""
    sxx, syy, sxy = _kernel_sums(x, y, bw)
    return float(sxx + syy - 2 * sxy)


def mmd_permutation_test(
    a: np.ndarray,
    b: np.ndarray,
    ref: np.ndarray,
    bw: float,
    n_perm: int = 1000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """``(mmd2_a, mmd2_b, p)`` of ``a`` and ``b`` against ``ref``.

    ``p`` is the fraction of label permutations of ``a ∪ b`` whose
    ``MMD²(A', ref) - MMD²(B', ref)`` is at most the observed difference.
    """
    rng = np.random.default_rng(seed)
    ma, mb = mmd2(a, ref, bw), mmd2(b, ref, bw)
    observed = ma - mb
    pool = np.concatenate([a, b])
    n = a.shape[0]
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(pool.shape[0])
        diff = mmd2(pool[perm[:n]], ref, bw) - mmd2(pool[perm[n:]], ref, bw)
        count += diff <= observed
    return ma, mb, (count + 1) / (n_perm + 1)
