"""Graph positional encodings: per-block features computed from the netlist alone.

A graph PE maps the dense b2b weight matrix to a per-block ``(n, dim)`` feature that is
concatenated onto the block features at the input projection. Each variant is a
``GRAPH_PE`` component with two entry points computing the same encoding:

* ``__call__(A (n, n))`` -- numpy, one case -> ``(n, dim)`` float32;
* ``batched(A (B, N, N), key_pad_mask (B, N))`` -- torch, a padded batch -> ``(B,
  N, dim)``; padded nodes are isolated in the Laplacian and their rows zeroed.

Variants:

* ``none`` -- an empty ``(n, 0)`` feature.
* ``rwpe`` -- random-walk return probabilities ``[(M^t)_ii]``, ``t = 1..k``, with the
  weighted transition ``M = A D^{-1}``.
* ``spectral_draw`` -- the spectral graph drawing (Hall, 1970): the eigenvectors ``u_2
  .. u_{k+1}`` of the generalized Laplacian problem ``L u = lambda D u``, each scaled to
  unit norm with its sign fixed by its skewness. ``normalize="log1p"`` transforms the
  edge weights first.
"""

import numpy as np
import torch

from trinity.registry import GRAPH_PE

_EPS = 1e-8
# Laplacian diagonal of a padded node (above the [0, 2] spectrum of L_sym).
_PAD_EIGENVALUE = 3.0


@GRAPH_PE.register("none")
class NoGraphPE:
    """No positional encoding: an empty ``(n, 0)`` feature."""

    dim = 0

    def __call__(self, adjacency: np.ndarray) -> np.ndarray:
        return np.zeros((adjacency.shape[0], 0), dtype=np.float32)

    def batched(
        self, adj_raw: torch.Tensor, key_pad_mask: torch.Tensor
    ) -> torch.Tensor:
        return adj_raw.new_zeros((adj_raw.shape[0], adj_raw.shape[1], 0))


@GRAPH_PE.register("rwpe")
class RandomWalkPE:
    """Weighted random-walk return-probability PE: ``[(M^t)_ii]`` for ``t=1..k``."""

    def __init__(self, k: int = 16) -> None:
        self.k = k
        self.dim = k

    def __call__(self, adjacency: np.ndarray) -> np.ndarray:
        n = adjacency.shape[0]
        a = adjacency.astype(np.float64)
        deg = a.sum(axis=1)
        # Transition M = A D^{-1}; an isolated node has a zero column.
        nz = deg > _EPS
        inv_deg = np.zeros_like(deg)
        inv_deg[nz] = 1.0 / deg[nz]
        m = a * inv_deg[None, :]
        out = np.zeros((n, self.k), dtype=np.float64)
        mk = np.eye(n, dtype=np.float64)
        for t in range(self.k):
            mk = mk @ m
            out[:, t] = np.diag(mk)
        return out.astype(np.float32)

    def batched(
        self, adj_raw: torch.Tensor, key_pad_mask: torch.Tensor
    ) -> torch.Tensor:
        """The batched ``__call__``: ``(B, N, N)`` -> ``(B, N, k)``."""
        b, n, _ = adj_raw.shape
        m = key_pad_mask.to(torch.float64)
        a = _masked_adj(adj_raw, m)
        deg = a.sum(dim=2)
        inv_deg = torch.where(deg > _EPS, 1.0 / deg, torch.zeros_like(deg))
        trans = a * inv_deg[:, None, :]
        out = adj_raw.new_zeros((b, n, self.k), dtype=torch.float64)
        mk = (
            torch.eye(n, dtype=torch.float64, device=adj_raw.device)
            .expand(b, n, n)
            .clone()
        )
        for t in range(self.k):
            mk = mk @ trans
            out[:, :, t] = torch.diagonal(mk, dim1=1, dim2=2)
        return (out * m[:, :, None]).to(torch.float32)


@GRAPH_PE.register("spectral_draw")
class SpectralDrawPE:
    """Spectral drawing: the generalized-Laplacian eigenvectors ``u_2 .. u_{k+1}``.

    Computed as the eigenvectors of ``L_sym = I - D^{-1/2} A D^{-1/2}`` mapped back by
    ``D^{-1/2}``; ``k = 2`` is the Fiedler pair. ``normalize`` = ``"log1p"`` transforms
    the edge weights first (``None`` = raw weights).
    """

    def __init__(self, k: int = 2, normalize: str | None = None) -> None:
        self.k = k
        self.dim = k
        self.normalize = normalize

    def __call__(self, adjacency: np.ndarray) -> np.ndarray:
        n = adjacency.shape[0]
        if n <= self.k + 1:
            return np.zeros((n, self.k), dtype=np.float32)
        a = adjacency.astype(np.float64)
        if self.normalize == "log1p":
            a = np.log1p(a)
        deg = a.sum(axis=1)
        nz = deg > _EPS
        d_inv_sqrt = np.zeros_like(deg)
        d_inv_sqrt[nz] = 1.0 / np.sqrt(deg[nz])
        l_sym = np.eye(n) - (d_inv_sqrt[:, None] * a * d_inv_sqrt[None, :])
        l_sym = 0.5 * (l_sym + l_sym.T)
        _, evecs = np.linalg.eigh(l_sym)
        u = evecs[:, 1 : self.k + 1] * d_inv_sqrt[:, None]
        return self._canonicalize(u).astype(np.float32)

    @staticmethod
    def _canonicalize(u: np.ndarray) -> np.ndarray:
        """Each column scaled to unit norm and
        signed so its third moment is non-negative.

        A column with ~0 skew is signed by its largest-magnitude entry.
        """
        out = np.empty_like(u)
        for c in range(u.shape[1]):
            v = u[:, c]
            norm = np.linalg.norm(v)
            v = v / norm if norm > _EPS else v
            skew = np.sum(v**3)
            if abs(skew) > _EPS:
                sign = 1.0 if skew >= 0 else -1.0
            else:
                sign = 1.0 if v[np.argmax(np.abs(v))] >= 0 else -1.0
            out[:, c] = v * sign
        return out

    def batched(
        self, adj_raw: torch.Tensor, key_pad_mask: torch.Tensor
    ) -> torch.Tensor:
        """The batched ``__call__``: ``(B, N, N)`` -> ``(B, N, k)``.

        A padded node gets the Laplacian diagonal ``_PAD_EIGENVALUE``, so its
        eigenvector sorts after the real ones; padded rows and cases with ``n <= k + 1``
        are zeroed.
        """
        b, n, _ = adj_raw.shape
        m = key_pad_mask.to(torch.float64)
        a = _masked_adj(adj_raw, m)
        if self.normalize == "log1p":
            a = torch.log1p(a) * m[:, :, None] * m[:, None, :]
        deg = a.sum(dim=2)
        d_inv_sqrt = torch.where(deg > _EPS, deg.rsqrt(), torch.zeros_like(deg))
        diag = torch.where(m > 0.5, 1.0, _PAD_EIGENVALUE).to(torch.float64)
        l_sym = (
            torch.diag_embed(diag) - d_inv_sqrt[:, :, None] * a * d_inv_sqrt[:, None, :]
        )
        l_sym = 0.5 * (l_sym + l_sym.transpose(1, 2))
        evecs = _eigh_vectors(l_sym)
        u = evecs[:, :, 1 : self.k + 1] * d_inv_sqrt[:, :, None]
        u = u * m[:, :, None]
        u = _canonicalize_batched(u)
        u = u * m[:, :, None]
        small = (key_pad_mask.sum(dim=1) <= self.k + 1)[:, None, None]
        u = torch.where(small, torch.zeros_like(u), u)
        return u.to(torch.float32)


_JITTERS = (1e-10, 1e-8, 1e-6)


def _eigh_vectors(l_sym: torch.Tensor) -> torch.Tensor:
    """Eigenvectors of a batch of symmetric matrices (ascending eigenvalues).

    When the batched solver fails, each matrix is solved alone with a growing diagonal
    jitter, then on the CPU.
    """
    try:
        return torch.linalg.eigh(l_sym)[1]
    except torch._C._LinAlgError:
        pass
    out = torch.empty_like(l_sym)
    eye = torch.eye(l_sym.shape[-1], dtype=l_sym.dtype, device=l_sym.device)
    for i in range(l_sym.shape[0]):
        out[i] = _eigh_single(l_sym[i], eye)
    return out


def _eigh_single(m: torch.Tensor, eye: torch.Tensor) -> torch.Tensor:
    """Eigenvectors of one symmetric matrix,
    retried with ``_JITTERS``, then on the CPU."""
    for jitter in (0.0, *_JITTERS):
        try:
            return torch.linalg.eigh(m + jitter * eye)[1]
        except torch._C._LinAlgError:
            continue
    return torch.linalg.eigh(m.cpu())[1].to(m.device)


def _masked_adj(adj_raw: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """``adj_raw (B, N, N)`` in float64 with padded
    rows / columns zeroed (float ``mask``)."""
    a = adj_raw.to(torch.float64)
    return a * mask[:, :, None] * mask[:, None, :]


def _canonicalize_batched(u: torch.Tensor) -> torch.Tensor:
    """The batched :meth:`SpectralDrawPE._canonicalize` over ``(B, N, k)``."""
    norm = torch.linalg.vector_norm(u, dim=1, keepdim=True)
    v = torch.where(norm > _EPS, u / norm, u)
    skew = (v**3).sum(dim=1)
    picked = torch.gather(v, 1, v.abs().argmax(dim=1, keepdim=True)).squeeze(1)
    sign = torch.where(
        skew.abs() > _EPS,
        torch.where(skew >= 0, 1.0, -1.0),
        torch.where(picked >= 0, 1.0, -1.0),
    )
    return v * sign[:, None, :]
