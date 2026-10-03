"""Area-preserving latent parameterization of a layout.

Each block maps to ``z = (cx/s, cy/s, rho)`` where ``(cx,cy)`` is its center, ``s`` is
the layout scale, and ``rho = log(w/h)`` is the log aspect ratio. Decoding uses the
target area ``a``: ``w = sqrt(a) e^{rho/2}``, ``h = sqrt(a) e^{-rho/2}``, so ``w * h ==
a`` exactly. ``rho`` is clamped to ``[-RHO_CLAMP, RHO_CLAMP]``.
"""

import numpy as np

RHO_CLAMP = 4.0


def xywh_to_z(xywh: np.ndarray, area_targets: np.ndarray, s: float) -> np.ndarray:
    """``[n,4]`` ``(x,y,w,h)`` -> ``[n,3]`` ``(cx/s, cy/s, rho)``."""
    cx = xywh[:, 0] + xywh[:, 2] / 2.0
    cy = xywh[:, 1] + xywh[:, 3] / 2.0
    rho = np.clip(np.log(xywh[:, 2] / xywh[:, 3]), -RHO_CLAMP, RHO_CLAMP)
    return np.stack([cx / s, cy / s, rho], axis=1)


def z_to_xywh(z: np.ndarray, area_targets: np.ndarray, s: float) -> np.ndarray:
    """``[n,3]`` ``(cx/s, cy/s, rho)`` -> ``[n,4]``
    ``(x,y,w,h)`` with ``w*h == area``."""
    rho = np.clip(z[:, 2], -RHO_CLAMP, RHO_CLAMP)
    root_a = np.sqrt(area_targets)
    w = root_a * np.exp(rho / 2.0)
    h = root_a * np.exp(-rho / 2.0)
    cx = z[:, 0] * s
    cy = z[:, 1] * s
    x = cx - w / 2.0
    y = cy - h / 2.0
    return np.stack([x, y, w, h], axis=1)
