"""Equidistant (Kannala-Brandt / OpenCV fisheye) camera undistortion.

Nothing here imports Rerun or decodes files, so the maths can be tested and reused on its
own, the same as ``euroc.py``, ``results.py`` and ``calib.py``.

The model, matching OKVIS's ``EquidistantDistortion`` and OpenCV's ``fisheye`` module::

    r = sqrt(x^2 + y^2)                                    (normalised pinhole coordinates)
    theta = atan(r)
    theta_d = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)
    (x', y') = (theta_d / r) * (x, y)

:func:`equidistant_map` inverts this by evaluating the *forward* model at every pixel of a
plain-pinhole raster built from the same ``K`` used for the logged ``Pinhole`` archetype, so
the undistorted result lines up with the frustum and any 2D overlay without a second camera
matrix to keep in sync -- pixel ``(u, v)`` of the output is the same ray as pixel ``(u, v)``
of a camera with no distortion at all. Building that map is plain numpy, done once per
camera. :func:`remap` is called once per *frame* though, and a hand-rolled numpy version
(fancy-indexing the four neighbours, then blending by hand) measured ~200ms on a 1920x1080
frame -- almost entirely the gather and the float32 blend, both of which touch every pixel
with a memory-access pattern that has no locality, since the fisheye map isn't affine.
``cv2.remap`` does the same bilinear sample in ~2ms, so OpenCV owns this one function.
"""

from __future__ import annotations

import cv2
import numpy as np


def equidistant_map(
    K: np.ndarray, distortion: np.ndarray, resolution: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Backward map from an undistorted raster (on ``K``) into the distorted source image.

    Returns ``(map_x, map_y)``, each ``(height, width)`` float32: the source pixel that
    belongs at each destination pixel, ready for :func:`remap`. Fewer than four coefficients
    are zero-padded and any beyond four are ignored, so a config that (unusually) supplies a
    shorter or longer list still loads rather than raising.
    """
    width, height = int(resolution[0]), int(resolution[1])
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    coeffs = np.zeros(4, dtype=np.float64)
    flat = np.asarray(distortion, dtype=np.float64).reshape(-1)
    coeffs[: min(4, flat.size)] = flat[:4]
    k1, k2, k3, k4 = coeffs

    u, v = np.meshgrid(
        np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64)
    )
    x = (u - cx) / fx
    y = (v - cy) / fy
    r = np.hypot(x, y)
    theta = np.arctan(r)
    theta2 = theta * theta
    theta_d = theta * (1.0 + theta2 * (k1 + theta2 * (k2 + theta2 * (k3 + theta2 * k4))))

    # r is zero only at the principal point, where the ratio's limit is 1 (theta ~ r for
    # small r, and the correction terms vanish faster than r); dividing directly would only
    # turn that single pixel into a 0/0.
    scale = np.divide(theta_d, r, out=np.ones_like(r), where=r > 1e-9)

    map_x = (fx * x * scale + cx).astype(np.float32)
    map_y = (fy * y * scale + cy).astype(np.float32)
    return map_x, map_y


def remap(image: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    """Bilinear-sample ``image`` at ``(map_x, map_y)``; pixels that fall outside come out black.

    ``image`` is ``(H, W)`` or ``(H, W, C)`` uint8; ``map_x``/``map_y`` are ``(H', W')``, the
    shape of the *output*, and need not match ``image``'s own shape.
    """
    return cv2.remap(
        image, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
