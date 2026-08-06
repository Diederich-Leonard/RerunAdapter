"""Turbo colormap plus fixed value-to-colour mappings.

Kept as plain data so the program depends only on ``rerun-sdk`` and ``numpy``.

Every mapping here is **fixed and absolute**: a given input value always produces the
same colour, regardless of which scan, which dataset or which value range happens to be
present. Nothing is stretched to the data's min/max and nothing is clipped, so colours
are comparable while scrubbing the timeline and across sensors.

Achieving that for unbounded quantities needs a normalisation that is monotonic over the
whole domain rather than a linear window with cutoffs. Ranges and intensities use
``v / (v + scale)``, which maps ``[0, inf)`` onto ``[0, 1)``; heights use a ``tanh``,
which maps all of the reals onto ``(0, 1)``. The reference scales below set where the
middle of the colormap falls; they are constants, not data-derived, which is what keeps
the mapping stable.
"""

from __future__ import annotations

import numpy as np

#: Turbo sampled at t = 0, 1/16, ..., 1.
_ANCHORS = np.array(
    [
        [48, 18, 59],
        [65, 69, 171],
        [70, 107, 227],
        [58, 143, 254],
        [33, 176, 242],
        [25, 204, 212],
        [36, 224, 176],
        [73, 236, 133],
        [124, 242, 93],
        [169, 251, 61],
        [205, 253, 40],
        [233, 245, 33],
        [252, 220, 42],
        [254, 184, 41],
        [250, 140, 27],
        [237, 95, 13],
        [122, 4, 3],
    ],
    dtype=np.float64,
)


def _build_lut(size: int = 256) -> np.ndarray:
    """Interpolate :data:`_ANCHORS` to a ``(size, 3)`` uint8 table."""
    t = np.linspace(0.0, 1.0, size)
    anchor_positions = np.linspace(0.0, 1.0, len(_ANCHORS))
    channels = [np.interp(t, anchor_positions, _ANCHORS[:, c]) for c in range(3)]
    return np.stack(channels, axis=1).round().astype(np.uint8)


#: ``(256, 3)`` uint8 Turbo table, built once at import.
TURBO = _build_lut()

#: Value that lands in the middle of the colormap, per quantity. Fixed on purpose.
INTENSITY_SCALE = 32.0  # lidar intensity / reflectivity, whatever the sensor's units
RANGE_SCALE = 20.0  # metres from the sensor
HEIGHT_SCALE = 5.0  # metres above/below the sensor


def turbo(t: np.ndarray) -> np.ndarray:
    """Map normalised values in ``[0, 1]`` to ``(N, 3)`` uint8 RGB.

    The uint8 return type is load-bearing: Rerun reads float colour arrays as 0..1 and
    integer arrays as 0..255, so returning floats would saturate everything to white.
    """
    t = np.asarray(t, dtype=np.float64)
    if t.size == 0:
        return np.empty((0, 3), dtype=np.uint8)
    # This clip only keeps LUT indices in bounds; the normalisers below already
    # guarantee the range, so no data value is ever cut off by it.
    index = np.clip(t * (len(TURBO) - 1), 0, len(TURBO) - 1).astype(np.intp)
    return TURBO[index]


def saturating(values: np.ndarray, scale: float) -> np.ndarray:
    """Map a non-negative quantity onto ``[0, 1)`` as ``v / (v + scale)``.

    Monotonic over the whole domain, so arbitrarily large values stay distinguishable
    and none are clipped. ``scale`` is the value that maps to the middle of the ramp.
    """
    v = np.asarray(values, dtype=np.float64)
    v = np.maximum(v, 0.0)  # these quantities are non-negative by definition
    return v / (v + float(scale))


def symmetric(values: np.ndarray, scale: float) -> np.ndarray:
    """Map a signed quantity onto ``(0, 1)`` as ``0.5 + 0.5 * tanh(v / scale)``.

    Zero always lands exactly in the middle of the colormap, and the mapping is
    monotonic over all of the reals, so nothing is clipped.
    """
    v = np.asarray(values, dtype=np.float64)
    return 0.5 + 0.5 * np.tanh(v / float(scale))


def fraction(values: np.ndarray, count: int) -> np.ndarray:
    """Map integers ``0 .. count - 1`` onto ``[0, 1]``, e.g. a lidar ring index."""
    v = np.asarray(values, dtype=np.float64)
    if count <= 1:
        return np.zeros_like(v)
    return v / float(count - 1)
