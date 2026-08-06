"""Sensor extrinsics and intrinsics from an OKVIS configuration file.

Pure stdlib + numpy + PyYAML: nothing here imports Rerun.

Frame conventions, following OKVIS: ``T_AB`` maps points from frame B into frame A, so
``p_A = T_AB * p_B``. The frames involved are

* ``W`` -- world, the frame the estimated trajectory is expressed in;
* ``S`` -- the IMU (sensor) frame, which is what a trajectory's ``T_WS`` refers to;
* ``B`` -- the body frame, given by ``imu_parameters.T_BS``;
* ``C_i`` -- camera *i*, given by ``cameras[i].T_SC``;
* ``L`` -- the lidar, given by ``lidar.T_SL``.

Both ``T_SC`` and ``T_SL`` are therefore already relative to the IMU, which is what makes
``S`` the natural root to hang the sensors off. ``T_BS`` runs the other way, so placing the
body frame under the IMU needs its inverse -- see :func:`invert`.

The files are OpenCV-flavoured YAML: they open with a ``%YAML:1.0``-style line that uses a
colon where the YAML spec wants a space, which PyYAML rejects as a malformed directive. It
is dropped before parsing. Matrices are stored as flat, row-major lists of 16 numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass(frozen=True)
class Camera:
    """Intrinsics and extrinsics for one camera."""

    index: int
    T_SC: np.ndarray  # (4, 4) camera -> IMU
    K: np.ndarray  # (3, 3) pinhole intrinsics
    resolution: tuple[int, int]  # (width, height)
    distortion: np.ndarray
    distortion_type: str
    camera_type: str


@dataclass(frozen=True)
class Calibration:
    cameras: list[Camera]
    T_BS: np.ndarray  # (4, 4) IMU -> body
    T_SL: np.ndarray | None  # (4, 4) lidar -> IMU, absent in configs without a lidar


def invert(T: np.ndarray) -> np.ndarray:
    """Inverse of a rigid 4x4 transform, without a general matrix inverse."""
    R = T[:3, :3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ T[:3, 3]
    return out


def _matrix(values, name: str) -> np.ndarray:
    """Turn a flat row-major list of 16 numbers into a validated 4x4 transform."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size != 16:
        raise ValueError(f"{name}: expected 16 numbers, got {array.size}")
    T = array.reshape(4, 4)

    if not np.allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError(f"{name}: bottom row is {T[3].tolist()}, expected [0, 0, 0, 1]")
    # A non-orthonormal rotation block usually means the matrix was pasted in the wrong
    # order; warn rather than fail, since the visualisation is still informative.
    R = T[:3, :3]
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-3):
        raise ValueError(f"{name}: rotation block is not orthonormal")
    return T


def load(path: str | Path) -> Calibration:
    """Read an OKVIS config and return the calibration it describes."""
    path = Path(path)
    lines = path.read_text().splitlines()
    if lines and lines[0].lstrip().startswith("%YAML"):
        lines = lines[1:]  # OpenCV writes '%YAML:1.0', which is not a valid directive

    document = yaml.safe_load("\n".join(lines))
    if not isinstance(document, dict):
        raise ValueError(f"{path}: not a YAML mapping")

    entries = document.get("cameras") or []
    cameras: list[Camera] = []
    for index, entry in enumerate(entries):
        focal = entry["focal_length"]
        principal = entry["principal_point"]
        width, height = entry["image_dimension"]
        K = np.array(
            [[focal[0], 0.0, principal[0]], [0.0, focal[1], principal[1]], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        cameras.append(
            Camera(
                index=index,
                T_SC=_matrix(entry["T_SC"], f"{path.name}: cameras[{index}].T_SC"),
                K=K,
                resolution=(int(width), int(height)),
                distortion=np.asarray(
                    entry.get("distortion_coefficients", []), dtype=np.float64
                ),
                distortion_type=str(entry.get("distortion_type", "none")),
                camera_type=str(entry.get("camera_type", "gray")),
            )
        )

    imu = document.get("imu_parameters") or {}
    T_BS = (
        _matrix(imu["T_BS"], f"{path.name}: imu_parameters.T_BS")
        if "T_BS" in imu
        else np.eye(4)
    )

    lidar = document.get("lidar") or {}
    T_SL = _matrix(lidar["T_SL"], f"{path.name}: lidar.T_SL") if "T_SL" in lidar else None

    return Calibration(cameras=cameras, T_BS=T_BS, T_SL=T_SL)
