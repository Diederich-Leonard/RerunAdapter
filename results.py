"""Readers for a SLAM result directory: estimated trajectory and submap meshes.

Pure stdlib + numpy: nothing here imports Rerun.

A result directory is expected to hold

* one or more ``*_trajectory.csv`` -- the optimised path of the IMU in the world frame;
* any number of ``mesh_*.ply`` -- submap meshes whose vertices are already in the world
  frame, so they need no transform to be placed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _natural_key(name: str) -> tuple:
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name))


# ----------------------------------------------------------------------------------
# Discovery
# ----------------------------------------------------------------------------------


def _trajectory_rank(path: Path) -> int:
    """Prefer the most-refined trajectory a run produced."""
    name = path.name
    if "final-ba" in name:
        return 0  # full final bundle adjustment
    if "final" in name:
        return 1  # final estimate
    return 2  # realtime estimate


def find_trajectory(results_dir: Path) -> Path | None:
    candidates = sorted(results_dir.glob("*_trajectory.csv"))
    return min(candidates, key=_trajectory_rank) if candidates else None


def find_meshes(results_dir: Path) -> list[Path]:
    return sorted(results_dir.glob("mesh_*.ply"), key=lambda p: _natural_key(p.name))


# ----------------------------------------------------------------------------------
# Trajectory
# ----------------------------------------------------------------------------------


@dataclass
class Trajectory:
    """Estimated poses of the IMU frame S in the world frame W."""

    timestamps_ns: np.ndarray  # (N,) int64
    positions: np.ndarray  # (N, 3) p_WS_W
    quaternions: np.ndarray  # (N, 4) q_WS as xyzw

    def __len__(self) -> int:
        return len(self.timestamps_ns)


def read_trajectory(path: str | Path) -> Trajectory:
    """Read an OKVIS trajectory CSV.

    Only the first eight columns are read -- ``timestamp``, ``p_WS_W_{xyz}`` and
    ``q_WS_{xyzw}``. That is deliberate rather than lazy: the rows carry a trailing comma,
    so a full-width parse sees one more field than the header names, and the ``-final``
    variants append an extra unnamed ``keyframeId`` column on top of that. Reading a fixed
    prefix by position sidesteps both quirks.

    Note the quaternion is stored **xyzw** (w last), which is also what Rerun expects.
    """
    path = Path(path)
    raw = np.loadtxt(path, delimiter=",", skiprows=1, usecols=range(8), dtype=np.float64,
                     ndmin=2)
    if len(raw) == 0:
        raise ValueError(f"{path}: no pose rows")

    # The realtime file repeats its final state; keep the first of any duplicate.
    timestamps = raw[:, 0].astype(np.int64)
    order = np.argsort(timestamps, kind="stable")
    raw, timestamps = raw[order], timestamps[order]
    unique = np.concatenate(([True], np.diff(timestamps) != 0))

    quaternions = raw[unique, 4:8]
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise ValueError(f"{path}: contains a zero-length quaternion")

    return Trajectory(
        timestamps_ns=timestamps[unique],
        positions=raw[unique, 1:4],
        quaternions=quaternions / norms,
    )


class PoseInterpolator:
    """Sample a :class:`Trajectory` at arbitrary timestamps.

    The trajectory is written at the estimator's own rate, which need not match the rate of
    the data being placed with it -- lidar scans in particular land between poses. Relying
    on Rerun's latest-at lookup would attach a scan to a pose up to one estimator period
    stale, which offsets the whole cloud; interpolating instead keeps it where it was
    measured. Queries outside the trajectory are clamped to its first or last pose.
    """

    def __init__(self, trajectory: Trajectory):
        if len(trajectory) == 0:
            raise ValueError("empty trajectory")
        self._t = trajectory.timestamps_ns.astype(np.float64)
        self._p = trajectory.positions
        self._q = trajectory.quaternions

    def at(self, timestamp_ns: int) -> tuple[np.ndarray, np.ndarray]:
        """``(position, quaternion_xyzw)`` at ``timestamp_ns``."""
        t = float(timestamp_ns)
        if t <= self._t[0]:
            return self._p[0], self._q[0]
        if t >= self._t[-1]:
            return self._p[-1], self._q[-1]

        i = int(np.searchsorted(self._t, t)) - 1
        t0, t1 = self._t[i], self._t[i + 1]
        alpha = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
        position = self._p[i] * (1.0 - alpha) + self._p[i + 1] * alpha
        return position, _slerp(self._q[i], self._q[i + 1], alpha)


def _slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """Spherical linear interpolation between two xyzw quaternions."""
    dot = float(np.dot(q0, q1))
    if dot < 0.0:  # take the shorter arc
        q1, dot = -q1, -dot
    if dot > 0.9995:  # nearly parallel: lerp and renormalise
        out = q0 + alpha * (q1 - q0)
        return out / np.linalg.norm(out)

    theta_0 = np.arccos(dot)
    theta = theta_0 * alpha
    sin_theta_0 = np.sin(theta_0)
    return (np.sin(theta_0 - theta) / sin_theta_0) * q0 + (np.sin(theta) / sin_theta_0) * q1


# ----------------------------------------------------------------------------------
# Meshes
# ----------------------------------------------------------------------------------


@dataclass
class Mesh:
    """One submap mesh, with vertices already in the world frame."""

    name: str
    vertices: np.ndarray  # (V, 3) float32
    triangles: np.ndarray | None  # (F, 3) int32, or None for a plain triangle list

    def __len__(self) -> int:
        return len(self.vertices)


def read_mesh_ply(path: str | Path) -> Mesh:
    """Read an ASCII PLY mesh, keeping only the vertex coordinates and faces.

    Vertex colours are ignored: the exports these come from write them, but write them all
    zero, so honouring them would render every mesh black.

    Meshes produced by marching cubes usually do not share vertices between faces
    (``V == 3F``, the faces being ``[0,1,2], [3,4,5], ...``). When that holds the face block
    is skipped entirely and the vertices are handed over as a plain triangle list, which
    avoids parsing and transferring a third of the file for no gain.
    """
    path = Path(path)
    header: list[str] = []
    with path.open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"{path}: no end_header found")
            text = line.decode("ascii", "replace").strip()
            header.append(text)
            if text == "end_header":
                break

    if not any(line.startswith("format ascii") for line in header):
        raise ValueError(f"{path}: only ASCII PLY is supported")

    counts: dict[str, int] = {}
    for line in header:
        if line.startswith("element"):
            _, name, number = line.split()[:3]
            counts[name] = int(number)

    num_vertices = counts.get("vertex", 0)
    num_faces = counts.get("face", 0)
    if num_vertices == 0:
        raise ValueError(f"{path}: no vertices")

    header_lines = len(header)
    vertices = np.loadtxt(
        path, skiprows=header_lines, max_rows=num_vertices, usecols=(0, 1, 2),
        dtype=np.float32, ndmin=2,
    )

    triangles = None
    if num_faces and num_vertices != 3 * num_faces:
        # Vertices are shared, so the face list is needed. Column 0 is the vertex count per
        # face, which is assumed to be 3.
        triangles = np.loadtxt(
            path, skiprows=header_lines + num_vertices, max_rows=num_faces,
            usecols=(1, 2, 3), dtype=np.int32, ndmin=2,
        )

    return Mesh(name=path.stem, vertices=vertices, triangles=triangles)
