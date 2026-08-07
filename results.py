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

NS_PER_S = 1_000_000_000


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


DEFAULT_MESH_PATTERN = r"mesh_.*\.ply"


def find_meshes(results_dir: Path, pattern: str | re.Pattern = DEFAULT_MESH_PATTERN) -> list[Path]:
    """Mesh files in ``results_dir`` whose *name* fully matches ``pattern``.

    A regex rather than a glob, and matched in full rather than searched, because a single
    result directory can hold several mesh sets whose names share a suffix -- supereight
    writes ``mesh_*.ply`` next to nvblox's ``nvblox_mesh_*.ply``. Full matching is what
    keeps the default from picking up both.
    """
    matches = re.compile(pattern).fullmatch
    return sorted(
        (path for path in results_dir.iterdir() if path.is_file() and matches(path.name)),
        key=lambda p: _natural_key(p.name),
    )


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


def _is_comma_separated(path: Path) -> bool:
    """Tell the OKVIS CSV apart from the space-separated reference format.

    The OKVIS CSV has commas in its header and in every row; reference files have none.
    """
    with path.open("rb") as handle:
        for line in handle:
            if line.strip():
                return b"," in line
    return False


def _load_reference_rows(path: Path) -> np.ndarray:
    """Rows of a space-separated reference file.

    Both header conventions are in use -- ``#``-commented and bare -- so an uncommented
    header line is skipped on a retry.
    """
    try:
        return np.loadtxt(path, comments="#", ndmin=2)
    except ValueError:
        return np.loadtxt(path, comments="#", skiprows=1, ndmin=2)


def read_trajectory(path: str | Path) -> Trajectory:
    """Read a trajectory in either of the two formats in use.

    Both are accepted so a *reference* trajectory can drive the scene exactly as an
    estimated one does, which is what makes the dataset viewable in a world frame before
    any SLAM output exists.

    * **OKVIS CSV** -- comma separated, one header line, timestamps in nanoseconds. Only
      the first eight columns are read (``timestamp``, ``p_WS_W_{xyz}``, ``q_WS_{xyzw}``),
      which is deliberate rather than lazy: the rows carry a trailing comma, so a
      full-width parse sees one more field than the header names, and the ``-final``
      variants append an unnamed ``keyframeId`` column on top of that.
    * **Reference** -- space separated, timestamps in seconds. The quaternion columns are
      required here, because orientation is what lets the sensors be placed; a
      position-only file can still be drawn as a line via :func:`read_ground_truth`.

    Either way the quaternion is **xyzw** (w last), which is also what Rerun expects.
    """
    path = Path(path)
    comma = _is_comma_separated(path)
    if comma:
        raw = np.loadtxt(path, delimiter=",", skiprows=1, usecols=range(8),
                         dtype=np.float64, ndmin=2)
    else:
        raw = _load_reference_rows(path)

    if raw.size == 0:
        raise ValueError(f"{path}: no pose rows")
    if not comma and raw.shape[1] < 8:
        raise ValueError(
            f"{path}: needs 8 columns (timestamp, position, xyzw quaternion) to place the "
            f"sensors, got {raw.shape[1]}"
        )

    # The realtime OKVIS file repeats its final state; keep the first of any duplicate.
    timestamps = (
        raw[:, 0].astype(np.int64)  # already nanoseconds
        if comma
        else np.rint(raw[:, 0] * NS_PER_S).astype(np.int64)  # seconds -> nanoseconds
    )
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


# PLY scalar type names, including the explicitly sized spellings, as numpy codes.
_PLY_TYPES = {
    "char": "i1", "int8": "i1",
    "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2",
    "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4",
    "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4",
    "double": "f8", "float64": "f8",
}


@dataclass
class _Element:
    """One ``element`` block of a PLY header."""

    name: str
    count: int
    # (property name, scalar type) for a plain property, or (name, (count type, item
    # type)) for a ``property list``.
    properties: list[tuple[str, str | tuple[str, str]]]


def _parse_ply_header(path: Path) -> tuple[str, list[_Element], int]:
    """``(format, elements, data_offset)`` for a PLY file.

    The offset is a byte count rather than a line count so that the binary payload can be
    seeked to directly; the ASCII path turns it back into a line count itself.
    """
    fmt: str | None = None
    elements: list[_Element] = []
    with path.open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"{path}: no end_header found")
            fields = line.decode("ascii", "replace").split()
            if not fields:
                continue
            keyword = fields[0]
            if keyword == "format":
                fmt = fields[1]
            elif keyword == "element":
                elements.append(_Element(fields[1], int(fields[2]), []))
            elif keyword == "property":
                if not elements:
                    raise ValueError(f"{path}: property outside of an element")
                if fields[1] == "list":
                    elements[-1].properties.append((fields[4], (fields[2], fields[3])))
                else:
                    elements[-1].properties.append((fields[2], fields[1]))
            elif keyword == "end_header":
                return fmt or "", elements, handle.tell()


def _scalar_dtype(path: Path, byte_order: str, type_name: str) -> str:
    try:
        return byte_order + _PLY_TYPES[type_name]
    except KeyError:
        raise ValueError(f"{path}: unsupported PLY type {type_name!r}") from None


def _read_binary_mesh(
    path: Path, byte_order: str, elements: list[_Element], offset: int, want_faces: bool
) -> tuple[np.ndarray, np.ndarray | None]:
    """Vertices and, if asked for, triangles of a binary PLY.

    Elements are fixed-width records laid out back to back, so each block is read straight
    into a structured array and the columns of interest are picked out afterwards. That
    means walking the elements in order even when only the two of them matter: a block that
    is skipped still has to be measured to find where the next one starts.
    """
    data = np.memmap(path, dtype=np.uint8, mode="r", offset=offset)
    vertices: np.ndarray | None = None
    triangles: np.ndarray | None = None
    cursor = 0

    for element in elements:
        lists = [(name, spec) for name, spec in element.properties if isinstance(spec, tuple)]
        if lists and element.name != "face":
            raise ValueError(
                f"{path}: cannot size element {element.name!r}, which has a list property"
            )

        if element.name == "face":
            if not want_faces:
                break  # nothing after the faces is read, so the layout no longer matters
            if len(element.properties) != 1 or not lists:
                raise ValueError(f"{path}: expected a single face list property")
            count_type, index_type = lists[0][1]
            # Only triangles are supported, which lets the variable-length list be read as
            # a fixed-width record; the counts are checked below rather than assumed.
            record = np.dtype([
                ("count", _scalar_dtype(path, byte_order, count_type)),
                ("indices", _scalar_dtype(path, byte_order, index_type), 3),
            ])
            faces = np.frombuffer(data, dtype=record, count=element.count, offset=cursor)
            if np.any(faces["count"] != 3):
                raise ValueError(f"{path}: only triangular faces are supported")
            triangles = faces["indices"].astype(np.int32)
            break

        record = np.dtype([
            (name, _scalar_dtype(path, byte_order, spec)) for name, spec in element.properties
        ])
        if element.name == "vertex":
            block = np.frombuffer(data, dtype=record, count=element.count, offset=cursor)
            missing = [axis for axis in "xyz" if axis not in record.names]
            if missing:
                raise ValueError(f"{path}: vertex element is missing {', '.join(missing)}")
            vertices = np.stack(
                [block["x"], block["y"], block["z"]], axis=1
            ).astype(np.float32)
            if not want_faces:
                break
        cursor += element.count * record.itemsize

    if vertices is None:
        raise ValueError(f"{path}: no vertex element")
    return vertices, triangles


def read_mesh_ply(path: str | Path) -> Mesh:
    """Read a PLY mesh, keeping only the vertex coordinates and faces.

    ASCII and both binary byte orders are accepted, since the two producers in play do not
    agree: the OKVIS submap exports are ASCII, the nvblox ones binary little-endian.

    Vertex colours are ignored: the exports these come from write them, but write them all
    zero, so honouring them would render every mesh black.

    Meshes produced by marching cubes usually do not share vertices between faces
    (``V == 3F``, the faces being ``[0,1,2], [3,4,5], ...``). When that holds the face block
    is skipped entirely and the vertices are handed over as a plain triangle list, which
    avoids parsing and transferring a third of the file for no gain.
    """
    path = Path(path)
    fmt, elements, data_offset = _parse_ply_header(path)

    counts = {element.name: element.count for element in elements}
    num_vertices = counts.get("vertex", 0)
    num_faces = counts.get("face", 0)
    if num_vertices == 0:
        raise ValueError(f"{path}: no vertices")

    # Vertices are shared only when the count rules out a plain triangle list.
    want_faces = bool(num_faces) and num_vertices != 3 * num_faces

    if fmt == "binary_little_endian":
        vertices, triangles = _read_binary_mesh(path, "<", elements, data_offset, want_faces)
    elif fmt == "binary_big_endian":
        vertices, triangles = _read_binary_mesh(path, ">", elements, data_offset, want_faces)
    elif fmt == "ascii":
        with path.open("rb") as handle:
            header_lines = handle.read(data_offset).count(b"\n")
        vertices = np.loadtxt(
            path, skiprows=header_lines, max_rows=num_vertices, usecols=(0, 1, 2),
            dtype=np.float32, ndmin=2,
        )
        triangles = None
        if want_faces:
            # Column 0 is the vertex count per face, which is assumed to be 3.
            triangles = np.loadtxt(
                path, skiprows=header_lines + num_vertices, max_rows=num_faces,
                usecols=(1, 2, 3), dtype=np.int32, ndmin=2,
            )
    else:
        raise ValueError(f"{path}: unsupported PLY format {fmt!r}")

    return Mesh(name=path.stem, vertices=vertices, triangles=triangles)


# ----------------------------------------------------------------------------------
# Ground truth and frame alignment
# ----------------------------------------------------------------------------------


@dataclass
class GroundTruth:
    """Reference positions, expressed in their own world frame."""

    timestamps_ns: np.ndarray  # (N,) int64
    positions: np.ndarray  # (N, 3)

    def __len__(self) -> int:
        return len(self.timestamps_ns)


def read_ground_truth(path: str | Path) -> GroundTruth:
    """Read a space-separated reference file: ``timestamp tx ty tz [qx qy qz qw ...]``.

    The timestamps are in **seconds** here, unlike the nanoseconds used by the sensor and
    trajectory CSVs, and are converted on read. Only the position columns are kept, since
    the alignment below is position-only -- so unlike :func:`read_trajectory`, a file
    without orientation is fine.
    """
    path = Path(path)
    raw = _load_reference_rows(path)
    if raw.shape[1] < 4:
        raise ValueError(f"{path}: expected at least 4 columns, got {raw.shape[1]}")

    timestamps_ns = np.rint(raw[:, 0] * NS_PER_S).astype(np.int64)
    order = np.argsort(timestamps_ns, kind="stable")
    return GroundTruth(timestamps_ns=timestamps_ns[order], positions=raw[order, 1:4])


def _nearest_index(pool: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Index of the nearest ``pool`` entry for each ``query`` value."""
    order = np.argsort(pool)
    sorted_pool = pool[order]
    position = np.clip(np.searchsorted(sorted_pool, query), 1, len(sorted_pool) - 1)
    left, right = sorted_pool[position - 1], sorted_pool[position]
    return order[np.where((query - left) <= (right - query), position - 1, position)]


def associate_nearest(
    t_e: np.ndarray, p_e: np.ndarray, t_g: np.ndarray, p_g: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Pair estimate and reference positions by nearest timestamp.

    The longer stream is subsampled, so both sides come back index-aligned and of equal
    length (that of the shorter stream).
    """
    if len(t_e) > len(t_g):
        return p_e[_nearest_index(t_e, t_g)], p_g
    return p_e, p_g[_nearest_index(t_g, t_e)]


def align_position_only(
    p_e: np.ndarray, p_g: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Rigid, no-scale Umeyama/Horn fit mapping the estimate onto the reference.

    Position-only: orientations are not used. Returns ``(p_e_aligned, T_ge)``.
    """
    e_centroid, g_centroid = p_e.mean(axis=0), p_g.mean(axis=0)
    H = (p_e - e_centroid).T @ (p_g - g_centroid)  # 3x3 cross-covariance

    U, _, Vt = np.linalg.svd(H)
    V = Vt.T
    R_ge = V @ U.T
    if np.linalg.det(R_ge) < 0.0:  # reflection fix
        V[:, 2] = -V[:, 2]
        R_ge = V @ U.T
    t_ge = g_centroid - R_ge @ e_centroid

    T_ge = np.eye(4)
    T_ge[:3, :3] = R_ge
    T_ge[:3, 3] = t_ge
    return (R_ge @ p_e.T).T + t_ge, T_ge


def align_ground_truth(
    trajectory: Trajectory, ground_truth: GroundTruth
) -> tuple[np.ndarray, float]:
    """Express the reference trajectory in the estimate's world frame.

    The alignment is computed the way the evaluation tooling does it -- nearest-timestamp
    association, then a position-only rigid fit of the estimate onto the reference -- but
    the resulting transform is applied the other way round, moving the reference onto the
    estimate rather than the estimate onto the reference. The fit is the same either way;
    inverting it just means the estimated poses, the meshes and the lidar stay exactly
    where they were logged and only the reference line moves.

    Returns ``(positions_in_estimate_frame, ate_rmse)``.
    """
    p_e, p_g = associate_nearest(
        trajectory.timestamps_ns.astype(np.float64),
        trajectory.positions,
        ground_truth.timestamps_ns.astype(np.float64),
        ground_truth.positions,
    )
    p_e_aligned, T_ge = align_position_only(p_e, p_g)
    rmse = float(np.sqrt(np.mean(np.sum((p_e_aligned - p_g) ** 2, axis=1))))

    R_eg = T_ge[:3, :3].T  # rigid inverse
    t_eg = -R_eg @ T_ge[:3, 3]
    return (R_eg @ ground_truth.positions.T).T + t_eg, rmse
