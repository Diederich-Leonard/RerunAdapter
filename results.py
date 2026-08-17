"""Readers for a SLAM result directory: estimated trajectory and submap meshes.

Stdlib + numpy + scipy (for SE3 composition): nothing here imports Rerun.

A result directory is expected to hold

* one or more ``*_trajectory.csv`` -- the optimised path of the IMU in the world frame;
* any number of ``mesh_*.ply`` -- submap meshes whose vertices are already in the world
  frame, so they need no transform to be placed.

:func:`read_pose_stream` reads a moving camera's own pose file, in either of two formats --
CSV rows already in the world frame, or a frame/pose JSON list rebased onto it (see
:func:`_rebase_to_anchor`).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

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


def _first_data_line(path: Path) -> bytes:
    """First line that is neither blank nor a ``#`` comment, or ``b""`` if there is none."""
    with path.open("rb") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped and not stripped.startswith(b"#"):
                return stripped
    return b""


def _is_comma_separated(path: Path) -> bool:
    """Tell a comma-separated pose file apart from the space-separated reference format.

    Decided on the first *data* row rather than the first line, because the header is not a
    reliable witness: a file can pair a space-separated ``# timestamp tx ty tz ...`` comment
    with comma-separated rows, and judging by the header would then get it backwards.
    """
    return b"," in _first_data_line(path)


def _load_rows(
    path: Path, delimiter: str | None = None, usecols: range | None = None
) -> np.ndarray:
    """Rows of a whitespace- or comma-separated numeric file.

    Both header conventions are in use -- ``#``-commented and bare -- so an uncommented
    header line is skipped on a retry.
    """
    try:
        return np.loadtxt(path, delimiter=delimiter, comments="#", usecols=usecols, ndmin=2)
    except ValueError:
        return np.loadtxt(
            path, delimiter=delimiter, comments="#", usecols=usecols, skiprows=1, ndmin=2
        )


def _clean_poses(
    path: Path, timestamps_ns: np.ndarray, positions: np.ndarray, quaternions: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort poses by time, drop repeated timestamps, and normalise the quaternions.

    Keeping the first of any duplicate matters because the realtime OKVIS file repeats its
    final state, and because Rerun's latest-at lookup would otherwise pick between rows
    sharing a timestamp arbitrarily.
    """
    order = np.argsort(timestamps_ns, kind="stable")
    timestamps_ns = timestamps_ns[order]
    positions, quaternions = positions[order], quaternions[order]

    unique = np.concatenate(([True], np.diff(timestamps_ns) != 0))
    timestamps_ns = timestamps_ns[unique]
    positions, quaternions = positions[unique], quaternions[unique]

    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise ValueError(f"{path}: contains a zero-length quaternion")
    return timestamps_ns, positions, quaternions / norms


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
        raw = _load_rows(path)

    if raw.size == 0:
        raise ValueError(f"{path}: no pose rows")
    if not comma and raw.shape[1] < 8:
        raise ValueError(
            f"{path}: needs 8 columns (timestamp, position, xyzw quaternion) to place the "
            f"sensors, got {raw.shape[1]}"
        )

    timestamps = (
        raw[:, 0].astype(np.int64)  # already nanoseconds
        if comma
        else np.rint(raw[:, 0] * NS_PER_S).astype(np.int64)  # seconds -> nanoseconds
    )
    timestamps, positions, quaternions = _clean_poses(
        path, timestamps, raw[:, 1:4], raw[:, 4:8]
    )
    return Trajectory(
        timestamps_ns=timestamps, positions=positions, quaternions=quaternions
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
# Pose streams
# ----------------------------------------------------------------------------------


@dataclass
class PoseStream:
    """A time-stamped sequence of rigid poses, carrying no frame convention of its own.

    What the poses *mean* is decided by whoever composes them -- unlike :class:`Trajectory`,
    which is specifically the IMU in the world. This is what a moving camera's own pose file
    is read into: there each pose is ``T_WC``, the camera's pose in the world frame -- either
    natively (a CSV file) or rebased onto it (a JSON file; see :func:`read_pose_stream`).
    """

    name: str
    timestamps_ns: np.ndarray  # (N,) int64
    positions: np.ndarray  # (N, 3)
    quaternions: np.ndarray  # (N, 4) xyzw, normalised

    def __len__(self) -> int:
        return len(self.timestamps_ns)


def _read_pose_rows_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``timestamp tx ty tz qx qy qz qw`` rows, comma- or whitespace-separated.

    Timestamps are **nanoseconds on the dataset clock** already, as in the sensor and
    trajectory CSVs rather than the seconds a reference file uses. Only the first eight
    columns are read, so a trailing comma or extra columns are fine.
    """
    delimiter = "," if _is_comma_separated(path) else None
    try:
        raw = _load_rows(path, delimiter=delimiter, usecols=range(8))
    except (IndexError, ValueError) as error:
        raise ValueError(
            f"{path}: expected 8 columns (timestamp, position, xyzw quaternion) "
            f"of numbers -- {error}"
        ) from None

    if raw.size == 0:
        raise ValueError(f"{path}: no pose rows")

    return raw[:, 0].astype(np.int64), raw[:, 1:4], raw[:, 4:8]


def _read_pose_rows_json(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A ``[{"pose": {"timestamp": ..., "pose": {"rotation": ..., "translation": ...}}}]``
    list, as written by gastonpy's camera pose export.

    Unlike the CSV format, ``timestamp`` here is **seconds** on the dataset clock, so it is
    converted to nanoseconds the same way a space-separated reference file's is. ``rotation``
    is already ``xyzw``, and ``translation`` is ``[x, y, z]``; ``frame_id``/``filename`` are
    ignored. The poses themselves are in whatever frame their own producer used -- not
    necessarily the OKVIS world frame -- which :func:`read_pose_stream` corrects for by
    rebasing onto an anchor pose.
    """
    try:
        entries = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: not valid JSON -- {error}") from None

    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path}: expected a non-empty JSON list of pose entries")

    try:
        seconds = np.array(
            [entry["pose"]["timestamp"] for entry in entries], dtype=np.float64
        )
        positions = np.array(
            [entry["pose"]["pose"]["translation"] for entry in entries], dtype=np.float64
        )
        quaternions = np.array(
            [entry["pose"]["pose"]["rotation"] for entry in entries], dtype=np.float64
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"{path}: expected each entry to have pose.timestamp, "
            f"pose.pose.translation (xyz) and pose.pose.rotation (xyzw) -- {error}"
        ) from None

    if positions.shape[1:] != (3,) or quaternions.shape[1:] != (4,):
        raise ValueError(
            f"{path}: expected translation to have 3 components and rotation 4, got "
            f"{positions.shape[1:]} and {quaternions.shape[1:]}"
        )

    timestamps_ns = np.rint(seconds * NS_PER_S).astype(np.int64)
    return timestamps_ns, positions, quaternions


def _rebase_to_anchor(
    anchor: np.ndarray, positions: np.ndarray, quaternions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Rigidly shift a pose sequence so its first pose becomes ``anchor`` (a 4x4 ``T_SC``).

    A JSON pose stream is expressed in its own producer's arbitrary frame -- there is no
    reason it shares OKVIS's world frame, and in practice it does not. But the estimated IMU
    trajectory always starts at the world origin with identity rotation (``T_WS(0) = I``),
    so the camera's true world pose at that same first timestamp is exactly the config's
    static ``T_SC``, with no need for the world-frame trajectory itself to compute it. Rigidly
    transforming the whole stream so it starts there -- rather than wherever its own
    producer's frame happened to put it -- turns it into ``T_WC`` while preserving all of its
    internal (relative) motion untouched.
    """
    anchor_rotation = Rotation.from_matrix(anchor[:3, :3])
    anchor_translation = anchor[:3, 3]

    first_rotation_inv = Rotation.from_quat(quaternions[0]).inv()
    correction_rotation = anchor_rotation * first_rotation_inv
    correction_translation = anchor_translation - correction_rotation.apply(positions[0])

    new_positions = correction_rotation.apply(positions) + correction_translation
    new_quaternions = (correction_rotation * Rotation.from_quat(quaternions)).as_quat()
    return new_positions, new_quaternions


def read_pose_stream(path: str | Path, *, anchor: np.ndarray | None = None) -> PoseStream:
    """Read a moving camera's pose file, in either of two formats.

    * **CSV** -- ``timestamp tx ty tz qx qy qz qw`` rows, comma- or whitespace-separated,
      timestamps already nanoseconds on the dataset clock, poses already ``T_WC``.
    * **JSON** -- a frame/pose list (``.json`` extension), timestamps in seconds (converted
      to nanoseconds on read), poses rebased onto ``anchor`` -- see :func:`_rebase_to_anchor`
      -- which for this format is required and should be the camera's config ``T_SC``.

    Either way a pose stream is only meaningful against the measurements it is being
    composed with, so its timestamps have to end up on the same dataset clock as those.
    """
    path = Path(path)
    is_json = path.suffix.lower() == ".json"
    if is_json:
        timestamps_ns, positions, quaternions = _read_pose_rows_json(path)
    else:
        timestamps_ns, positions, quaternions = _read_pose_rows_csv(path)

    timestamps, positions, quaternions = _clean_poses(
        path, timestamps_ns, positions, quaternions
    )

    if is_json:
        if anchor is None:
            raise ValueError(
                f"{path}: a JSON pose file needs the camera's static T_SC (from --config) "
                f"to rebase its own frame onto the world frame"
            )
        positions, quaternions = _rebase_to_anchor(anchor, positions, quaternions)

    return PoseStream(
        name=path.stem,
        timestamps_ns=timestamps,
        positions=positions,
        quaternions=quaternions,
    )


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
    raw = _load_rows(path)
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
