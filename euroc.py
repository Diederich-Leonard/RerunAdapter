"""Readers for datasets in the EuRoC/ASL directory layout.

Pure stdlib + numpy, plus Pillow for the one function that decodes image bytes
(:func:`read_image`, needed only for on-the-fly undistortion): nothing here imports Rerun,
so the readers can be tested and reused on their own.

Expected layout -- stream folders directly under the dataset root::

    <root>/<image stream>/data.csv     #timestamp [ns],filename
    <root>/<image stream>/data/<ts>.<ext>
    <root>/imu0/data.csv               #timestamp [ns],w_RS_S_{xyz},a_RS_S_{xyz}
    <root>/lidar0/data.csv             #timestamp [ns],x,y,z,intensity,ring

Nothing about the sensor count is assumed: any folder holding both a ``data.csv`` and a
``data/`` subfolder is treated as an image stream, whatever it is called and however many
there are. One IMU and one lidar are assumed.

Two portability notes:

* **Line endings vary**, sometimes within a single dataset (a CRLF header followed by LF
  rows). ``csv.reader`` with ``newline=""`` and ``np.loadtxt`` both cope with either;
  hand-rolled ``split(",")`` does not, and leaves a trailing ``\\r`` on the last field.
* **Timestamps are integer nanoseconds on whatever clock the dataset uses** -- some are
  Unix epoch, others are elapsed time since the recording started. They are only ever
  compared against each other, never interpreted as wall-clock dates.
"""

from __future__ import annotations

import csv
import itertools
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

NS_PER_S = 1_000_000_000

#: Columns a lidar CSV must have to be a point cloud: timestamp, x, y, z, intensity, ring.
LIDAR_MIN_COLUMNS = 6


def _natural_key(name: str) -> tuple:
    """Sort key that orders ``cam2`` before ``cam10``."""
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name))


def _window_slice(
    timestamps_ns: np.ndarray, start_ns: int | None, end_ns: int | None
) -> tuple[int, int]:
    """Index range of ``timestamps_ns`` falling inside ``[start_ns, end_ns]``."""
    lo = 0 if start_ns is None else int(np.searchsorted(timestamps_ns, start_ns, "left"))
    hi = (
        len(timestamps_ns)
        if end_ns is None
        else int(np.searchsorted(timestamps_ns, end_ns, "right"))
    )
    return lo, hi


# ----------------------------------------------------------------------------------
# Discovery
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Dataset:
    """Which streams exist under a dataset root."""

    root: Path
    images: dict[str, Path]
    imu_csv: Path | None
    lidar_csv: Path | None

    def is_empty(self) -> bool:
        return not self.images and self.imu_csv is None and self.lidar_csv is None


def discover(root: str | Path) -> Dataset:
    """Find the sensor streams under ``root``.

    Every stream is optional so a partial dataset still visualises; the caller decides
    whether the result is usable.
    """
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"{root} is not a directory")

    images = {
        entry.name: entry
        for entry in sorted(root.iterdir(), key=lambda p: _natural_key(p.name))
        if entry.is_dir() and (entry / "data.csv").is_file() and (entry / "data").is_dir()
    }

    def first_csv(pattern: str) -> Path | None:
        for entry in sorted(root.glob(pattern), key=lambda p: _natural_key(p.name)):
            candidate = entry / "data.csv"
            if entry.is_dir() and candidate.is_file():
                return candidate
        return None

    return Dataset(
        root=root,
        images=images,
        imu_csv=first_csv("imu*"),
        lidar_csv=first_csv("lidar*"),
    )


# ----------------------------------------------------------------------------------
# Image streams
# ----------------------------------------------------------------------------------


@dataclass
class ImageIndex:
    """Timestamps paired with image files for one image stream."""

    name: str
    timestamps_ns: np.ndarray  # (N,) int64, strictly increasing
    paths: list[Path]
    missing: int  # rows whose image file was absent

    def __len__(self) -> int:
        return len(self.timestamps_ns)

    def window(self, start_ns: int | None, end_ns: int | None) -> ImageIndex:
        """Restrict to frames inside ``[start_ns, end_ns]``."""
        lo, hi = _window_slice(self.timestamps_ns, start_ns, end_ns)
        return ImageIndex(
            name=self.name,
            timestamps_ns=self.timestamps_ns[lo:hi],
            paths=self.paths[lo:hi],
            missing=self.missing,
        )


def read_image_index(stream_dir: str | Path) -> ImageIndex:
    """Read ``<stream>/data.csv`` and pair each timestamp with its file in ``data/``.

    Rows whose image is missing are dropped and counted rather than raising, so one
    truncated stream does not block the whole visualisation.
    """
    stream_dir = Path(stream_dir)
    csv_path = stream_dir / "data.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"missing {csv_path}")

    rows: dict[int, str] = {}
    with csv_path.open(newline="") as handle:
        for record in csv.reader(handle):
            if not record or record[0].lstrip().startswith("#") or len(record) < 2:
                continue
            rows.setdefault(int(record[0]), record[1].strip())

    data_dir = stream_dir / "data"
    timestamps: list[int] = []
    paths: list[Path] = []
    missing = 0
    for timestamp in sorted(rows):
        path = data_dir / rows[timestamp]
        if path.is_file():
            timestamps.append(timestamp)
            paths.append(path)
        else:
            missing += 1

    return ImageIndex(
        name=stream_dir.name,
        timestamps_ns=np.asarray(timestamps, dtype=np.int64),
        paths=paths,
        missing=missing,
    )


def read_image(path: str | Path) -> np.ndarray:
    """Decode one image file into ``(H, W)`` greyscale or ``(H, W, 3)`` RGB.

    Only needed for on-the-fly processing (currently: undistortion) -- the normal logging
    path streams the encoded bytes straight to the viewer and never decodes anything.
    """
    with Image.open(path) as handle:
        if handle.mode != "L":
            handle = handle.convert("RGB")
        return np.asarray(handle)


# ----------------------------------------------------------------------------------
# IMU
# ----------------------------------------------------------------------------------


@dataclass
class ImuData:
    """One IMU stream, split into named triplets.

    The fields are named rather than returned as a raw array because the EuRoC column
    order is **gyro first, then accelerometer** -- the opposite of how the two are
    usually spoken about. Keeping the split here means the ordering is applied once.
    """

    timestamps_ns: np.ndarray  # (N,) int64
    gyro: np.ndarray  # (N, 3) rad/s
    accel: np.ndarray  # (N, 3) m/s^2

    def __len__(self) -> int:
        return len(self.timestamps_ns)

    def window(self, start_ns: int | None, end_ns: int | None) -> ImuData:
        """Restrict to samples inside ``[start_ns, end_ns]``."""
        lo, hi = _window_slice(self.timestamps_ns, start_ns, end_ns)
        return ImuData(
            timestamps_ns=self.timestamps_ns[lo:hi],
            gyro=self.gyro[lo:hi],
            accel=self.accel[lo:hi],
        )


def read_imu(csv_path: str | Path) -> ImuData:
    """Read an ``imu0/data.csv``."""
    csv_path = Path(csv_path)
    raw = np.loadtxt(csv_path, delimiter=",", comments="#", dtype=np.float64, ndmin=2)
    if raw.shape[1] < 7:
        raise ValueError(f"{csv_path}: expected 7 columns, got {raw.shape[1]}")

    raw = raw[np.argsort(raw[:, 0], kind="stable")]
    return ImuData(
        timestamps_ns=raw[:, 0].astype(np.int64),
        gyro=raw[:, 1:4].copy(),
        accel=raw[:, 4:7].copy(),
    )


# ----------------------------------------------------------------------------------
# Lidar
# ----------------------------------------------------------------------------------


@dataclass
class LidarScan:
    """All points from one time interval, in the lidar sensor frame, no-returns removed."""

    index: int  # 0-based scan number in the file
    timestamp_ns: int  # start of the interval this scan covers
    xyz: np.ndarray  # (N, 3) float32
    intensity: np.ndarray  # (N,) float32
    ring: np.ndarray  # (N,) int16
    num_rows: int  # rows read for this scan, before filtering
    num_invalid: int  # rows dropped as (0, 0, 0) no-returns

    def __len__(self) -> int:
        return len(self.xyz)


class LidarScanReader:
    """Streaming reader for a point-per-row lidar CSV.

    These files are large (from ~1 GB to ~100 GB in the datasets this targets), so one is
    never loaded whole. Every row is treated as an individual point, and points are grouped
    into scans **by time**: everything whose timestamp falls in the same ``period_ns``
    interval forms one point cloud.

    Grouping by time rather than by a fixed row count is what makes this portable. Some
    exports drop invalid returns, so their number of rows per revolution follows the scene
    instead of the sensor -- one dataset here varies between 111k and 146k rows per
    revolution -- and no fixed row count can track that. A time interval always can.

    Each scan is parsed, yielded and discarded, so peak memory stays a few MB regardless of
    file size, and ``max_scans`` stops reading rather than skipping.
    """

    def __init__(self, path: str | Path, period_ns: int, chunk_rows: int = 200_000):
        self.path = Path(path)
        self.period_ns = int(period_ns)
        if self.period_ns <= 0:
            raise ValueError("period_ns must be positive")
        self.chunk_rows = int(chunk_rows)

    def column_count(self) -> int:
        """Number of comma-separated fields in the first data row.

        Used to tell a point cloud from something else sharing the ``lidar*`` name --
        a position-only ground-truth track, for instance, has just four columns.
        """
        with self.path.open("rb") as handle:
            for line in handle:
                if line.startswith(b"#") or not line.strip():
                    continue
                return line.count(b",") + 1
        return 0

    def time_range(self) -> tuple[int, int]:
        """``(first, last)`` timestamp, read from the head and tail only."""
        size = self.path.stat().st_size
        with self.path.open("rb") as handle:
            first_line = handle.readline()
            if first_line.startswith(b"#"):
                first_line = handle.readline()
            if not first_line:
                raise ValueError(f"{self.path} contains no data rows")
            first = int(first_line.split(b",")[0])

            handle.seek(max(0, size - 65536))
            tail = handle.read()

        fields = first_line.count(b",")
        complete = [line for line in tail.split(b"\n") if line.count(b",") == fields]
        if not complete:
            raise ValueError(f"{self.path}: could not read a complete row from the tail")
        return first, int(complete[-1].split(b",")[0])

    def _line_chunks(self):
        """Yield lists of raw data lines, streaming through the file."""
        with self.path.open("rb") as handle:
            first_line = handle.readline()
            if not first_line.startswith(b"#"):
                handle.seek(0)  # no header, rewind
            while True:
                lines = list(itertools.islice(handle, self.chunk_rows))
                if not lines:
                    return  # clean EOF
                yield lines
                del lines

    @staticmethod
    def _timestamp_of(line: bytes) -> int | None:
        """Timestamp of one row, read straight from the bytes.

        Returns ``None`` if the field will not parse, so a truncated final line makes the
        caller fall back to parsing rather than mis-skipping.
        """
        try:
            return int(line.split(b",", 1)[0].strip())
        except ValueError:
            return None

    def scans(self, start_ns: int | None = None, end_ns: int | None = None):
        """Yield one :class:`LidarScan` per ``period_ns`` interval within a time window.

        Rows are read in chunks and cut wherever the interval index changes; a scan that
        straddles a chunk boundary is accumulated across chunks, so the interval, not the
        read size, decides where scans begin and end.

        The file is sorted by time, so a chunk lying wholly before the window is skipped
        without being parsed and reading stops at the first chunk past its end. Only the
        cheap leading timestamp field is inspected to decide, which is what makes a narrow
        window affordable on a file far too large to read in full.
        """
        emitted = 0
        pending: list[np.ndarray] = []
        bucket: int | None = None

        for lines in self._line_chunks():
            if end_ns is not None:
                first = self._timestamp_of(lines[0])
                if first is not None and first > end_ns:
                    break
            if start_ns is not None:
                last = self._timestamp_of(lines[-1])
                if last is not None and last < start_ns:
                    continue

            rows = np.loadtxt(lines, delimiter=",", dtype=np.float64, ndmin=2)
            if start_ns is not None or end_ns is not None:
                keep = np.ones(len(rows), dtype=bool)
                if start_ns is not None:
                    keep &= rows[:, 0] >= start_ns
                if end_ns is not None:
                    keep &= rows[:, 0] <= end_ns
                rows = rows[keep]
                if len(rows) == 0:
                    continue

            # Interval index per row. float64 rounding is order-preserving, so this stays
            # non-decreasing even for epoch-nanosecond timestamps (which exceed float64's
            # exact-integer range and so carry ~256 ns of rounding); a point can only ever
            # land in the wrong interval within that tolerance of a boundary.
            buckets = np.floor_divide(rows[:, 0], self.period_ns).astype(np.int64)
            edges = np.flatnonzero(buckets[1:] != buckets[:-1]) + 1

            for segment, segment_buckets in zip(
                np.split(rows, edges), np.split(buckets, edges), strict=True
            ):
                segment_bucket = int(segment_buckets[0])
                if bucket is None or segment_bucket == bucket:
                    pending.append(segment)
                    bucket = segment_bucket
                    continue

                yield self._build_scan(emitted, bucket, pending)
                emitted += 1
                pending, bucket = [segment], segment_bucket

        if pending:
            yield self._build_scan(emitted, bucket, pending)

    def _build_scan(self, index: int, bucket: int, parts: list[np.ndarray]) -> LidarScan:
        """Assemble the rows collected for one interval into a :class:`LidarScan`."""
        raw = parts[0] if len(parts) == 1 else np.concatenate(parts)
        num_rows = len(raw)

        # Exact (0, 0, 0) rows are no-returns in some exports. Filter on the coordinates
        # only: those rows can carry a nonzero intensity, so filtering on intensity
        # would keep them.
        kept = raw[np.any(raw[:, 1:4] != 0.0, axis=1)]

        return LidarScan(
            index=index,
            # The start of the interval, as an exact integer.
            timestamp_ns=bucket * self.period_ns,
            xyz=np.ascontiguousarray(kept[:, 1:4], dtype=np.float32),
            intensity=np.ascontiguousarray(kept[:, 4], dtype=np.float32),
            ring=np.ascontiguousarray(kept[:, 5], dtype=np.int16),
            num_rows=num_rows,
            num_invalid=num_rows - len(kept),
        )
