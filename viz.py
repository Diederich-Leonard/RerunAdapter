#!/usr/bin/env python3
"""Visualise a dataset in EuRoC/ASL layout in Rerun.

Every image stream gets its own 2D view, the lidar is shown in a lidar-fixed 3D view, and
the IMU is split into two graphs -- translational acceleration and rotational velocity --
all sharing the dataset's own nanosecond timeline.

Nothing about the sensor count is hardcoded: image streams are discovered from the
directory layout, so two cameras or five work the same way. One IMU and one lidar are
assumed.

This is the only module that talks to Rerun's logging API; ``euroc.py`` reads the dataset
and ``blueprint.py`` owns the entity paths and the view layout.

Usage:
    python viz.py <dataset_path>
    python viz.py <dataset_path> --max-scans 20
    python viz.py <dataset_path> --lidar-frequency 10
    python viz.py <dataset_path> --save scene.rrd
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import rerun as rr

import blueprint as bp
import colormap
import euroc

#: Point size in UI points. Rerun reads a negative radius as a screen-space size, which
#: keeps a sparse cloud visible whether you are 1 m or 100 m from it.
POINT_RADIUS_UI = -1.0

#: Per-axis series colours, shared by both IMU plots so x/y/z read the same in each.
AXIS_COLORS = ((230, 80, 80), (90, 200, 110), (90, 150, 240))

LIDAR_COLOR_MODES = ("intensity", "ring", "z", "range", "none")

#: Advise using --max-scans above this file size, since the default reads everything.
LARGE_LIDAR_BYTES = 4 * 1024**3


def positive_float(text: str) -> float:
    value = float(text)
    if value <= 0.0 or not np.isfinite(value):
        raise argparse.ArgumentTypeError(f"must be a positive number (got {text!r})")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "With no output flag the Rerun viewer is spawned automatically.\n"
            "Use --save scene.rrd to write a recording instead (open it later with\n"
            "`rerun scene.rrd`), or --serve for a browser-based viewer over SSH."
        ),
    )
    parser.add_argument(
        "dataset_path",
        type=Path,
        help="Dataset root holding the stream folders (cam0/, imu0/, lidar0/, ...)",
    )

    streams = parser.add_argument_group("streams")
    streams.add_argument("--no-images", action="store_true", help="skip the image streams")
    streams.add_argument("--no-imu", action="store_true", help="skip the IMU")
    streams.add_argument("--no-lidar", action="store_true",
                         help="skip the lidar (the only slow part)")

    lidar = parser.add_argument_group("lidar")
    lidar.add_argument(
        "--lidar-frequency", type=positive_float, default=10.0, metavar="HZ",
        help="rotation rate; all points within one 1/HZ interval form one point cloud "
             "(default: 10)",
    )
    lidar.add_argument("--max-scans", type=int, default=None,
                       help="stop after N scans (default: read the whole file)")
    lidar.add_argument("--lidar-color", default="intensity", choices=LIDAR_COLOR_MODES,
                       help="per-point colouring (default: intensity)")

    rr.script_add_args(parser)
    return parser.parse_args()


def set_time(timestamp_ns: int) -> None:
    """Move the recording cursor to ``timestamp_ns`` on the dataset clock.

    ``np.timedelta64`` is the only form that carries raw nanoseconds through exactly:
    Rerun reads a bare int or float ``duration`` as *seconds*, and rejects ``timestamp``
    values this large as a suspected unit mix-up.
    """
    rr.set_time(bp.TIMELINE, duration=np.timedelta64(int(timestamp_ns), "ns"))


def time_column(timestamps_ns: np.ndarray) -> rr.TimeColumn:
    """Columnar equivalent of :func:`set_time` for bulk sends."""
    return rr.TimeColumn(bp.TIMELINE, duration=timestamps_ns.astype("timedelta64[ns]"))


def log_static_scene(with_lidar: bool) -> None:
    """Declare frame orientations so the 3D view starts the right way up."""
    rr.log(bp.WORLD, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    if with_lidar:
        # Logged on the lidar entity itself because that entity is a view origin.
        rr.log(bp.LIDAR, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)


def log_imu(imu: euroc.ImuData) -> None:
    """Log both IMU triplets as scalar series, one columnar call per axis.

    ``send_columns`` ships a whole series in one call; the per-sample ``rr.log``
    equivalent would be hundreds of thousands of round trips on the larger datasets.
    """
    if len(imu) == 0:
        return

    times = [time_column(imu.timestamps_ns)]
    plots = (
        (bp.IMU_ACCEL_PLOT, "a", imu.accel),
        (bp.IMU_GYRO_PLOT, "w", imu.gyro),
    )
    for entity, prefix, values in plots:
        for axis_index, axis in enumerate("xyz"):
            path = f"{entity}/{axis}"
            rr.log(
                path,
                rr.SeriesLines(
                    names=[f"{prefix}_{axis}"],
                    colors=[AXIS_COLORS[axis_index]],
                    widths=[1.0],
                ),
                static=True,
            )
            rr.send_columns(
                path,
                indexes=times,
                columns=rr.Scalars.columns(scalars=values[:, axis_index]),
            )


def scan_colors(scan: euroc.LidarScan, mode: str, rings: int) -> np.ndarray | None:
    """Per-point colours for one scan, or ``None`` for uniform white.

    Every mapping is fixed and absolute, so a given value always yields the same colour
    -- across scans and across datasets -- and no value is clipped.
    """
    if mode == "none":
        return None
    if mode == "intensity":
        return colormap.turbo(
            colormap.saturating(scan.intensity, colormap.INTENSITY_SCALE)
        )
    if mode == "ring":
        return colormap.turbo(colormap.fraction(scan.ring, rings))
    if mode == "z":
        return colormap.turbo(colormap.symmetric(scan.xyz[:, 2], colormap.HEIGHT_SCALE))
    if mode == "range":
        return colormap.turbo(
            colormap.saturating(np.linalg.norm(scan.xyz, axis=1), colormap.RANGE_SCALE)
        )
    raise ValueError(f"unknown lidar colour mode: {mode!r}")


def log_scan(scan: euroc.LidarScan, mode: str, rings: int) -> None:
    """Log one scan. Each replaces the last, so only the current one is shown."""
    set_time(scan.timestamp_ns)
    rr.log(
        bp.LIDAR_POINTS,
        rr.Points3D(
            scan.xyz, colors=scan_colors(scan, mode, rings), radii=[POINT_RADIUS_UI]
        ),
    )


def main() -> int:
    args = parse_args()

    try:
        dataset = euroc.discover(args.dataset_path)
    except NotADirectoryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if dataset.is_empty():
        print(f"error: no sensor streams found in {args.dataset_path}", file=sys.stderr)
        return 2

    period_ns = int(round(euroc.NS_PER_S / args.lidar_frequency))

    image_dirs = {} if args.no_images else dict(dataset.images)
    imu_csv = None if args.no_imu else dataset.imu_csv
    lidar_csv = None if args.no_lidar else dataset.lidar_csv

    # ---- read the cheap streams ----------------------------------------------------
    try:
        images = {
            name: euroc.read_image_index(path) for name, path in image_dirs.items()
        }
        imu = euroc.read_imu(imu_csv) if imu_csv is not None else None
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # ---- lidar: only if it really is a point cloud ----------------------------------
    lidar_reader = None
    if lidar_csv is not None:
        candidate = euroc.LidarScanReader(lidar_csv, period_ns=period_ns)
        try:
            columns = candidate.column_count()
        except OSError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if columns >= euroc.LIDAR_MIN_COLUMNS:
            lidar_reader = candidate
        else:
            print(
                f"warning: {lidar_csv} has {columns} columns, not a point cloud "
                f"(expected at least {euroc.LIDAR_MIN_COLUMNS}: timestamp,x,y,z,"
                f"intensity,ring) -- skipping",
                file=sys.stderr,
            )

    # ---- report the timeline extent before logging anything -------------------------
    spans: list[tuple[int, int]] = []
    for index in images.values():
        if len(index):
            spans.append((int(index.timestamps_ns[0]), int(index.timestamps_ns[-1])))
    if imu is not None and len(imu):
        spans.append((int(imu.timestamps_ns[0]), int(imu.timestamps_ns[-1])))
    if lidar_reader is not None:
        try:
            spans.append(lidar_reader.time_range())
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    if not spans:
        print("error: every selected stream is empty", file=sys.stderr)
        return 2

    start_ns = min(span[0] for span in spans)
    end_ns = max(span[1] for span in spans)
    print(
        f"{dataset.root}\n"
        f"  timeline {start_ns / euroc.NS_PER_S:.3f} .. {end_ns / euroc.NS_PER_S:.3f} s "
        f"({(end_ns - start_ns) / euroc.NS_PER_S:.1f} s)"
    )
    if lidar_reader is not None:
        size = lidar_reader.path.stat().st_size
        print(
            f"  lidar    {size / 1e9:.1f} GB, {args.lidar_frequency:g} Hz "
            f"-> {period_ns / 1e6:g} ms per scan"
        )
        if size > LARGE_LIDAR_BYTES and args.max_scans is None:
            print(
                f"warning: reading all of {lidar_reader.path.name} "
                f"({size / 1e9:.0f} GB); pass --max-scans N to stop early",
                file=sys.stderr,
            )

    # ---- connect to Rerun ----------------------------------------------------------
    layout = bp.build(
        list(images),
        with_lidar=lidar_reader is not None,
        with_imu=imu is not None and len(imu) > 0,
    )
    rr.script_setup(args, "euroc_viz", default_blueprint=layout)
    # Force the layout active rather than relying on default_blueprint alone. The viewer
    # caches a blueprint per application id, and that cache wins over a default -- so a
    # layout remembered from a dataset with different streams would otherwise shadow this
    # one, leaving views pointing at entities that do not exist here.
    rr.send_blueprint(layout, make_active=True, make_default=True)
    started = time.perf_counter()

    log_static_scene(with_lidar=lidar_reader is not None)

    # ---- IMU -----------------------------------------------------------------------
    if imu is not None and len(imu):
        log_imu(imu)
        print(
            f"  imu0     {len(imu):>7} samples   "
            f"mean |a| = {np.linalg.norm(imu.accel, axis=1).mean():.2f} m/s^2"
        )

    # ---- image streams -------------------------------------------------------------
    for name, index in images.items():
        for timestamp_ns, path in zip(index.timestamps_ns, index.paths, strict=True):
            set_time(int(timestamp_ns))
            rr.log(bp.image_entity(name), rr.EncodedImage(path=path))
        note = f"   ({index.missing} image files missing)" if index.missing else ""
        print(f"  {name:<8} {len(index):>7} frames{note}")

    # ---- lidar ---------------------------------------------------------------------
    if lidar_reader is not None:
        scan_count = 0
        point_count = 0
        invalid_count = 0
        row_count = 0
        rings = 1
        lidar_started = time.perf_counter()

        for scan in lidar_reader.scans(max_scans=args.max_scans):
            if scan_count == 0 and len(scan):
                # Fix the ring-colour domain from the first scan and keep it for the whole
                # run, so a given ring index keeps one colour. Every azimuth firing covers
                # all rings, so even a partial first interval sees the full set.
                rings = int(scan.ring.max()) + 1
            log_scan(scan, args.lidar_color, rings)
            scan_count += 1
            point_count += len(scan)
            invalid_count += scan.num_invalid
            row_count += scan.num_rows

            if scan_count % 100 == 0:
                elapsed = time.perf_counter() - lidar_started
                print(
                    f"    ...{scan_count} scans, {point_count / 1e6:.1f}M points "
                    f"({elapsed:.0f}s, {elapsed / scan_count * 1000:.0f} ms/scan)",
                    flush=True,
                )

        dropped = (
            f", dropped {invalid_count / row_count * 100:.0f}% (0,0,0) returns"
            if row_count and invalid_count
            else ""
        )
        print(
            f"  lidar0   {scan_count:>7} scans    "
            f"{point_count / 1e6:.1f}M points{dropped}"
        )

    print(f"done in {time.perf_counter() - started:.1f}s")

    # Teardown flushes the sink, so the recording is only complete after this call.
    rr.script_teardown(args)

    if getattr(args, "save", None):
        size_mb = Path(args.save).stat().st_size / 1e6
        print(f"wrote {args.save} ({size_mb:.0f} MB) -- open it with:  rerun {args.save}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
