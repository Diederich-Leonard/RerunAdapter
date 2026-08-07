#!/usr/bin/env python3
"""Visualise a dataset in EuRoC/ASL layout in Rerun, optionally against SLAM results.

Two modes:

*Dataset only* -- point it at a dataset root. Every image stream gets its own 2D view, the
lidar is shown in a lidar-fixed 3D view, and the IMU is split into two graphs.

*With results* -- add ``--results`` and ``--config`` to overlay a SLAM run. A world-frame 3D
view then shows the estimated trajectory, the submap meshes, every camera as a posed pinhole
frustum, the lidar carried into world coordinates, and each frame (world, IMU, body,
cameras, lidar) as a coordinate triad.

Nothing about the sensor count is hardcoded: image streams are discovered from the directory
layout, so two cameras or five work the same way. One IMU and one lidar are assumed.

This is the only module that talks to Rerun's logging API; ``euroc.py`` reads the dataset,
``results.py`` the trajectory and meshes, ``calib.py`` the extrinsics, and ``blueprint.py``
owns the entity paths and the view layout.

Usage:
    python viz.py <dataset_path>
    python viz.py <dataset_path> --max-scans 20
    python viz.py <dataset_path> --results <results_dir> --config <okvis2.yaml>
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
import calib
import colormap
import euroc
import results

#: Point size in UI points. Rerun reads a negative radius as a screen-space size, which
#: keeps a sparse cloud visible whether you are 1 m or 100 m from it.
POINT_RADIUS_UI = -1.0

#: Per-axis colours, shared by the IMU plots and the coordinate triads so x/y/z read alike.
AXIS_COLORS = ((230, 80, 80), (90, 200, 110), (90, 150, 240))

#: Coordinate triad sizes in metres: the world origin gets a longer one so it stands out.
FRAME_AXIS_LENGTH = 0.5
WORLD_AXIS_LENGTH = 0.5

#: How far in front of each camera to draw its frustum, in metres.
IMAGE_PLANE_DISTANCE = 1.0

TRAJECTORY_COLOR = (255, 190, 60)
GROUNDTRUTH_COLOR = (80, 220, 120)
TRAJECTORY_RADIUS = 0.1

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

    overlay = parser.add_argument_group("SLAM results")
    overlay.add_argument(
        "--results", type=Path, default=None, metavar="DIR",
        help="result directory holding a *_trajectory.csv and mesh_*.ply files",
    )
    overlay.add_argument(
        "--config", type=Path, default=None, metavar="FILE",
        help="OKVIS config supplying the extrinsics (T_SC per camera, T_SL, T_BS)",
    )
    overlay.add_argument(
        "--trajectory", type=Path, default=None, metavar="FILE",
        help="trajectory to move the sensors along, instead of the one picked from "
             "--results; accepts an OKVIS CSV or a reference file, and works on its own",
    )
    overlay.add_argument(
        "--groundtruth", type=Path, default=None, metavar="FILE",
        help="reference trajectory (space separated, timestamps in seconds); rigidly "
             "aligned to the estimate before plotting",
    )

    streams = parser.add_argument_group("streams")
    streams.add_argument("--no-images", action="store_true", help="skip the image streams")
    streams.add_argument("--no-imu", action="store_true", help="skip the IMU")
    streams.add_argument("--no-lidar", action="store_true",
                         help="skip the lidar (the only slow part)")
    streams.add_argument("--no-meshes", action="store_true", help="skip the submap meshes")

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


# ----------------------------------------------------------------------------------
# Time
# ----------------------------------------------------------------------------------


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


# ----------------------------------------------------------------------------------
# Frames
# ----------------------------------------------------------------------------------


def log_transform(entity: str, T: np.ndarray) -> None:
    """Log a static 4x4 transform mapping this entity's frame into its parent's."""
    rr.log(
        entity,
        rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3]),
        static=True,
    )


def log_frame_axes(frame: str, length: float = FRAME_AXIS_LENGTH) -> None:
    """Draw a frame's coordinate triad.

    Rerun 0.35's ``Transform3D`` has no ``axis_length``, so the axes are drawn explicitly.
    They go on a child entity, which both keeps them separately toggleable and lets them
    inherit the frame's transform, so the triad follows its frame for free.
    """
    rr.log(
        bp.axes_entity(frame),
        rr.Arrows3D(
            vectors=np.eye(3) * length,
            origins=np.zeros((3, 3)),
            colors=list(AXIS_COLORS),
            radii=[length * 0.03],
        ),
        static=True,
    )


def log_pose(timestamp_ns: int, position: np.ndarray, quaternion_xyzw: np.ndarray) -> None:
    """Place the IMU frame S in the world at ``timestamp_ns``."""
    set_time(timestamp_ns)
    rr.log(bp.IMU, rr.Transform3D(translation=position,
                                  quaternion=rr.Quaternion(xyzw=quaternion_xyzw)))


def log_calibration(
    calibration: calib.Calibration, streams: list[str], *, with_lidar: bool
) -> None:
    """Log the static rig: body, camera and lidar frames relative to the IMU."""
    log_frame_axes(bp.IMU)

    # The config gives T_BS, which maps IMU coordinates into the body frame. Hanging the
    # body frame under the IMU needs the opposite direction.
    log_transform(bp.BODY, calib.invert(calibration.T_BS))
    log_frame_axes(bp.BODY)

    for name, camera in zip(streams, calibration.cameras):
        entity = bp.stream_entity(name)
        log_transform(entity, camera.T_SC)
        log_frame_axes(entity)
        rr.log(
            bp.image_entity(name),
            rr.Pinhole(
                image_from_camera=camera.K,
                resolution=list(camera.resolution),
                # OKVIS camera frames are x right, y down, z forward.
                camera_xyz=rr.ViewCoordinates.RDF,
                image_plane_distance=IMAGE_PLANE_DISTANCE,
            ),
            static=True,
        )

    if with_lidar:
        if calibration.T_SL is not None:
            log_transform(bp.LIDAR, calibration.T_SL)
        log_frame_axes(bp.LIDAR)


# ----------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------


def log_static_scene(with_lidar: bool) -> None:
    """Declare frame orientations so the 3D views start the right way up."""
    rr.log(bp.WORLD, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    if with_lidar:
        # Logged on the lidar entity itself because that entity is a view origin.
        rr.log(bp.LIDAR, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)


def log_trajectory(trajectory: results.Trajectory) -> None:
    """Log the whole estimated path once, then every pose at its own timestamp."""
    rr.log(
        bp.TRAJECTORY,
        rr.LineStrips3D(
            [trajectory.positions.astype(np.float32)],
            colors=[TRAJECTORY_COLOR],
            radii=[TRAJECTORY_RADIUS],
        ),
        static=True,
    )
    for timestamp_ns, position, quaternion in zip(
        trajectory.timestamps_ns, trajectory.positions, trajectory.quaternions, strict=True
    ):
        log_pose(int(timestamp_ns), position, quaternion)


def log_ground_truth(positions: np.ndarray) -> None:
    """Log the reference trajectory, already carried into the estimate's world frame."""
    rr.log(
        bp.GROUNDTRUTH,
        rr.LineStrips3D(
            [positions.astype(np.float32)],
            colors=[GROUNDTRUTH_COLOR],
            radii=[TRAJECTORY_RADIUS],
        ),
        static=True,
    )


def log_meshes(paths: list[Path]) -> tuple[int, int]:
    """Log every submap mesh as static geometry. Returns ``(meshes, vertices)``.

    The vertices are already in world coordinates, so no transform is applied, and the
    meshes are static so the whole map is present from the first frame onward.
    """
    vertices = 0
    for index, path in enumerate(paths):
        mesh = results.read_mesh_ply(path)
        rr.log(
            bp.mesh_entity(mesh.name),
            rr.Mesh3D(
                vertex_positions=mesh.vertices,
                triangle_indices=mesh.triangles,
                albedo_factor=(175, 175, 175, 40),
                face_rendering="Front"
            ),
            static=True,
        )
        vertices += len(mesh)
    return len(paths), vertices


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


# ----------------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------------


def load_results(
    args: argparse.Namespace,
) -> tuple[results.Trajectory | None, Path | None, list[Path]]:
    """Resolve the trajectory and mesh list from ``--results`` / ``--trajectory``."""
    trajectory_path = args.trajectory
    meshes: list[Path] = []

    if args.results is not None:
        if not args.results.is_dir():
            raise NotADirectoryError(f"{args.results} is not a directory")
        if trajectory_path is None:
            trajectory_path = results.find_trajectory(args.results)
            if trajectory_path is None:
                raise FileNotFoundError(f"no *_trajectory.csv in {args.results}")
        if not args.no_meshes:
            meshes = results.find_meshes(args.results)

    trajectory = (
        results.read_trajectory(trajectory_path) if trajectory_path is not None else None
    )
    return trajectory, trajectory_path, meshes


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

    # ---- results and calibration ----------------------------------------------------
    # Loaded before the timeline is worked out, because a trajectory carries timestamps of
    # its own and may be the only time-varying thing present.
    try:
        trajectory, trajectory_path, mesh_paths = load_results(args)
        calibration = calib.load(args.config) if args.config is not None else None
        ground_truth = (
            results.read_ground_truth(args.groundtruth)
            if args.groundtruth is not None
            else None
        )
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

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
    if trajectory is not None and len(trajectory):
        spans.append((int(trajectory.timestamps_ns[0]), int(trajectory.timestamps_ns[-1])))

    # Meshes and the reference trajectory are static, so they are worth showing even with no
    # time-varying stream at all.
    has_static = bool(mesh_paths) or ground_truth is not None
    if not spans and not has_static:
        print("error: nothing selected to visualise", file=sys.stderr)
        return 2

    print(f"{dataset.root}")
    if spans:
        start_ns = min(span[0] for span in spans)
        end_ns = max(span[1] for span in spans)
        print(
            f"  timeline {start_ns / euroc.NS_PER_S:.3f} .. "
            f"{end_ns / euroc.NS_PER_S:.3f} s "
            f"({(end_ns - start_ns) / euroc.NS_PER_S:.1f} s)"
        )
    if trajectory is not None:
        print(f"  traj     {trajectory_path.name}: {len(trajectory)} poses")
    if mesh_paths:
        print(f"           {len(mesh_paths)} mesh files")
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

    if calibration is not None:
        print(
            f"  config   {args.config.name}: {len(calibration.cameras)} cameras, "
            f"T_SL {'present' if calibration.T_SL is not None else 'absent'}"
        )
        if len(calibration.cameras) != len(images) and images:
            print(
                f"warning: {len(images)} image streams but {len(calibration.cameras)} "
                f"cameras in the config; pairing the first "
                f"{min(len(images), len(calibration.cameras))} in order",
                file=sys.stderr,
            )
        if lidar_reader is not None and calibration.T_SL is None:
            print(
                "warning: the config has no lidar.T_SL, so the lidar is left coincident "
                "with the IMU frame in the world view",
                file=sys.stderr,
            )
    if trajectory is None and (calibration is not None or mesh_paths):
        print(
            "warning: no trajectory, so the rig stays at the world origin",
            file=sys.stderr,
        )

    with_world = (
        trajectory is not None
        or calibration is not None
        or bool(mesh_paths)
        or ground_truth is not None
    )

    # ---- connect to Rerun ----------------------------------------------------------
    layout = bp.build(
        list(images),
        with_lidar=lidar_reader is not None,
        with_imu=imu is not None and len(imu) > 0,
        with_world=with_world,
    )
    rr.script_setup(args, "euroc_viz", default_blueprint=layout)
    # Force the layout active rather than relying on default_blueprint alone. The viewer
    # caches a blueprint per application id, and that cache wins over a default -- so a
    # layout remembered from a dataset with different streams would otherwise shadow this
    # one, leaving views pointing at entities that do not exist here.
    rr.send_blueprint(layout, make_active=True, make_default=True)
    started = time.perf_counter()

    log_static_scene(with_lidar=lidar_reader is not None)
    if with_world:
        log_frame_axes(bp.WORLD, WORLD_AXIS_LENGTH)
    if calibration is not None:
        log_calibration(calibration, list(images), with_lidar=lidar_reader is not None)

    # Poses are interpolated onto the timestamp of whatever is being placed. The trajectory
    # runs at the estimator's rate, so without this a lidar scan would inherit a pose up to
    # one estimator period stale, offsetting the whole cloud in the world view.
    poses = results.PoseInterpolator(trajectory) if trajectory is not None else None

    def place(timestamp_ns: int) -> None:
        if poses is not None:
            position, quaternion = poses.at(timestamp_ns)
            log_pose(timestamp_ns, position, quaternion)

    if trajectory is not None:
        log_trajectory(trajectory)

    if ground_truth is not None:
        if trajectory is not None:
            gt_positions, rmse = results.align_ground_truth(trajectory, ground_truth)
            print(
                f"  gt       {args.groundtruth.name}: {len(ground_truth)} poses, "
                f"aligned to the estimate (ATE RMSE {rmse:.3f} m)"
            )
        else:
            gt_positions = ground_truth.positions
            print(f"  gt       {args.groundtruth.name}: {len(ground_truth)} poses")
            print(
                "warning: no estimated trajectory to align against, so the reference is "
                "drawn in its own frame",
                file=sys.stderr,
            )
        log_ground_truth(gt_positions)

    if mesh_paths:
        count, vertices = log_meshes(mesh_paths)
        print(f"  meshes   {count:>7} submaps   {vertices / 1e6:.1f}M vertices")

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
            place(int(timestamp_ns))
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
            place(scan.timestamp_ns)
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
