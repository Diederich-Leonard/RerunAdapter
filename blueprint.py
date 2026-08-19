"""Entity paths and the default Rerun view layout.

Rerun composes transforms down the entity path, so every sensor is logged in its own native
frame and the viewer does the placing -- no point cloud or mesh is transformed by hand.
Because an OKVIS calibration expresses both cameras and lidar relative to the IMU, and a
trajectory gives the IMU's pose in the world, the IMU frame S is the natural moving root::

    /world                        static ViewCoordinates (z up)
    /world/trajectory             static LineStrips3D -- the whole estimated path
    /world/mesh/<name>            static Mesh3D -- vertices already in world coordinates
    /world/imu                    Transform3D per pose: T_WS, from the trajectory
    /world/imu/body               static Transform3D: T_SB, i.e. inverse of the config T_BS
    /world/imu/<stream>           Transform3D per pose: T_SC for a *static* camera (one not
                                  named in --camera-pose)
    /world/<stream>               Transform3D per pose: T_WC for a camera named in
                                  --camera-pose, rooted directly under /world since its pose
                                  file already places it in the world frame
    /world/<stream>/image         static Pinhole + Image (undistorted) or EncodedImage per frame
    /world/imu/lidar0             static Transform3D: T_SL
    /world/imu/lidar0/points      Points3D per scan, in raw sensor coordinates
    /plots/imu/accel/{x,y,z}      whole series, sent columnar
    /plots/imu/gyro/{x,y,z}       "

Two consequences worth spelling out:

* **Without a trajectory or calibration, nothing changes structurally.** No ``Transform3D``
  is logged, every frame collapses to identity, and the same paths still work -- which is
  what lets dataset-only mode and results mode share one hierarchy.
* **The lidar is logged once and viewable in two frames.** A spatial view renders relative
  to its origin and ignores transforms at or above it, so a view rooted at
  ``/world/imu/lidar0`` is lidar-fixed, while one rooted at ``/world`` shows the same points
  carried into the world by ``T_SL`` then ``T_WS``. No duplicated data.

The IMU scalar plots sit outside ``/world`` so the world view's default ``$origin/**``
contents never sweeps up non-spatial entities.
"""

from __future__ import annotations

import math

import rerun.blueprint as rrb

#: The one timeline everything is logged against. Dataset clocks are elapsed or epoch
#: nanoseconds rather than wall-clock dates, so it is logged as a *duration*.
TIMELINE = "sensor_time"

WORLD = "/world"
TRAJECTORY = f"{WORLD}/trajectory"
GROUNDTRUTH = f"{WORLD}/groundtruth"
MESHES = f"{WORLD}/mesh"
IMU = f"{WORLD}/imu"
BODY = f"{IMU}/body"
LIDAR = f"{IMU}/lidar0"
LIDAR_POINTS = f"{LIDAR}/points"
IMU_PLOTS = "/plots/imu"
IMU_ACCEL_PLOT = f"{IMU_PLOTS}/accel"
IMU_GYRO_PLOT = f"{IMU_PLOTS}/gyro"


def stream_entity(name: str, *, moving: bool = False) -> str:
    """Entity holding one image stream's frame of reference (its extrinsics land here).

    A camera named in ``--camera-pose`` (``moving=True``) roots directly under ``/world``,
    since its pose file already gives ``T_WC``; a static camera stays under ``/world/imu``,
    since its config ``T_SC`` still needs to compose with the trajectory's ``T_WS``.
    """
    return f"{WORLD}/{name}" if moving else f"{IMU}/{name}"


def image_entity(name: str, *, moving: bool = False) -> str:
    """Entity holding one image stream's pinhole model and images."""
    return f"{stream_entity(name, moving=moving)}/image"


def mesh_entity(name: str) -> str:
    return f"{MESHES}/{name}"


def build(
    streams: list[str],
    *,
    with_lidar: bool = True,
    with_imu: bool = True,
    with_world: bool = False,
    moving: frozenset[str] = frozenset(),
) -> rrb.Blueprint:
    """Image views on the left; 3D views and IMU plots on the right.

    ``with_world`` adds the world-frame 3D view that shows the trajectory, meshes, camera
    frusta and the lidar carried into world coordinates. It shares a tab strip with the
    lidar-fixed view rather than taking its own panel, since the two answer different
    questions about the same data and are rarely wanted side by side.

    ``moving`` names the streams whose images live under the world-rooted entity path
    instead of the IMU-rooted one, so their 2D view points at the entity they're actually
    logged to.
    """

    panels: list[rrb.BlueprintPart] = []

    views_3d: list[rrb.BlueprintPart] = []
    if with_world:
        views_3d.append(
            rrb.Spatial3DView(
                origin=WORLD,
                name="world",
                line_grid=rrb.LineGrid3D(visible=True),
            )
        )
    if with_lidar:
        views_3d.append(
            rrb.Spatial3DView(
                origin=LIDAR,  # <- the lidar-fixed frame
                name="lidar0 (sensor frame)",
                line_grid=rrb.LineGrid3D(visible=True),
            )
        )

    imu: list[rrb.BlueprintPart] = []
    if with_imu:
        imu.append(
            rrb.TimeSeriesView(
                origin=IMU_ACCEL_PLOT,
                name="acceleration [m/s^2]",
                plot_legend=rrb.PlotLegend(corner=rrb.Corner2D.RightBottom),
            )
        )
        imu.append(
            rrb.TimeSeriesView(
                origin=IMU_GYRO_PLOT,
                name="angular rate [rad/s]",
                plot_legend=rrb.PlotLegend(corner=rrb.Corner2D.RightBottom),
            )
        )

    # Both 3D views show the same points in different frames, so they share a tab strip
    # rather than halving each other's width.
    spatial = (
        None
        if not views_3d
        else views_3d[0]
        if len(views_3d) == 1
        else rrb.Tabs(*views_3d, name="3D")
    )

    if spatial is not None and not imu:
        panels.append(spatial)
    elif spatial is not None and imu:
        panels.append(rrb.Vertical(spatial, rrb.Horizontal(*imu), row_shares=[4, 1]))
    elif imu:
        panels.append(rrb.Vertical(*imu))

    if streams:
        panels.append(
            rrb.Vertical(
                contents=[
                    rrb.Spatial2DView(origin=image_entity(name, moving=name in moving), name=name)
                    for name in streams
                ],
                name="images",
            )
        )

    if not panels:
        return rrb.Blueprint(collapse_panels=False)
    if len(panels) == 1:
        return rrb.Blueprint(panels[0], collapse_panels=False)
    return rrb.Blueprint(
        rrb.Horizontal(*panels, column_shares=[4, 1]),
        collapse_panels=False,
    )
