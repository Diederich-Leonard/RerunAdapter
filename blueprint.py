"""Entity paths and the default Rerun view layout.

The entity hierarchy is deliberately the *final* one, even though this iteration logs no
transforms at all::

    /world                        static ViewCoordinates (z up)
    /world/body                   [later] Transform3D per pose, from a trajectory
    /world/body/<stream>          [later] static Transform3D (T_SC) + Pinhole
    /world/body/<stream>/image    EncodedImage per frame, one entity per image stream
    /world/body/lidar0            static ViewCoordinates; [later] Transform3D (T_SL)
    /world/body/lidar0/points     Points3D per scan, in raw sensor coordinates
    /plots/imu/accel/{x,y,z}      whole series, sent columnar
    /plots/imu/gyro/{x,y,z}       "

With no ``Transform3D`` logged, every frame collapses to identity, so a 3D view whose
**origin** is ``/world/body/lidar0`` is exactly a lidar-fixed frame -- Rerun renders a
spatial view relative to its origin and ignores transforms at or above it. That is why
the same view keeps working untouched once a trajectory arrives and the sensor starts
moving: points are always stored in raw sensor coordinates, and only the *view* decides
which frame you look from.

Consequently, adding a results overlay later means adding transforms and new sibling
entities (``/world/trajectory``, ``/world/mesh/...``) plus a second 3D view rooted at
``/world``. No existing path moves.

The IMU plots sit outside ``/world`` so a future world view's default ``$origin/**``
contents never sweeps up non-spatial entities.
"""

from __future__ import annotations

import math

import rerun.blueprint as rrb

#: The one timeline everything is logged against. Dataset clocks are elapsed or epoch
#: nanoseconds rather than wall-clock dates, so it is logged as a *duration*.
TIMELINE = "sensor_time"

WORLD = "/world"
BODY = f"{WORLD}/body"
LIDAR = f"{BODY}/lidar0"
LIDAR_POINTS = f"{LIDAR}/points"
IMU_PLOTS = "/plots/imu"
IMU_ACCEL_PLOT = f"{IMU_PLOTS}/accel"
IMU_GYRO_PLOT = f"{IMU_PLOTS}/gyro"


def stream_entity(name: str) -> str:
    """Entity holding one image stream's frame of reference (extrinsics land here)."""
    return f"{BODY}/{name}"


def image_entity(name: str) -> str:
    """Entity holding one image stream's images."""
    return f"{stream_entity(name)}/image"


def build(
    streams: list[str],
    *,
    with_lidar: bool = True,
    with_imu: bool = True,
) -> rrb.Blueprint:
    """Image views on the left; the lidar view and IMU plots on the right.

    Adapts to however many image streams the dataset has, and drops absent streams from
    the layout rather than leaving empty panels.
    """
    right: list[rrb.BlueprintPart] = []
    if with_lidar:
        right.append(
            rrb.Spatial3DView(
                origin=LIDAR,  # <- the lidar-fixed frame
                name="lidar0 (sensor frame)",
                line_grid=rrb.LineGrid3D(visible=True),
            )
        )
    if with_imu:
        right.append(
            rrb.TimeSeriesView(
                origin=IMU_ACCEL_PLOT,
                name="acceleration [m/s^2]",
                plot_legend=rrb.PlotLegend(corner=rrb.Corner2D.RightBottom),
            )
        )
        right.append(
            rrb.TimeSeriesView(
                origin=IMU_GYRO_PLOT,
                name="angular rate [rad/s]",
                plot_legend=rrb.PlotLegend(corner=rrb.Corner2D.RightBottom),
            )
        )

    panels: list[rrb.BlueprintPart] = []
    if streams:
        panels.append(
            rrb.Grid(
                contents=[
                    rrb.Spatial2DView(origin=image_entity(name), name=name)
                    for name in streams
                ],
                # Keep the tiles roughly square whatever the stream count is.
                grid_columns=max(1, math.ceil(math.sqrt(len(streams)))),
                name="images",
            )
        )
    if right:
        if len(right) > 1:
            # Give the 3D view more height than the plots below it.
            shares = [2] + [1] * (len(right) - 1) if with_lidar else [1] * len(right)
            panels.append(rrb.Vertical(*right, row_shares=shares))
        else:
            panels.append(right[0])

    if not panels:
        return rrb.Blueprint(collapse_panels=False)
    if len(panels) == 1:
        return rrb.Blueprint(panels[0], collapse_panels=False)
    return rrb.Blueprint(
        rrb.Horizontal(*panels, column_shares=[3, 2]),
        collapse_panels=False,
    )
