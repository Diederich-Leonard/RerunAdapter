# EuRoC/ASL dataset → Rerun visualizer

Streams a dataset in EuRoC/ASL layout into the [Rerun](https://rerun.io) viewer.

## Install

```bash
uv venv
uv pip install -r requirements.txt
source .venv/bin/activate
```

## Usage

All commands below assume the venv is activated.

```bash
python3 ./viz.py <dataset_path>
```

The dataset root is the folder holding the stream directories.

Add `--results` and `--config` to overlay a SLAM run:

```bash
python3 ./viz.py <dataset_path> \
    --results ../../results \
    --config ../../src/okvis2-x-private/config/elios3-eth/okvis2.yaml
```

That adds a world-frame 3D view with the estimated trajectory, the submap meshes, every
camera as a posed pinhole frustum, the lidar carried into world coordinates, and each frame
as a coordinate triad. `--results` picks the most refined `*_trajectory.csv` it finds
(`-final-ba` over `-final` over the realtime one); `--trajectory FILE` overrides that.

`--trajectory` also works on its own, and takes a reference file as readily as an OKVIS CSV,
so a dataset can be viewed moving through the world before any SLAM output exists:

```bash
python3 ./viz.py <dataset_path> --trajectory <groundtruth.txt> --config <okvis2.yaml>
```

The viewer is spawned automatically. Over SSH use `--serve`, or write a file and open it
locally:

```bash
python3 ./viz.py <dataset_path> --save scene.rrd
rerun scene.rrd
```

## Entity hierarchy

Rerun composes transforms down the entity path, so every sensor is logged in its own native
frame and the viewer does the placing — no point cloud or mesh is transformed by hand.

```
/world                        static ViewCoordinates (z up)
/world/axes                   static Arrows3D           -> world frame triad
/world/trajectory             static LineStrips3D       -> the whole estimated path
/world/groundtruth            static LineStrips3D       -> reference path, aligned (green)
/world/mesh/<name>            static Mesh3D             -> vertices already in world frame
/world/imu                    Transform3D per pose      -> T_WS from the trajectory
/world/imu/body               static Transform3D        -> T_SB, i.e. inverse of T_BS
/world/imu/<stream>           static Transform3D        -> T_SC for that camera
/world/imu/<stream>/image     static Pinhole + EncodedImage per frame
/world/imu/lidar0             static Transform3D        -> T_SL
/world/imu/lidar0/points      Points3D per scan, in raw sensor coordinates
/plots/imu/accel/{x,y,z}      scalar series, columnar   -> view "acceleration [m/s^2]"
/plots/imu/gyro/{x,y,z}                                 -> view "angular rate [rad/s]"
```

The IMU frame `S` is the moving root, because an OKVIS calibration expresses both cameras
(`T_SC`) and lidar (`T_SL`) relative to the IMU, and the trajectory gives the IMU's pose in
the world (`T_WS`). `T_BS` runs the other way, so the body frame is placed with its inverse.

Two consequences:

- **The lidar is logged once and viewable in two frames.** A spatial view renders relative
  to its origin and ignores transforms at or above it, so the view rooted at
  `/world/imu/lidar0` is lidar-fixed, while the one rooted at `/world` shows the same points
  carried out by `T_SL` then `T_WS`. They share a tab strip.
- **Without a trajectory or config nothing changes structurally.** No `Transform3D` is
  logged, every frame collapses to identity, and the same paths still work — which is what
  lets both modes share one hierarchy.

Poses are interpolated (lerp + slerp) onto the timestamp of whatever is being placed. The
trajectory runs at the estimator's rate, so without that a lidar scan would inherit a pose
up to one estimator period stale, offsetting the whole cloud.

## Lidar frequency

Every row of the lidar CSV is one point. Points are grouped into scans **by time**: all
points whose timestamp falls in the same `1 / --lidar-frequency` interval form one point
cloud.

A `lidar*` folder whose CSV has fewer than six columns is not a point cloud and is skipped
with a warning

## Ground truth alignment

`--groundtruth` takes a space-separated reference file (`timestamp tx ty tz ...`, timestamps
in **seconds**, header optionally `#`-commented) and draws it in green next to the amber
estimate.

The two do not share a frame, so they are aligned as follows:
nearest-timestamp association subsampling the longer stream, then a position-only rigid
Umeyama/Horn fit with no scale. The resulting transform is applied **inverted** — moving the
reference onto the estimate rather than the other way round — so the estimated poses, the
meshes and the lidar stay exactly where they were logged and only the reference line moves.
The fit is identical either way.

## Colours are fixed and absolute

A given value always produces the same colour — across scans and across datasets. Nothing
is stretched to the current min/max and nothing is clipped. Since intensity is unbounded
in practice (0–206 in one dataset, up to 1758 in another), the unbounded quantities use a
monotonic normalisation over their whole domain rather than a linear window with cutoffs:

| Mode | Mapping | Midpoint of the ramp |
|---|---|---|
| `intensity` | `v / (v + 32)` | 32 |
| `range` | `r / (r + 20)` | 20 m |
| `z` | `0.5 + 0.5·tanh(z / 5)` | 0 m, always mid-colormap |
| `ring` | `ring / (rings − 1)` | ring count read from the first scan, then held fixed |
| `none` | uniform white | — |

## Options

| Flag | Meaning |
|---|---|
| `--results DIR` | result directory with `*_trajectory.csv` and `mesh_*.ply` |
| `--config FILE` | OKVIS config supplying `T_SC`, `T_SL`, `T_BS` |
| `--trajectory FILE` | explicit trajectory, instead of the one picked from `--results` |
| `--groundtruth FILE` | reference trajectory, rigidly aligned before plotting |
| `--lidar-frequency HZ` | scan interval is 1/HZ (default `10`) |
| `--max-scans N` | stop after N scans (default: whole file) |
| `--lidar-color` | `intensity` (default), `ring`, `z`, `range`, `none` |
| `--no-images`, `--no-imu`, `--no-lidar`, `--no-meshes` | skip a stream |
| `--save F.rrd`, `--serve`, `--connect`, `--headless` | standard Rerun output flags |

## Code layout

```
viz.py         CLI, the logging loop, and the only Rerun logging calls
euroc.py       stream discovery + readers (images, IMU, streaming lidar) — no Rerun
results.py     trajectory and submap mesh readers, pose interpolation — no Rerun
calib.py       OKVIS config -> intrinsics and T_SC / T_SL / T_BS — no Rerun
blueprint.py   entity paths and the default view layout
colormap.py    turbo table and the fixed value->colour mappings
```

## Expected input layout

```
<root>/<stream>/data.csv      #timestamp [ns],filename
<root>/<stream>/data/<ts>.png|jpg
<root>/imu0/data.csv          #timestamp [ns],w_RS_S_{xyz},a_RS_S_{xyz}   (gyro first)
<root>/lidar0/data.csv        #timestamp [ns],x,y,z,intensity,ring
```

Any folder containing both a `data.csv` and a `data/` subfolder is treated as an image
stream, whatever it is named. Grayscale and colour are handled identically because the
encoded bytes are passed through untouched and the viewer decodes them.

## Known limitations

- **Without `--config` there is no rig geometry.** No transforms are logged, so all frames
  are identity: image streams are 2D views only (no frusta), and the lidar cannot be placed
  in the world. Calibration generally lives outside these datasets.
- **Camera frusta follow the discovered image streams**, paired with the config's cameras in
  order, so `--no-images` also removes them. A config with a different camera count than the
  dataset has streams pairs the first *n* and warns.
- **`rr.Pinhole` carries no distortion model.** The frusta and any 2D overlay assume a plain
  pinhole while the logged pixels are still distorted, which these configs are
  (`distortion_type: equidistant`).
- **A config without `lidar.T_SL` leaves the lidar at the IMU frame** and warns; not every
  OKVIS config has that section.
- **No motion compensation within a scan**, and the whole scan is logged at the start of
  its interval.
- **The IMU is not drawn in the 3D view**, only as the frame triad and the two graphs.
