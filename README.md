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

The viewer is spawned automatically. Over SSH use `--serve`, or write a file and open it
locally:

```bash
python3 ./viz.py <dataset_path> --save scene.rrd
rerun scene.rrd
```

## Entity hierarchy

```
/world                        static ViewCoordinates (z up)
/world/body/<stream>/image    EncodedImage per frame   -> one 2D view per image stream
/world/body/lidar0/points     Points3D per scan        -> view "lidar0 (sensor frame)"
/plots/imu/accel/{x,y,z}      scalar series, columnar  -> view "acceleration [m/s^2]"
/plots/imu/gyro/{x,y,z}                                -> view "angular rate [rad/s]"
```

## Lidar frequency

Every row of the lidar CSV is one point. Points are grouped into scans **by time**: all
points whose timestamp falls in the same `1 / --lidar-frequency` interval form one point
cloud.

A `lidar*` folder whose CSV has fewer than six columns is not a point cloud and is skipped
with a warning

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
| `--lidar-frequency HZ` | scan interval is 1/HZ (default `10`) |
| `--max-scans N` | stop after N scans (default: whole file) |
| `--lidar-color` | `intensity` (default), `ring`, `z`, `range`, `none` |
| `--no-images`, `--no-imu`, `--no-lidar` | skip a stream |
| `--save F.rrd`, `--serve`, `--connect`, `--headless` | standard Rerun output flags |

## Code layout

```
viz.py         CLI, the logging loop, and the only Rerun logging calls
euroc.py       stream discovery + readers (images, IMU, streaming lidar) — no Rerun
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

- **No extrinsics, so no rig geometry.** No transforms are logged, so all frames are
  identity: image streams are 2D views only (no frusta), and the lidar cannot be placed in
  a world frame. Calibration generally lives outside these datasets.
- **No motion compensation within a scan**, and the whole scan is logged at the start of
  its interval.
