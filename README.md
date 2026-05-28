# Alliance-Interface-Inspection

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/numpy-%E2%89%A51.23-013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![Plotly](https://img.shields.io/badge/plotly-%E2%89%A55.18-3F4F75.svg)](https://plotly.com/python/)
[![NetworkX](https://img.shields.io/badge/networkx-%E2%89%A53.0-008080.svg)](https://networkx.org/)
[![rosbags](https://img.shields.io/badge/rosbags-%E2%89%A50.11-3775A9.svg)](https://pypi.org/project/rosbags/)

Inspect ROS 1 / ROS 2 bag files with **[rosbags](https://pypi.org/project/rosbags/)**: printed topic listing, summaries for standard ROS message types, and **interactive Plotly** charts in your browser (IMU, point clouds, lidar scans, TF trees).

Supports ROS 1 **`.bag`** files, ROS 2 bag **folders** (with `metadata.yaml`), and **`.db3` / `.mcap`** when you pass the appropriate path.

## Install and Run

You need **Python 3.10+**. On Debian/Ubuntu, if creating a virtual environment fails, install **`python3-venv`** first.

Open a terminal:

```bash
cd /path/to/KJ_ros_interface
python3 -m venv .venv
source .venv/bin/activate          # Linux / macOS

# Windows (Command Prompt):  .venv\Scripts\activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

cd bag_inspection_metrics
python source_data_inspection.py -f /path/to/rosbag_folder --preview html
```

Replace **`/path/to/KJ_ros_interface`** with the folder where you unpacked this repo, and **`/path/to/rosbag_folder`** with your bag directory or file path.

`-f` accepts a `.bag` file, an ROS 2 bag directory, or a `.db3` / `.mcap` path.

### `--preview` (`html` is the default)

| Value | Effect |
|--------|--------|
| `html` | Saves an HTML file and opens it in your browser (scroll and zoom the charts). |
| `plotly` | Tries to open each chart in a Plotly viewer (may not work on headless machines). |
| `both` | Browser HTML and Plotly viewer. |

## Plot Info

### IMU (`sensor_msgs/msg/Imu`)

For each IMU topic, one panel plots **angular rate** (solid lines, rad/s on the left axis) and **linear acceleration** (dashed lines, m/s² on the right axis) against **recording time**.

### Point cloud (`sensor_msgs/msg/PointCloud2`)

For each lidar/cloud topic there are usually **two panels**:

1. **Over time:** **points per message**, **ROS height**, and **ROS width** versus **bag recording timestamps** on the horizontal axis (playback order—not always the same as stamped sensor time).
2. **Histogram:** how often each **point‑count** bucket appears across the whole recording.

Those two panels use **extra vertical spacing** between them.

### Laser scan (`sensor_msgs/msg/LaserScan`)

Each scan topic is drawn as **many translucent rings**: each ROS message contributes one polar scan unfolded into **x / y meters**, so overlapping scans show density over the sampled part of the bag.

### TF transforms (`tf2_msgs/msg/TFMessage`)

**`/tf`** (moving transforms) and **`/tf_static`** (fixed transforms) each get one panel with a **frame tree**—**parent frame → child frame**. Dynamic **`/tf`** reads only **the first portion of messages** so very long bags stay responsive; **`/tf_static`** uses the full topic.

### What is *not* drawn here

Only some “standard ROS” topics have previews. Standard topics without a viewer are listed under **Std-topic plot notes** in the terminal.

## Licence / status

Experimental tooling used as-is.
