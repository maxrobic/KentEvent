# KentEvent

Event-based spherical tracking with the Kent (FB5) distribution.

This repository contains the code for the Kent filter used in our paper:
- `cpp/`: a C++ tracker (`KentTracker`) that replays a pair of synchronized event-camera recordings, lifts sparse optical flow onto the unit sphere and tracks the target bearing and heading with a Kent filter.
- `matlab/`: a MATLAB implementation of the Kent distribution and the Kent filter.

## C++ tracker

### Dependencies

| Dependency | Version tested | Notes |
|---|---|---|
| [Prophesee Metavision SDK](https://docs.prophesee.ai/4.6.2/installation/index.html) | 4.6.2 | Modules `base`, `core`, `driver`, `cv` and `ui`. The `cv` algorithms used here (sparse optical flow, anti-flicker, spatio-temporal contrast filter) come with the Metavision SDK; they are not part of [OpenEB](https://github.com/prophesee-ai/openeb) alone. |
| [OpenCV](https://opencv.org/) | 4.x (tested with 4.9) | `core`, `imgproc`, `imgcodecs`, `calib3d` |
| [PCL](https://pointclouds.org/) | 1.12 | `common`, `io`, `features`, `visualization` (built with VTK) |
| CMake | ≥ 3.5 | |
| C++ compiler | C++17 (tested with GCC 11, Ubuntu 22.04) | |

### Build

```bash
cd cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
```

If OpenCV is installed in a non-standard location, point CMake to it with `-DOpenCV_DIR=/path/to/opencv/lib/cmake/opencv4`, and add its `lib` folder to `LD_LIBRARY_PATH` before running.

### Usage

```bash
./KentTracker <cam1.raw> <cam2.raw> [key=value ...]
```

The two `.raw` files are synchronized recordings from the master and slave cameras of the fisheye stereo rig. They are replayed in real time. The program shows three windows: the event frames of both cameras, and the unit sphere with the Kent estimate. On the sphere, cyan marks the mode/bearing, magenta the major axis ξ1 (heading) and yellow the minor axis ξ2. Press `Q` or `Esc` in a camera window to quit, or `H` to print the data-association statistics. The program also stops at the end of the recordings.

The camera intrinsics and the unified-model parameter ξ of both cameras are hard-coded in `KentTracker.cpp` (`K1`, `xi1`, `K2`, `xi2`). Replace them with the calibration of your own rig.

Optional parameters:

| Key | Default | Description |
|---|---|---|
| `bearing_gate` | 15 | Bearing gate for data association (deg) |
| `heading_gate` | 60 | Gate on the flow-heading discrepancy (deg) |
| `heading_weight` | 1 | Weight of the heading term in the association cost (fixed-weight mode) |
| `beta_activate` | 5 | Minimum ovalness β before the heading is used for association |
| `use_heading` | 1 | `1`: Kent-aware (bearing + heading) association, `0`: bearing-only nearest neighbour |
| `adaptive` / `--adaptive` | 0 | Adaptive association weights (κ/κ_max for bearing, β/β_max for heading) |
| `reinit_kappa_min` | 50 | Reinitialize the filter when κ falls below this value |
| `reinit_miss_limit` | 100 | Reinitialize after this many consecutive missed associations |
| `pred_k_noise` | 500 | Process-noise concentration κ of the prediction |
| `pred_beta_noise` | 200 | Process-noise ovalness β of the prediction |
| `min_speed` | 2 | Discard optical-flow vectors slower than this (px/s) |
| `use_mask` | 0 | `1`: reject flows on masked pixels (value 255 in the mask images) |
| `mask_master`, `mask_slave` | `mask_master.png`, `mask_slave.png` | Grayscale mask images, used when `use_mask=1` |
| `log_dir` | `.` | Output directory for log files |
| `tag` | current date | Suffix of the log file names |
| `alog` | 0 | `1`: also log the A_sum vs A_opt approximation error to `aerror_<tag>.dat` |
| `dump_dir` | (empty) | If set, save screenshots of the sphere view to `<dump_dir>/frame_%05d.png` |
| `cam_cx`, `cam_cy` | 0.1, 0.4 | Shift of the sphere view in the window |

Example:

```bash
./KentTracker cam1.raw cam2.raw tag=seq1 log_dir=results --adaptive
```

### Output

`<log_dir>/trackerKent_<tag>.dat` is a space-separated file with a header line. It has one row per main-loop iteration, starting from the first initialization of the filter:

`t u_x u_y u_z xi1_x xi1_y xi1_z k beta nInit useHeadingAssoc headingDisc assocDiverged assocDivergeCount headingRejectCount bearingW headingW assocDb assocDh assocCost`

`t` is the time in seconds since the first initialization. `u` is the estimated bearing (mode μ), `xi1` the major axis (heading), and `k` and `beta` the concentration and ovalness. `nInit` counts the (re)initializations. The remaining columns describe the data association of the latest update.

## MATLAB code

`matlab/KentDistribution.m` and `matlab/KentFilter.m` are meant to be used with the [libDirectional](https://github.com/libDirectional/libDirectional) library. They extend its `AbstractHypersphericalDistribution` and `AbstractHypersphericalFilter` classes. Add libDirectional to your MATLAB path, then add the `matlab/` folder of this repository.

## Calibration and live version

For more information on the calibration of the camera rig, or on the live version of this code (running directly on the cameras), refer to [github.com/maxrobic/SFERA](https://github.com/maxrobic/SFERA).

## License

This code is released under the [MIT License](LICENSE): you may use, modify and redistribute it, provided the copyright notice crediting the authors is kept. Third-party dependencies (Prophesee Metavision SDK, OpenCV, PCL, libDirectional) remain under their own licenses.
