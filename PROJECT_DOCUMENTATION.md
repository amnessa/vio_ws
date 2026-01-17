# VIO EKF Project Documentation

## Visual-Inertial Odometry using Error-State Extended Kalman Filter

This project implements a Visual-Inertial Odometry (VIO) system using an Error-State Extended Kalman Filter (ES-EKF) for robot localization. The system fuses IMU measurements with ArUco marker observations to estimate robot pose in real-time.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Files and Components](#files-and-components)
4. [EKF Node (ekf_node.py)](#ekf-node-ekf_nodepy)
5. [Vision Node (vision_node.py)](#vision-node-vision_nodepy)
6. [Evaluation Node (eval_node.py)](#evaluation-node-eval_nodepy)
7. [Ground Truth Broadcaster (pose_tf_broadcaster.cpp)](#ground-truth-broadcaster-pose_tf_broadcastercpp)
8. [Configuration Parameters](#configuration-parameters)
9. [Launch System](#launch-system)
10. [ROS2 Topics](#ros2-topics)
11. [Coordinate Frames](#coordinate-frames)
12. [Filter Modes](#filter-modes)

---

## Project Overview

### Purpose
Estimate 6-DOF robot pose (position and orientation) by fusing:
- **IMU**: High-rate (200Hz) inertial measurements for motion prediction
- **Camera**: Lower-rate (~30Hz) ArUco marker observations for drift correction

### Key Features
- **Robocentric Formulation**: Velocity expressed in body frame to decouple observable states from unobservable global yaw
- **Error-State EKF**: Efficient quaternion-based orientation with minimal parameterization
- **ZUPT**: Zero-Velocity Update for stationary periods
- **Outlier Rejection**: Mahalanobis distance-based measurement gating
- **Time Delay Compensation**: State buffer for matching vision measurements with IMU state

---

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Gazebo Sim     │     │  ROS2 Bridge    │     │   ROS2 Nodes    │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ - IMU sensor    │────▶│ /imu            │────▶│  EKF Node       │
│ - Camera sensor │────▶│ /camera         │────▶│  Vision Node    │
│ - Diff drive    │◀────│ /cmd_vel        │◀────│  (detects       │
│ - Ground truth  │────▶│ /ground_truth   │     │   ArUco)        │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
                                                ┌─────────────────┐
                                                │  Outputs        │
                                                │ - /vio/odom     │
                                                │ - /vio/path     │
                                                │ - TF transforms │
                                                └─────────────────┘
```

---

## Files and Components

### Source Files (`src/vio_ekf/src/`)

| File | Language | Description |
|------|----------|-------------|
| `ekf_node.py` | Python | Main EKF filter - prediction and correction steps |
| `vision_node.py` | Python | ArUco marker detection and pixel coordinate extraction |
| `eval_node.py` | Python | Trajectory evaluation - computes RMSE, ATE, generates plots |
| `pose_tf_broadcaster.cpp` | C++ | Extracts ground truth pose from Gazebo and publishes to ROS2 |

### Configuration (`src/vio_ekf/config/`)

| File | Description |
|------|-------------|
| `ekf_params.yaml` | Main EKF tuning parameters (process noise, measurement noise, modes) |
| `imu.yaml` | IMU sensor configuration for Gazebo bridge |
| `camera.yaml` | Camera sensor configuration for Gazebo bridge |
| `diff_drive.yaml` | Differential drive controller configuration |

### Launch (`src/vio_ekf/launch/`)

| File | Description |
|------|-------------|
| `vio_ekf.launch.py` | Main launch file - starts Gazebo, bridges, all nodes, RViz |

### Scripts (`src/vio_ekf/scripts/`)

| File | Description |
|------|-------------|
| `generate_aruco_simulation.py` | Generates ArUco marker models for Gazebo world |

---

## EKF Node (ekf_node.py)

The core filter implementation using Error-State Extended Kalman Filter.

### Class: `EKFNode`

#### State Vector (16 elements)
```
x = [p_w(3), v_b(3), q(4), ba(3), bg(3)]
     ├── Position in WORLD frame (for visualization)
     ├── Velocity in BODY frame (robocentric - decoupled from yaw)
     ├── Quaternion [w, x, y, z] (world to body rotation)
     ├── Accelerometer bias (body frame)
     └── Gyroscope bias (body frame)
```

#### Error State Vector (15 elements)
```
δx = [δp(3), δv(3), δθ(3), δba(3), δbg(3)]
```

### Key Methods

#### `__init__(self)`
Initializes the EKF node:
- Declares and reads ROS2 parameters
- Initializes state vector `x` and covariance matrix `P`
- Sets up subscribers (IMU, camera, ground truth) and publishers
- Configures IMU and camera extrinsics
- Loads ArUco marker map (24 markers in circular pattern)

#### `imu_callback(self, msg)`
Handles incoming IMU messages at 200Hz:
1. Transforms IMU readings from IMU frame to body frame
2. Applies lever-arm compensation for IMU offset
3. During initialization: collects samples for bias estimation
4. After initialization:
   - Downsamples IMU (4:1 averaging → 50Hz)
   - Calls `predict()` or `predict_synthetic()` depending on mode
   - Publishes cmd_vel in synthetic mode

#### `predict(self, dt, a_m, w_m)`
ES-EKF prediction step using IMU measurements:
1. **Bias Correction**: Subtracts estimated biases from measurements
2. **ZUPT Detection**: Checks gyro for stationary periods
3. **Nominal State Propagation**:
   - Computes body-frame acceleration: `acc = a_m - ba - R^T·g - ω×v`
   - Updates velocity: `v_new = v + acc·dt`
   - Updates position: `p_new = p + R·v·dt`
   - Updates orientation via quaternion integration
4. **Covariance Propagation**: `P = F·P·F^T + Q`

#### `predict_synthetic(self, dt, fixed_vx, fixed_omega_z)`
Synthetic "frozen" prediction for measurement-only testing:
- Ignores actual IMU measurements
- Uses configurable constant velocity and yaw rate
- Maintains healthy covariance with synthetic process noise
- Publishes matching cmd_vel to move the robot

#### `vision_callback(self, msg)`
Handles ArUco marker observations:
1. Matches vision timestamp to buffered IMU state
2. Collects valid landmark observations
3. Requires 2+ markers for observability
4. Calls `batch_update()` for measurement update

#### `batch_update(self, observations, buffered_state)`
Iterated EKF (IEKF) measurement update:
1. Projects landmarks from world → body → camera → pixels
2. Computes residuals (measured - predicted pixels)
3. Rejects outliers (>100px error)
4. Builds stacked Jacobian H and measurement matrix
5. Computes Kalman gain: `K = P·H^T·(H·P·H^T + R)^(-1)`
6. Updates state: `x = x + K·(z - h(x))`
7. Updates covariance using Joseph form

#### `initialize_from_imu(self)`
Computes initial state from stationary IMU readings:
- Gyro mean → gyroscope bias
- Accel direction → initial orientation (gravity alignment)
- Accel magnitude → gravity model calibration

#### `_zupt_velocity_update(self)`
Formal EKF update when stationary:
- Measurement: body velocity = 0
- Updates both state and covariance consistently

#### `_zupt_gravity_update(self, a_corrected)`
Tilt correction from accelerometer during stationary:
- Uses accelerometer as tilt sensor
- Corrects roll/pitch drift

#### `_reinitialize_orientation(self, a_m, w_m)`
Smart recovery when filter diverges:
- Re-estimates roll/pitch from accelerometer
- Preserves yaw (gravity doesn't provide yaw)
- Resets velocity to zero
- Expands covariance

#### `publish_state(self, stamp)`
Publishes filter outputs:
- Odometry message to `/vio/odom`
- Path for trajectory visualization
- TF transform `map → base_footprint_vio`
- Diagnostics (biases, covariances, velocities)

#### `publish_synthetic_cmd_vel(self)`
Publishes Twist messages to `/cmd_vel` matching synthetic prediction parameters.

---

## Vision Node (vision_node.py)

ArUco marker detection frontend.

### Class: `VisionNode`

#### Purpose
Detects ArUco markers (DICT_4X4_50) in camera images and publishes their pixel coordinates with unique IDs.

#### Key Features
- **Subpixel Corner Refinement**: Improves measurement accuracy
- **Aggressive Detection**: Tuned for varying lighting and distances
- **CLAHE Enhancement**: Contrast Limited Adaptive Histogram Equalization for poor textures

#### Methods

##### `image_callback(self, msg)`
Processes each camera frame:
1. Converts ROS Image to OpenCV format
2. Applies CLAHE contrast enhancement
3. Detects ArUco markers
4. Extracts center pixel coordinates
5. Publishes PoseArray with (u, v, marker_id)

#### Output Format
```
PoseArray:
  poses[i].position.x = u (pixel)
  poses[i].position.y = v (pixel)
  poses[i].position.z = marker_id
```

---

## Evaluation Node (eval_node.py)

Trajectory evaluation and visualization.

### Class: `EvaluationNode`

#### Purpose
Computes error metrics and generates plots comparing EKF estimate to ground truth.

#### Metrics Computed
- **RMSE**: Root Mean Square Error
- **ATE**: Absolute Trajectory Error
- **Per-axis errors**: X, Y, Z position errors
- **Yaw error**: Heading error in degrees

#### Methods

##### `est_callback(self, msg)`
Processes EKF odometry estimates:
- Matches with closest ground truth
- Computes position and yaw errors
- Logs real-time metrics

##### `diag_callback(self, msg)`
Processes EKF diagnostics:
- Stores bias histories
- Stores covariance histories
- Stores velocity and speed

##### `save_results(self)`
Generates comprehensive plots:
- XY trajectory comparison
- Error over time (position and yaw)
- Bias evolution
- Covariance evolution
- Velocity profiles

---

## Ground Truth Broadcaster (pose_tf_broadcaster.cpp)

### Class: `PoseTfBroadcaster`

#### Purpose
Extracts ground truth robot pose from Gazebo world info and publishes to ROS2.

#### Functionality
- Subscribes to `/world/vio_world/dynamic_pose/info`
- Extracts pose for "turtlebot3" model
- Publishes to `/ground_truth/odom`
- Broadcasts TF: `map → base_footprint_gt`

---

## Configuration Parameters

### File: `config/ekf_params.yaml`

#### Filter Mode Switches

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_prediction` | bool | true | Enable IMU-based prediction step |
| `enable_correction` | bool | true | Enable camera measurement updates |
| `enable_zupt` | bool | true | Enable Zero-Velocity Update |
| `enable_imu_lever_arm` | bool | true | Compensate for IMU offset from body origin |

#### Synthetic Prediction Mode

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `synthetic_velocity` | float | 0.0 | Forward velocity when prediction disabled (m/s) |
| `synthetic_omega` | float | 0.0 | Yaw rate when prediction disabled (rad/s) |

When `enable_prediction=false`:
- Filter uses synthetic velocity/omega instead of IMU
- Publishes matching cmd_vel to move robot in simulation
- Useful for testing vision-only performance

#### Process Noise (Q) - IMU Trust

| Parameter | Type | Default | Effect |
|-----------|------|---------|--------|
| `sigma_accel` | float | 1.5 | Accel noise σ (m/s²). Higher = less trust in IMU |
| `sigma_gyro` | float | 0.5 | Gyro noise σ (rad/s). Higher = less trust in IMU |
| `sigma_accel_bias` | float | 0.01 | Accel bias random walk. Higher = faster bias adaptation |
| `sigma_gyro_bias` | float | 0.01 | Gyro bias random walk. Higher = faster bias adaptation |

#### Measurement Noise (R) - Camera Trust

| Parameter | Type | Default | Effect |
|-----------|------|---------|--------|
| `R_camera` | float | 35.0 | Pixel noise variance. Lower = more trust in camera |

#### ZUPT Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `zupt_gyro_threshold` | float | 0.02 | Gyro threshold for stationary detection (rad/s) |
| `R_zupt_velocity` | float | 0.001 | ZUPT velocity measurement noise |

#### Outlier Rejection

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mahalanobis_threshold` | float | 60.0 | Threshold for measurement gating |

#### Timing

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `imu_downsample_factor` | int | 4 | 200Hz / factor = effective IMU rate |
| `max_time_delay` | float | 0.15 | Max acceptable vision-IMU time mismatch (s) |

---

## Launch System

### File: `launch/vio_ekf.launch.py`

#### Nodes Launched

| Node | Package | Description |
|------|---------|-------------|
| `gz_sim` | ExecuteProcess | Gazebo Harmonic simulation |
| `parameter_bridge` | ros_gz_bridge | ROS2 ↔ Gazebo message bridge |
| `pose_tf_broadcaster` | vio_ekf | Ground truth extraction |
| `robot_state_publisher` | robot_state_publisher | Robot URDF publishing |
| `ekf_node` | vio_ekf | Main EKF filter |
| `vision_node` | vio_ekf | ArUco detection |
| `rviz2` | rviz2 | Visualization |

#### TF Frames Published
- `world → map` (static, identity)
- `world → odom` (static, identity)
- `map → base_footprint_vio` (from EKF)
- `map → base_footprint_gt` (from ground truth)

---

## ROS2 Topics

### Subscriptions

| Topic | Type | Node | Description |
|-------|------|------|-------------|
| `/imu` | sensor_msgs/Imu | ekf_node | IMU measurements at 200Hz |
| `/camera` | sensor_msgs/Image | vision_node | Camera images |
| `/camera_info` | sensor_msgs/CameraInfo | ekf_node | Camera intrinsics |
| `/vio/landmarks` | geometry_msgs/PoseArray | ekf_node | Detected ArUco markers |
| `/ground_truth/odom` | nav_msgs/Odometry | ekf_node, eval_node | Ground truth pose |

### Publications

| Topic | Type | Node | Description |
|-------|------|------|-------------|
| `/vio/odom` | nav_msgs/Odometry | ekf_node | EKF pose estimate |
| `/vio/path` | nav_msgs/Path | ekf_node | Trajectory history |
| `/vio/diagnostics` | std_msgs/Float64MultiArray | ekf_node | Filter internals |
| `/vio/landmarks` | geometry_msgs/PoseArray | vision_node | Detected markers |
| `/cmd_vel` | geometry_msgs/Twist | ekf_node | Robot velocity commands (synthetic mode) |

---

## Coordinate Frames

### Frame Definitions

| Frame | Convention | Description |
|-------|------------|-------------|
| `world` | ENU | Gazebo world frame (East-North-Up) |
| `map` | ENU | ROS map frame (= world) |
| `odom` | ENU | Odometry frame (= world) |
| `base_footprint` | ROS | Robot body center on ground |
| `base_link` | ROS | Robot body center |
| `camera_link` | ROS | Camera optical frame |
| `imu_link` | ROS | IMU sensor frame |

### Body Frame Convention
- **X**: Forward
- **Y**: Left
- **Z**: Up

### Camera Optical Frame Convention
- **X**: Right
- **Y**: Down
- **Z**: Forward (depth)

### Transformation: Body → Camera
```python
R_b_c = [[ 0, -1,  0],   # Camera X = -Body Y
         [ 0,  0, -1],   # Camera Y = -Body Z
         [ 1,  0,  0]]   # Camera Z = Body X
```

---

## Filter Modes

### 1. Full VIO (Default)
```yaml
enable_prediction: true
enable_correction: true
enable_zupt: true
```
Full sensor fusion: IMU prediction + Camera correction + ZUPT.

### 2. IMU Only (Prediction Only)
```yaml
enable_prediction: true
enable_correction: false
enable_zupt: true
```
Tests IMU integration quality. Expect drift over time.

### 3. Camera Only (Synthetic Prediction)
```yaml
enable_prediction: false
enable_correction: true
synthetic_velocity: 0.5
synthetic_omega: 0.2
```
Tests camera measurement quality. Robot moves with constant velocity.
The filter uses synthetic motion model and sends matching cmd_vel commands.

### 4. Stationary Test
```yaml
enable_prediction: false
enable_correction: true
synthetic_velocity: 0.0
synthetic_omega: 0.0
```
Robot stays still. Tests vision correction on stationary target.

---

## Usage

### Running the System
```bash
# Build
cd /workspaces/vio_ws
colcon build --packages-select vio_ekf

# Source
source install/setup.bash

# Launch
ros2 launch vio_ekf vio_ekf.launch.py
```

### Running Evaluation
```bash
# In a new terminal after the system is running
source install/setup.bash
ros2 run vio_ekf eval_node.py
# Drive robot around, then Ctrl+C to save plots
```

### Modifying Parameters
Edit `src/vio_ekf/config/ekf_params.yaml` and re-launch.

---

## Troubleshooting

### Filter Divergence
- Check for large acceleration spikes in logs
- Reduce `sigma_accel` to trust IMU more
- Check camera is detecting markers (`/vio/landmarks` topic)

### Robot Not Moving (Synthetic Mode)
- Verify `enable_prediction: false` in config
- Check `synthetic_velocity` and `synthetic_omega` are non-zero
- Verify `/cmd_vel` topic is being published

### High Position Error
- Increase `R_camera` if camera is noisy
- Decrease `R_camera` if filter isn't correcting
- Check marker positions in `self.map` match world file

### No Markers Detected
- Check camera topic is publishing (`ros2 topic hz /camera`)
- Check debug images in `/tmp/aruco_debug_*.png`
- Verify marker size and dictionary match (DICT_4X4_50)
