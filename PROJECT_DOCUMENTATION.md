# VIO ES-EKF Project Documentation

## Overview

This project implements a **Visual-Inertial Odometry (VIO)** system using an **Error-State Extended Kalman Filter (ES-EKF)** for a TurtleBot3 ground robot in Gazebo simulation. The filter fuses high-rate IMU data (200Hz) with camera-based ArUco marker observations to estimate the robot's pose, velocity, and sensor biases.

### Key Design Decisions

1. **Robocentric Formulation**: Velocity is expressed in body frame (`v_b`) rather than world frame to decouple observable states from unobservable global yaw.
2. **Error-State (MEKF) Approach**: Uses minimal 15D error state for orientation (rotation vector) while maintaining full 16D nominal state with quaternion.
3. **Iterated EKF (IEKF)**: Vision updates use Gauss-Newton iterations to handle large residuals.
4. **Position-Only Vision Updates**: Only position is directly observed; orientation/velocity corrections come through covariance cross-correlations.

---

## File Structure

```
src/vio_ekf/
├── src/
│   ├── ekf_node.py          # Main ES-EKF implementation
│   ├── eval_node.py         # Evaluation and plotting node
│   ├── vision_node.py       # ArUco marker detection
│   └── pose_tf_broadcaster.cpp  # TF frame broadcaster
├── config/                   # Gazebo sensor configurations
├── launch/                   # ROS2 launch files
├── models/                   # Gazebo world models
├── rviz/                     # RViz configurations
├── urdf/                     # Robot URDF descriptions
└── worlds/                   # Gazebo world files
```

---

## Core Classes and Functions

### 1. EKFNode (`ekf_node.py`)

The main ROS2 node implementing the Error-State Extended Kalman Filter.

#### State Representation

```
Nominal State (16D): [p_w(3), v_b(3), q(4), ba(3), bg(3)]
  - p_w:  Position in WORLD frame (meters)
  - v_b:  Velocity in BODY frame (robocentric formulation)
  - q:    Quaternion [w, x, y, z] (world to body orientation)
  - ba:   Accelerometer bias (m/s²)
  - bg:   Gyroscope bias (rad/s)

Error State (15D): [δp(3), δv(3), δθ(3), δba(3), δbg(3)]
  - δθ:   Rotation vector (minimal representation)
```

#### Constructor `__init__(self)`

Initializes the EKF node with:
- State vector `self.x` (16D) and covariance `self.P` (15×15)
- Noise parameters: `sigma_a`, `sigma_g`, `Q_ba`, `Q_bg`, `R_cam`
- ZUPT parameters for zero-velocity updates
- IMU downsampling buffer (200Hz → 50Hz)
- ArUco landmark map (24 markers in circular arrangement at 2.5m radius)
- Camera intrinsics and extrinsics (`K`, `R_b_c`, `t_b_c`)
- ROS2 subscribers and publishers

#### Key Methods

| Method | Description |
|--------|-------------|
| `predict(dt, a_m, w_m)` | IMU odometry prediction step (see below) |
| `batch_update(observations)` | IEKF vision update with multiple landmarks |
| `update(lm_pos_world, u_meas, v_meas)` | Single landmark EKF update (legacy) |
| `_zupt_velocity_update()` | Formal EKF update to zero velocity when stationary |
| `_zupt_gravity_update(a_corrected)` | Gravity-based tilt correction when stationary |
| `_reinitialize_orientation(a_m, w_m)` | Recovery from filter divergence |
| `initialize_from_imu()` | Initial orientation estimation from gravity |
| `publish_state(timestamp)` | Publish odometry, path, and diagnostics |
| `compute_ate()` | Calculate Absolute Trajectory Error |
| `compute_nees()` | Calculate Normalized Estimation Error Squared |

---

### Prediction Step (`predict`)

The prediction step implements body-frame velocity dynamics:

```
Position:    p_w_dot = R_wb @ v_b
Velocity:    v_b_dot = a_corrected - g_body - ω × v_b
Orientation: q_dot = 0.5 * q ⊗ [0, ω_corrected]
```

**Key Implementation Details:**

1. **IMU Downsampling**: 200Hz → 50Hz by averaging every 4 samples
   ```python
   self.imu_buffer.append({'accel': a_m, 'gyro': w_m})
   if len(self.imu_buffer) >= self.imu_downsample_factor:
       a_avg = np.mean([s['accel'] for s in self.imu_buffer], axis=0)
       w_avg = np.mean([s['gyro'] for s in self.imu_buffer], axis=0)
   ```

2. **Gravity Compensation**: Subtract gravity in body frame
   ```python
   g_body = R_wb.T @ self.g  # Gravity in body frame
   acc_body = a_corrected - g_body - coriolis  # SUBTRACT gravity
   ```

3. **ba_z Lock**: Z-accelerometer bias locked to zero (unobservable for ground robot)
   ```python
   self.x[12] = 0.0  # ba_z locked to zero
   ```

4. **Ground Constraints**:
   - Z-position = 0
   - Z-velocity = 0
   - Roll/pitch clamped to ±10°

5. **Error-State Jacobians** (Discrete-time F matrix):
   ```
   F = I + Fc * dt where Fc is the continuous-time Jacobian

   Fc structure (15×15):
   - dp/dv:  R_wb (body velocity to world position)
   - dv/dθ:  skew(a_corrected - g_body)
   - dv/dba: -I
   - dθ/dθ:  -skew(ω)
   - dθ/dbg: -I
   ```

---

### Vision Update (`batch_update`)

Implements Iterated EKF (IEKF) for robust convergence with large residuals.

**Algorithm:**
1. For each iteration (up to 4):
   - Linearize measurement model at current estimate
   - Compute stacked residual and Jacobian for all landmarks
   - Apply per-observation gate (reject if residual > 100px)
   - Compute Kalman gain using PRIOR covariance
   - Update state estimate
   - Check convergence

**Measurement Model:**
```
h(x) = π(R_bc @ (R_wb^T @ (lm_world - p_w) - t_bc))

where π is the pinhole projection:
  u = fx * X/Z + cx
  v = fy * Y/Z + cy
```

**Position-Only Jacobian:**
```python
H = np.zeros((2, 15))
H[:, 0:3] = J_proj @ (-R_bc @ R_wb.T)  # Only position columns
```

**Covariance Update (Joseph Form):**
```python
IKH = I - K @ H
P = IKH @ P @ IKH.T + K @ R @ K.T  # Numerically stable
```

---

### ZUPT (Zero-Velocity Update)

Two-part update when robot is stationary:

#### `_zupt_velocity_update()`
Formal EKF update with measurement z = [0, 0, 0] (zero velocity):
```python
H = [0, 0, 0 | I_3×3 | 0, 0, 0 | 0, 0, 0 | 0, 0, 0]  # Selects velocity
K = P @ H.T @ inv(H @ P @ H.T + R_zupt)
dx = K @ (0 - v_b)
```

#### `_zupt_gravity_update(a_corrected)`
Treats accelerometer as tilt sensor:
```python
a_expected = R_wb.T @ g_world  # What accel should read if aligned
z_res = a_measured - a_expected
# Jacobian: d(R^T @ g)/dθ = -skew(a_expected)
```

**Stationary Detection:**
- Gyro norm < 0.02 rad/s for 0.25 seconds
- Vision motion cooldown not active (no recent position correction > 2cm)

---

### Helper Functions

#### `skew_symmetric(v)`
Computes the skew-symmetric (cross-product) matrix:
```python
def skew_symmetric(v):
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
```

#### `quat_to_rotation_matrix(q)`
Converts quaternion [w, x, y, z] to 3×3 rotation matrix:
```python
def quat_to_rotation_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
    ])
```

---

### 2. VisionNode (`vision_node.py`)

ArUco marker detection node using OpenCV.

#### Class: `VisionNode`

| Method | Description |
|--------|-------------|
| `__init__()` | Initializes ArUco detector (DICT_4X4_50), CLAHE for contrast enhancement |
| `image_callback(msg)` | Detects markers, publishes pixel coordinates and IDs |

**Detection Pipeline:**
1. Convert image to grayscale
2. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
3. Detect ArUco markers with subpixel corner refinement
4. Extract marker centers (average of 4 corners)
5. Publish as `PoseArray` with: `x=u, y=v, z=marker_id`

**ArUco Parameters (Aggressive for Simulation):**
```python
cornerRefinementMethod = CORNER_REFINE_SUBPIX
minMarkerPerimeterRate = 0.005  # Detect tiny markers
maxMarkerPerimeterRate = 8.0    # And very large ones
errorCorrectionRate = 1.0       # Maximum error correction
```

---

### 3. EvaluationNode (`eval_node.py`)

Real-time evaluation with ground truth comparison and diagnostic plotting.

#### Class: `EvaluationNode`

| Method | Description |
|--------|-------------|
| `__init__()` | Subscribes to `/vio/odom`, `/ground_truth/odom`, `/vio/diagnostics` |
| `est_callback(msg)` | Computes position/yaw error, logs metrics |
| `gt_callback(msg)` | Stores latest ground truth |
| `diag_callback(msg)` | Records bias, covariance, velocity diagnostics |
| `save_plots()` | Generates 7 evaluation plots on shutdown |

**Diagnostics Message Format** (`/vio/diagnostics`):
```
Float64MultiArray.data[0:6]   = [ba_x, ba_y, ba_z, bg_x, bg_y, bg_z]
Float64MultiArray.data[6:21]  = P diagonal (position, velocity, orientation, ba, bg)
Float64MultiArray.data[21:25] = [vx, vy, vz, speed]
Float64MultiArray.data[25]    = vision_correction_magnitude
```

**Generated Plots:**
1. `trajectory_plot.png` - 2D trajectory comparison
2. `error_plot.png` - Position error over time
3. `position_comparison.png` - X, Y, Z comparison
4. `yaw_comparison.png` - Yaw angle comparison
5. `bias_plot.png` - Accelerometer and gyroscope biases
6. `covariance_plot.png` - Uncertainty evolution (6 subplots)
7. `velocity_plot.png` - Body-frame velocity components

---

### 4. pose_tf_broadcaster (`pose_tf_broadcaster.cpp`)

C++ node for broadcasting TF frames.

**Published Transforms:**
- `odom` → `base_link` (from EKF estimate)

---

## ROS2 Topics

### Subscriptions

| Topic | Type | Description |
|-------|------|-------------|
| `/imu` | `sensor_msgs/Imu` | IMU data at 200Hz |
| `/camera` | `sensor_msgs/Image` | Camera images |
| `/camera_info` | `sensor_msgs/CameraInfo` | Camera intrinsics |
| `/vio/landmarks` | `geometry_msgs/PoseArray` | Detected ArUco markers |
| `/ground_truth/odom` | `nav_msgs/Odometry` | Ground truth pose |

### Publications

| Topic | Type | Description |
|-------|------|-------------|
| `/vio/odom` | `nav_msgs/Odometry` | EKF pose estimate |
| `/vio/path` | `nav_msgs/Path` | Trajectory visualization |
| `/vio/diagnostics` | `std_msgs/Float64MultiArray` | Internal filter states |

---

## Configuration Parameters

### Noise Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sigma_a` | 1.5 m/s² | Accelerometer noise std dev |
| `sigma_g` | 0.5 rad/s | Gyroscope noise std dev |
| `Q_ba` | 1e-4 | Accelerometer bias random walk |
| `Q_bg` | 1e-5 | Gyroscope bias random walk |
| `R_cam` | 35.0 px² | Pixel measurement noise variance |

### ZUPT Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `zupt_gyro_threshold` | 0.02 rad/s | Stationary detection threshold |
| `zupt_window_size` | 50 | Samples for averaging (~250ms) |
| `R_zupt_velocity` | 0.001 (m/s)² | Zero-velocity measurement noise |
| `R_zupt_gravity` | 0.1 | Gravity measurement noise |

### Vision Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mahalanobis_threshold` | 60.0 | Outlier rejection threshold |
| `max_consecutive_outliers` | 5 | Before accepting measurements |
| Per-observation gate | 100 px | Reject if residual > 100px |

---

## Landmark Map

24 ArUco markers arranged in a single ring at 2.5m radius, evenly spaced at 15° intervals:

```python
self.map = {
    0.0:  np.array([2.50, 0.00, 0.30]),   # East
    6.0:  np.array([0.00, 2.50, 0.30]),   # North
    12.0: np.array([-2.50, 0.00, 0.30]),  # West
    18.0: np.array([0.00, -2.50, 0.30]),  # South
    # ... 20 more markers
}
```

All markers are at z=0.30m (camera height) facing inward toward the origin.

---

## Key Fixes and Design Decisions

### 1. Gravity Sign Fix
**Problem:** Accelerometer measures specific force, not acceleration.
**Fix:** `acc_body = a_corrected - g_body` (subtract gravity, not add)

### 2. FEJ Disabled
**Problem:** First-Estimate Jacobians caused gradient to point wrong direction when yaw deviated from first observation.
**Fix:** Use current `R_wb` for Jacobian computation instead of stored `R_fej`.

### 3. ba_z Locked to Zero
**Problem:** Z-accelerometer bias is unobservable for ground robots (coupled with gravity).
**Fix:** Lock `ba_z = 0`, zero its covariance column/row, zero process noise.

### 4. Full dx Application with Limits
**Problem:** Zeroing non-position corrections broke covariance cross-correlations.
**Fix:** Apply full correction with conservative limits (position ±1.5m, velocity ±0.5m/s, etc.).

### 5. Vision-Based ZUPT Gating
**Problem:** ZUPT fired during straight-line motion (low gyro but moving).
**Fix:** If vision correction > 2cm, disable ZUPT for 0.5s cooldown.

### 6. IMU Downsampling
**Problem:** 200Hz IMU spikes (50 m/s²) caused filter divergence.
**Fix:** Average every 4 samples → 50Hz (spikes become 12.5 m/s² max).

### 7. Yaw Preservation on Re-init
**Problem:** IMU re-initialization destroyed yaw (only roll/pitch recoverable from gravity).
**Fix:** Preserve current yaw, only reset roll/pitch during re-initialization.

---

## Usage

### Build
```bash
cd /workspaces/vio_ws
colcon build --packages-select vio_ekf
source install/setup.bash
```

### Run
```bash
# Terminal 1: Launch simulation
ros2 launch vio_ekf simulation.launch.py

# Terminal 2: Run EKF
ros2 run vio_ekf ekf_node.py

# Terminal 3: Run vision
ros2 run vio_ekf vision_node.py

# Terminal 4: Run evaluation
ros2 run vio_ekf eval_node.py
```

### View Results
Press `Ctrl+C` in eval_node terminal to generate plots in `/workspaces/vio_ws/`.

---

## References

1. Huang, Mourikis, Roumeliotis - "Observability-based Rules for Designing Consistent EKF SLAM Estimators" (IJRR 2010)
2. Solà, J. - "Quaternion kinematics for the error-state Kalman filter"
3. Forster et al. - "On-Manifold Preintegration for Real-Time Visual-Inertial Odometry"
4. Bloesch et al. - "Iterated Extended Kalman Filter Based Visual-Inertial Odometry"
