# VIO ES-EKF Implementation Journey Report

**Project:** Visual-Inertial Odometry for TurtleBot3 Ground Robot
**Platform:** ROS2 Humble, Gazebo Simulation
**Date:** January 2026

---

## Executive Summary

This report documents the implementation journey of an Error-State Extended Kalman Filter (ES-EKF) for Visual-Inertial Odometry. The project evolved through multiple debugging iterations, with the filter initially diverging within 20-30 seconds to eventually achieving stable, accurate pose estimation. The breakthrough came from implementing comprehensive diagnostic plotting, which revealed previously hidden internal filter dynamics and led to three critical fixes.

**Final Performance:** Stable operation with sub-meter accuracy over extended runs.

---

## 1. Initial Implementation

### 1.1 Starting Point

The project began with a standard ES-EKF implementation:
- **State:** 16D nominal state [position, velocity, quaternion, accel_bias, gyro_bias]
- **Error State:** 15D using rotation vector for orientation
- **Sensors:** 200Hz IMU, camera with 24 ArUco markers
- **Vision:** Pinhole projection model with known landmark map

### 1.2 Initial Problems

The filter exhibited catastrophic divergence:
- Position drifted to **6+ kilometers** from ground truth within 30 seconds
- Velocity estimates exploded to **50+ m/s** (robot physically limited to ~0.5 m/s)
- Filter would trigger re-initialization repeatedly
- No visibility into *why* divergence occurred

---

## 2. First Wave of Fixes

### 2.1 Gravity Sign Error (Critical)

**Discovery:** The accelerometer measurement model had an incorrect sign.

**Problem:**
```python
# WRONG: Adding gravity instead of subtracting
acc_body = a_corrected + g_body
```

**Physics:** The accelerometer measures specific force (what it "feels"), not acceleration. When stationary, it reads +g upward to counteract gravity. The correct formulation:

```python
# CORRECT: Subtract gravity
acc_body = a_corrected - g_body
```

**Impact:** This single fix reduced velocity divergence significantly, but filter still drifted.

### 2.2 Robocentric Velocity Formulation

**Problem:** World-frame velocity couples with unobservable global yaw, causing yaw errors to leak into velocity estimates.

**Solution:** Reformulated velocity in body frame:
```
v_b_dot = a_b - ba + R_wb^T @ g - ω × v_b  (body frame)
p_w_dot = R_wb @ v_b                        (world frame for position)
```

**Impact:** Decoupled observable motion from unobservable yaw. Improved stability during turns.

### 2.3 Position-Only Vision Updates

**Problem:** Full measurement Jacobian (position + orientation) injected spurious orientation corrections that destabilized gravity compensation.

**Solution:** Masked Kalman gain to only update position:
```python
K_masked = K.copy()
K_masked[3:, :] = 0  # Zero out velocity, orientation, bias corrections
```

**Impact:** Prevented vision from corrupting orientation estimates. Later refined to use proper H matrix structure.

---

## 3. Second Wave: ZUPT and Ground Constraints

### 3.1 Zero-Velocity Update (ZUPT)

**Problem:** During stationary periods, small IMU biases integrated into large velocity/position errors.

**Solution:** Implemented formal EKF ZUPT when gyro activity < 0.02 rad/s:
```python
def _zupt_velocity_update(self):
    """Formal EKF update: z = [0,0,0], H selects velocity"""
    H = np.zeros((3, 15))
    H[0:3, 3:6] = np.eye(3)  # Velocity selection
    # Standard EKF update with z_meas = [0, 0, 0]
```

**Impact:** Collapsed velocity uncertainty during stops, preventing drift accumulation.

### 3.2 ZUPT Gravity Tilt Correction

**Problem:** Orientation drifted even when stationary due to gyro bias.

**Solution:** Treat accelerometer as tilt sensor during ZUPT:
```python
def _zupt_gravity_update(self, a_corrected):
    """When stationary, accel should read R^T @ g"""
    a_expected = R_wb.T @ g_world
    z_res = a_measured - a_expected
    # EKF update corrects orientation
```

**Impact:** Locked roll/pitch during stationary periods.

### 3.3 ZUPT False Triggering Fix

**Problem:** ZUPT fired during straight-line motion (low gyro but high velocity).

**Root Cause:** Gyro-only detection couldn't distinguish "moving straight" from "stationary."

**Solution:** Added vision-based motion detection:
```python
if self.last_vision_correction > 0.02:  # 2cm
    self.vision_motion_cooldown = 25  # 0.5 seconds at 50Hz

# Only ZUPT if gyro quiet AND no recent vision motion
if gyro_says_stationary and not vision_says_moving:
    self._zupt_velocity_update()
```

**Impact:** Eliminated false ZUPT during straight-line driving.

### 3.4 Yaw Preservation During Re-initialization

**Problem:** When filter diverged and re-initialized from IMU, yaw was lost.

**Root Cause:** Gravity only provides roll/pitch information, not yaw.

**Solution:** Preserve current yaw during re-initialization:
```python
def _reinitialize_orientation(self, a_m, w_m):
    # Extract current yaw before reset
    current_yaw = R.from_quat(...).as_euler('zyx')[0]

    # Compute new roll/pitch from gravity
    new_roll, new_pitch = compute_from_gravity(a_m)

    # Reconstruct quaternion preserving yaw
    self.x[6:10] = R.from_euler('zyx', [current_yaw, new_pitch, new_roll]).as_quat()
```

**Impact:** Maintained heading through filter recovery events.

---

## 4. IMU Processing Improvements

### 4.1 IMU Downsampling (200Hz → 50Hz)

**Problem:** Single-sample spikes in IMU data (50 m/s² acceleration) caused velocity jumps.

**Analysis:** At 200Hz, one bad sample could inject:
```
Δv = 50 m/s² × 0.005s = 0.25 m/s per spike
```

**Solution:** Average every 4 IMU samples before processing:
```python
self.imu_buffer.append({'accel': a_m, 'gyro': w_m})
if len(self.imu_buffer) >= 4:
    a_avg = np.mean([s['accel'] for s in self.imu_buffer], axis=0)
    w_avg = np.mean([s['gyro'] for s in self.imu_buffer], axis=0)
    dt_total = accumulated_dt
    self.predict(dt_total, a_avg, w_avg)
```

**Impact:** Spike of 50 m/s² becomes 12.5 m/s² after averaging with 3 normal samples. Dramatically smoother predictions.

---

## 5. The Diagnostic Breakthrough

### 5.1 The Visibility Problem

After all the above fixes, the filter *still* diverged after 20-30 seconds. The fundamental problem: **we couldn't see what was happening inside the filter.**

Available information was limited to:
- Final position estimate
- Ground truth comparison
- Occasional log messages

This was like debugging a black box.

### 5.2 Implementing Diagnostics Publisher

**Solution:** Added comprehensive diagnostics topic to EKFNode:
```python
def publish_state(self, timestamp):
    # ... existing odometry publishing ...

    # NEW: Publish internal diagnostics
    diag_msg = Float64MultiArray()
    diag_msg.data = [
        # Biases (6 values)
        self.x[10], self.x[11], self.x[12],  # ba_x, ba_y, ba_z
        self.x[13], self.x[14], self.x[15],  # bg_x, bg_y, bg_z

        # Covariance diagonal (15 values)
        self.P[0,0], self.P[1,1], ..., self.P[14,14],

        # Velocity (4 values)
        self.x[3], self.x[4], self.x[5], speed,

        # Vision correction magnitude
        self.last_vision_correction
    ]
    self.pub_diag.publish(diag_msg)
```

### 5.3 Implementing Diagnostic Plots

**Solution:** Enhanced EvaluationNode to record and plot diagnostics:

```python
def diag_callback(self, msg):
    """Record all diagnostic data"""
    self.accel_bias['x'].append(msg.data[0])
    self.accel_bias['y'].append(msg.data[1])
    self.accel_bias['z'].append(msg.data[2])
    # ... record all 26 values

def save_plots(self):
    """Generate 7 comprehensive plots"""
    # Plot 1: Trajectory comparison
    # Plot 2: Position error over time
    # Plot 3: X, Y, Z position comparison
    # Plot 4: Yaw comparison
    # Plot 5: BIAS EVOLUTION (NEW - KEY DIAGNOSTIC)
    # Plot 6: COVARIANCE EVOLUTION (NEW - KEY DIAGNOSTIC)
    # Plot 7: VELOCITY EVOLUTION (NEW - KEY DIAGNOSTIC)
```

### 5.4 What the Plots Revealed

Running the filter and examining the new diagnostic plots revealed **three smoking guns**:

---

## 6. The Three Critical Fixes (From Diagnostic Analysis)

### 6.1 Suspect #1: FEJ Implementation Error

**What the plots showed:** During turns, position error spiked dramatically, then partially recovered, but with a permanent offset.

**Forensic analysis:** The First-Estimate Jacobian (FEJ) stored the rotation matrix when a landmark was first observed. But the measurement residual used the *current* rotation. This mismatch caused the gradient to point in the wrong direction during turns.

```python
# PROBLEMATIC CODE:
R_fej = self.landmark_first_estimates[lm_id]['R']  # Old rotation
# But residual computed with current R_wb!
z_res = z_meas - project(R_wb, ...)  # Current rotation
```

**Fix:** Disable FEJ, use current rotation for Jacobian:
```python
# Use CURRENT rotation for Jacobian (not FEJ)
J_pos = -self.R_b_c @ R_wb.T  # Current R_wb, not R_fej
```

**Impact:** Eliminated position jumps during turns.

### 6.2 Suspect #2: ba_z Saturation

**What the plots showed:** `ba_z` (Z-accelerometer bias) rapidly saturated to -0.5 m/s² (the clipping limit) and stayed there.

**Forensic analysis:** For a ground robot with `v_z = 0` enforced, `ba_z` is **completely unobservable**. It's perfectly coupled with gravity: any `ba_z` error is compensated by a roll/pitch error. The filter was pushing `ba_z` to absorb model mismatches.

**Fix:** Lock `ba_z` to zero:
```python
# In predict():
self.x[12] = 0.0  # ba_z locked to zero

# In covariance initialization:
self.P[11, :] = 0  # Zero ba_z row
self.P[:, 11] = 0  # Zero ba_z column

# In process noise:
# Q_ba_z = 0 (no random walk)
```

**Impact:** Eliminated mysterious orientation drift caused by `ba_z` fighting with gravity.

### 6.3 Suspect #3: dx Zeroing Breaking Covariance

**What the plots showed:** Covariance grew unboundedly for velocity and orientation states, even though vision updates were occurring.

**Forensic analysis:** Earlier, we implemented position-only updates by zeroing `dx[3:15]`:
```python
# PROBLEMATIC CODE:
dx = K @ z_res
dx[3:6] = 0.0   # Zero velocity correction
dx[6:9] = 0.0   # Zero orientation correction
dx[9:15] = 0.0  # Zero bias corrections
```

This broke the covariance update. The Kalman gain `K` was computed assuming full correction would be applied. Zeroing parts of `dx` while using full `K` for covariance update created inconsistency.

**Fix:** Apply full correction with conservative limits:
```python
# Apply full dx, but with safety limits
dx = K @ z_res

# Limit corrections instead of zeroing
dx[0:3] = np.clip(dx[0:3], -1.5, 1.5)    # Position ±1.5m
dx[3:6] = np.clip(dx[3:6], -0.5, 0.5)    # Velocity ±0.5m/s
dx[6:9] = np.clip(dx[6:9], -0.3, 0.3)    # Orientation ±17°
dx[9:15] = np.clip(dx[9:15], -0.1, 0.1)  # Biases ±0.1

self.x += dx  # Apply full (limited) correction
```

**Impact:** Covariance now properly bounded by vision updates. Cross-correlations maintained.

---

## 7. Evolution of Diagnostic Capability

| Stage | Visibility | Debugging Approach |
|-------|------------|-------------------|
| Initial | Position only | "It drifts... somewhere" |
| + GT comparison | Position error | "Error grows over time" |
| + Logging | Occasional prints | "Something spikes at t=23s" |
| **+ Diagnostics** | **Full internal state** | **"ba_z saturates → causes tilt → velocity explodes"** |

The diagnostic plots transformed debugging from guesswork to forensic analysis.

---

## 8. Final System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        EKF Node                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │   IMU    │───▶│Downsample│───▶│ Predict  │───▶│  State   │  │
│  │ 200Hz    │    │  4:1     │    │ (50Hz)   │    │ Buffer   │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│                                        │                        │
│                                        ▼                        │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                  │
│  │  Vision  │───▶│   IEKF   │───▶│  State   │                  │
│  │ Markers  │    │  Update  │    │ Publish  │                  │
│  └──────────┘    └──────────┘    └──────────┘                  │
│                                        │                        │
│                       ┌────────────────┴────────────────┐      │
│                       ▼                                  ▼      │
│                 /vio/odom                        /vio/diagnostics│
└─────────────────────────────────────────────────────────────────┘
                        │                                  │
                        ▼                                  ▼
              ┌─────────────────────────────────────────────────┐
              │               Eval Node                          │
              │  • Record trajectories                           │
              │  • Record diagnostics                            │
              │  • Generate 7 plots on exit                      │
              └─────────────────────────────────────────────────┘
```

---

## 9. Lessons Learned

### 9.1 Technical Lessons

1. **Observability matters:** Unobservable states (yaw, ba_z) must be constrained or they absorb errors.

2. **Consistency is critical:** Jacobian and residual must use the same linearization point.

3. **Covariance updates must match state updates:** Zeroing part of dx while using full K breaks the filter.

4. **Ground robots are special:** Z-constraints, locked ba_z, and ZUPT are essential.

5. **IMU preprocessing helps:** Downsampling smooths noise before it enters the filter.

### 9.2 Process Lessons

1. **Visibility enables debugging:** Without diagnostics, we were blind. With diagnostics, root causes became obvious.

2. **Plot everything:** Time-series plots of internal states reveal dynamics that logs miss.

3. **Forensic analysis works:** Correlating when errors grow with what states change identifies root causes.

4. **Fix one thing at a time:** Each fix was validated before moving to the next.

---

## 10. Summary of All Changes

| Category | Change | Impact |
|----------|--------|--------|
| **Physics** | Gravity sign fix (subtract, not add) | Correct acceleration model |
| **Formulation** | Robocentric velocity (body frame) | Decouple yaw from velocity |
| **Vision** | Position-only H matrix | Prevent spurious orientation updates |
| **Vision** | Per-observation 100px gate | Reject bad measurements |
| **Vision** | IEKF iterations (up to 4) | Handle large residuals |
| **Vision** | Require 2+ markers | Observability guarantee |
| **ZUPT** | Formal velocity update | Proper covariance collapse |
| **ZUPT** | Gravity tilt correction | Lock roll/pitch when stationary |
| **ZUPT** | Vision motion detection | Prevent false triggering |
| **Re-init** | Preserve yaw | Maintain heading through recovery |
| **IMU** | 4:1 downsampling | Smooth out spikes |
| **Ground** | Z position/velocity = 0 | Ground plane constraint |
| **Ground** | Roll/pitch clamp ±10° | Physical limit |
| **FEJ** | Disabled (use current R) | Correct gradient direction |
| **Bias** | ba_z locked to zero | Remove unobservable state |
| **Update** | Full dx with limits | Maintain covariance consistency |
| **Diagnostics** | Bias/covariance/velocity plots | Enable forensic debugging |

---

## 11. Conclusion

The VIO ES-EKF project evolved from a diverging prototype to a stable localization system through systematic debugging. The key breakthrough was implementing comprehensive diagnostic plotting, which transformed opaque filter behavior into interpretable time-series data. This visibility enabled identification of three critical issues (FEJ, ba_z, dx zeroing) that would have been nearly impossible to find otherwise.

The final system demonstrates that robust VIO requires not just correct mathematics, but also:
- Proper handling of unobservable states
- Consistency between all filter components
- Domain-specific constraints (ground robot physics)
- Comprehensive internal visibility for debugging

**The diagnostic infrastructure is as valuable as the filter itself.**

---

*Report generated: January 2026*
