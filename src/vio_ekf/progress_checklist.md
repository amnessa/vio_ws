**Project 3: Visual-Inertial ES-EKF Navigation Checklist**

**Phase 1: Environment Setup (Simulated World)**

* [x] **Build Package:** Create the `vio_ekf` ROS2 package structure.
* [x] **World:** Verify `landmarks.sdf` loads in Gazebo Fortress with colored cylinders.
* [x] **Robot:** Verify the robot spawns with IMU/Camera plugins loaded.
* [x] **Sensors:** Check topics `/imu` and `/camera` via `ros2 topic list`.
* [x] **Ground Truth:** Verify `pose_tf_broadcaster` is publishing TFs from Map -> Odom -> Base.

**Phase 2: Visual Frontend (Visual Data Processing)**

* [x] **Feature Extraction:** Implement simple color thresholding in `vision_node.py`.
* [x] **Pose Estimation:** Vision node outputs pixel coordinates (u, v, id) ready for EKF update.
* [x] **Data Synchronization:** Added time-sync check between visual detections and IMU data with configurable tolerance.

**Phase 3: The ES-EKF Backend (Implementation)**

* [x] **State Structures:** Nominal State (16D) and Error State (15D) defined in `ekf_node.py`.
* [x] **Noise Parameters:** Updated from Gazebo IMU SDF (σ_a=1.7e-2, σ_g=2e-4).

* [x] **Prediction Step (IMU):**
  * [x] Implement Euler integration for Nominal State propagation.
  * [x] Compute Jacobian Fx and propagate Covariance P.

* [x] **Update Step (Vision):**
  * [x] Compute Jacobian H (observation w.r.t error state).
  * [x] Compute Kalman Gain K and estimate error state δx.

* [x] **Injection & Reset:** Injection of δx into nominal state implemented (additive for p,v,b; multiplicative for q).
* [x] **Bias Initialization:** Biases (ba, bg) initialized and estimated in state vector.

**Phase 4: Evaluation & Analysis**

* [x] **Ground Truth Logging:** Added ground truth subscription and trajectory storage.
* [x] **Trajectory Validation:** Path visualization published to `/vio/path`.
* [x] **ATE Metric:** Implemented `compute_ate()` method for RMSE calculation.
* [x] **NEES Metric:** Implemented `compute_nees()` method for filter consistency check.


* [ ] **Drift Analysis:** Compare "IMU Only" (Dead Reckoning) vs. "Visual-Inertial" (Fusion) modes.


**Next Steps:**
1. Run the full system to validate sensor data flow
2. Perform drift analysis experiments
3. Generate trajectory plots for the final report