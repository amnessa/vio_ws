**Project 3: Visual-Inertial ES-EKF Navigation Checklist**

**Phase 1: Environment Setup (Simulated World)**

* [x] **Build Package:** Create the `vio_ekf` ROS2 package structure.
* [x] **World:** Verify `landmarks.sdf` loads in Gazebo Fortress with colored cylinders.
* [x] **Robot:** Verify the robot spawns with IMU/Camera plugins loaded.
* [x] **Sensors:** Check topics `/imu` and `/camera` via `ros2 topic list`.
* [x] **Ground Truth:** Verify `pose_tf_broadcaster` is publishing TFs from Map -> Odom -> Base.

**Phase 2: Visual Frontend (Visual Data Processing)**

* [x] **Feature Extraction:** Implement simple color thresholding in `vision_node.py`.


* [ ] **Pose Estimation:** Ensure vision node outputs an estimated pose or pixel coordinates ready for the EKF update.


* [ ] **Data Synchronization:** Ensure visual detection messages and IMU data are time-synced for the filter.



**Phase 3: The ES-EKF Backend (Implementation)**

* [ ] **State Structures:** Define classes for Nominal State () and Error State ().


* [ ] **Prediction Step (IMU):**
* [ ] Implement Runge-Kutta or Euler integration for Nominal State.


* [ ] Compute Jacobian  and propagate Covariance .




* [ ] **Update Step (Vision):**
* [ ] Compute Jacobian  (observation w.r.t error state).


* [ ] Compute Kalman Gain  and estimate error state .




* [ ] **Injection & Reset:** Implement the injection of  into  and reset  to zero.


* [ ] **Bias Initialization:** Ensure biases  are initialized and estimated correctly.



**Phase 4: Evaluation & Analysis**

* [ ] **Trajectory Validation:** Plot Estimated Path vs. Ground Truth.


* [ ] **ATE Metric:** Calculate Absolute Trajectory Error (RMSE).


* [ ] **NEES Metric:** Calculate Normalized Estimation Error Squared to verify filter consistency.


* [ ] **Drift Analysis:** Compare "IMU Only" (Dead Reckoning) vs. "Visual-Inertial" (Fusion) modes.



**Next Step:** Would you like to start implementing the **State Structures** for the ES-EKF (Phase 3), or should we finalize the **Visual Frontend** (Phase 2) data synchronization first?