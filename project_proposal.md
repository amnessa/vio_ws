# Visual–Inertial Extended Kalman Filter Navigation

## Objective:

Implement a basic EKF that fuses simulated camera (landmarks or optical flow) and planar Inertial

Measurement Unit (IMU) measurements for navigation (body state estimation).

## Tasks:
* Simulate 2D planar environment possibly with 3D features (for camera images) and known

landmarks.

* Generate IMU readings (acceleration + gyro) and camera pixel observations.
* Implement an EKF where the state includes position, velocity, orientation, and landmark

features.

* Compare IMU-only, vision only and vision+IMU fusion results.
## Deliverables:
* EKF implementation and plots of pose estimates vs. ground truth.

* Report analyzing drift reduction from visual updates.

* Use quaternion-based orientation representation.

* Add bias initialization and estimation for the IMU.

* Optional: Also use real data from a public dataset."

Also the project environment is ros2, gazebo. "

