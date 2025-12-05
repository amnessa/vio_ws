Project 3: Visual-Inertial EKF Navigation Checklist

Phase 1: Environment Setup (Simulated World)

[x] Build Package: Create the vio_ekf ROS2 package structure. (In Progress)

[ ] World: Verify landmarks.sdf loads in Gazebo Fortress with colored cylinders.

[ ] Robot: Verify the robot spawns with IMU/Camera plugins loaded.

[ ] Sensors: Check topics /imu and /camera via ros2 topic list.

[ ] Ground Truth: Verify pose_tf_broadcaster is publishing TFs from Map -> Odom -> Base.

Phase 2: Visual Frontend (The Eyes)

[ ] Feature Extraction: Implement simple color thresholding in vision_node.py.

[ ] Data Association: For this phase, we assume Known Correspondence (Red=1, Green=2).

[ ] Test: Echo /vio/landmarks. Do pixel coordinates change as you move?

Phase 3: The EKF Backend (The Brain)

[ ] State Vector: Define 16-element state: $[p, v, q, b_a, b_g]$.

[ ] Prediction Step (IMU): Dead Reckoning implementation.

[ ] Update Step (Vision): Jacobian $H$ and Innovation.

[ ] Bias Estimation: Monitor convergence of $b_a, b_g$.

Phase 4: Analysis

[ ] Comparison: Compare Dead Reckoning vs. VIO vs. Ground Truth.