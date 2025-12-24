Based on the "EE585_Probabilistic_Robotics_Term_Project" source, here is the implementation plan for Çağdaş's role (Sensor Data Retrieval) and the answers to your specific design questions.

### 1. Sensor Specifications

*   **Camera Type:** **Monocular Pinhole Camera**.
    *   The project specifies a **Monocular Camera**.
    *   While the literature survey mentions fisheye models (e.g., ORB-SLAM3), the methodology describes a standard projection model $h(x_{nom})$ for the residual $r = z_{meas} - h(x_{nom})$. A standard pinhole model is recommended for the "Basic EKF" to simplify the Jacobian derivations.
*   **IMU Behavior:**
    *   **Output:** High-frequency linear acceleration ($a_m$) and angular velocity ($\omega_m$).
    *   **Errors:** It suffers from **measurement noise** and **time-varying biases** (initial and drifting).
    *   **Drift:** Computing position requires double integration of acceleration, causing errors to grow **quadratically** over time if left uncorrected.

### 2. Algorithm to Handle IMU: Error-State EKF (ES-EKF)

You should use the **Error-State Extended Kalman Filter (ES-EKF)**. The IMU handles the **Prediction Step** (propagation) of the filter.

**The ES-EKF Prediction Logic:**
1.  **Nominal State ($x$):** Propagate the "ideal" state using non-linear kinematics.
    *   $\dot{p} = v$
    *   $\dot{v} = R(q)(a_m - b_a) - g$
    *   $\dot{q} = \frac{1}{2} q \otimes (\omega_m - b_g)$
2.  **Error State ($\delta x$):** Propagate the uncertainty (covariance $P$).
    *   $P_{k|k-1} = F_x P_{k-1|k-1} F_x^T + Q_{imu}$.
3.  **Bias Correction:** Explicitly model and subtract biases ($b_a, b_g$) from raw measurements before integration. These biases are estimated in the state vector and updated during the camera correction step.

### 3. Implementation Plan for Çağdaş (Data Retrieval)

Çağdaş is responsible for retrieving and validating sensor data from Gazebo Fortress via ROS 2.

**Step 1: Create the ROS 2 Node**
You need a node to subscribe to the Gazebo bridge topics and synchronize them.

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, Image
from nav_msgs.msg import Odometry # For Ground Truth validation

class SensorInterface(Node):
    def __init__(self):
        super().__init__('sensor_interface')

        # 1. Subscribe to IMU (High Rate ~200Hz)
        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',  # Ensure this matches ros_gz_bridge output
            self.imu_callback,
            10)

        # 2. Subscribe to Camera (Low Rate ~30Hz)
        self.cam_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.cam_callback,
            10)

        # 3. Subscribe to Ground Truth (for validation deliverables)
        self.gt_sub = self.create_subscription(
            Odometry,
            '/ground_truth/odom',
            self.gt_callback,
            10)

    def imu_callback(self, msg):
        # TODO: Feed this to ES-EKF Prediction Step
        # Extract angular_velocity and linear_acceleration
        pass

    def cam_callback(self, msg):
        # TODO: Feed this to Feature Extractor -> ES-EKF Correction
        pass

    def gt_callback(self, msg):
        # Store for ATE (Absolute Trajectory Error) calculation
        pass
```
