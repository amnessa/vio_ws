#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, CameraInfo
from geometry_msgs.msg import PoseArray, PoseStamped, TransformStamped
from nav_msgs.msg import Path, Odometry
from tf2_ros import TransformBroadcaster

import numpy as np
from scipy.spatial.transform import Rotation as R

class EKFNode(Node):
    def __init__(self):
        super().__init__('ekf_node')

        # --- Initialization Phase ---
        # Collect IMU samples while stationary to estimate biases and initial orientation
        self.initialized = False
        self.init_samples = []
        self.init_sample_count = 800  # Number of samples to collect (at 200Hz = 4 seconds)
        self.gravity_magnitude = 9.81  # Expected gravity magnitude (will be updated from IMU)
        self.g = np.array([0.0, 0.0, 9.81])  # Gravity vector in world frame (will be updated)

        # --- State Definitions ---
        # State: [p(3), v(3), q(4), ba(3), bg(3)] = 16 elements
        # Error State: [dp(3), dv(3), dtheta(3), dba(3), dbg(3)] = 15 elements
        self.x = np.zeros(16)
        self.x[6] = 1.0  # Initial quaternion (w=1, x=0, y=0, z=0)

        # Covariance Matrix (15x15)
        self.P = np.eye(15) * 0.1
        self.P[0:3, 0:3] *= 0.01  # Small initial position uncertainty
        self.P[6:9, 6:9] *= 0.01 # Initial orientation uncertainty
        self.P[9:12, 9:12] *= 0.01 # Accel bias uncertainty
        self.P[12:15, 12:15] *= 0.001 # Gyro bias uncertainty

        # Noise Parameters - Updated based on observed IMU statistics
        # From initialization: accel_std ~0.1 m/s^2, gyro_std ~0.05 rad/s (Y axis)
        # These are higher than datasheet due to Gazebo simulation noise
        self.sigma_a = 0.15     # Accel noise stddev (m/s^2) - conservative based on observed
        self.sigma_g = 0.01     # Gyro noise stddev (rad/s) - conservative based on observed
        self.Q_a = self.sigma_a ** 2  # Accel noise variance
        self.Q_g = self.sigma_g ** 2  # Gyro noise variance
        self.Q_ba = 1e-4  # Accel bias random walk (increased for faster adaptation)
        self.Q_bg = 1e-5  # Gyro bias random walk (increased for faster adaptation)
        self.R_cam = 5.0  # Pixel measurement noise (variance)

        # Time sync tolerance (max acceptable delay between IMU and vision)
        self.max_time_delay = 0.05  # 50ms tolerance

        # Landmarks (Known Map from landmarks.sdf)
        # ID 1: Red, ID 2: Green
        self.map = {
            1.0: np.array([2.0, 2.0, 0.5]),
            2.0: np.array([4.0, -2.0, 0.5])
        }

        # Calibration (Extrinsics: Base -> Camera)
        # From Waffle Pi SDF: x=0.064, y=-0.065, z=0.094
        self.T_b_c = np.eye(4)
        self.T_b_c[0:3, 3] = [0.064, -0.065, 0.094]
        # Rotation: Standard camera Z-forward vs ROS X-forward usually requires check
        # For now assuming Identity rotation relative to base for simplicity,
        # or we fix in camera_info callback if we get extrinsic there.
        # Waffle Pi camera points forward (X).
        # But OpenCV projection assumes Z is depth.
        # Transform Optical (Z-forward) to Standard (X-forward):
        # R_opt_std = [0 0 1; -1 0 0; 0 -1 0]
        self.R_b_c = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])

        # Camera Intrinsics (will be updated callback)
        self.K = np.array([[530.0, 0, 320.0], [0, 530.0, 240.0], [0, 0, 1]])

        # ROS Infrastructure
        self.last_imu_time = None
        self.path_msg = Path()

        # Ground truth trajectory for evaluation (ATE calculation)
        self.gt_trajectory = []
        self.est_trajectory = []

        self.sub_imu = self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.sub_vision = self.create_subscription(PoseArray, '/vio/landmarks', self.vision_callback, 10)
        self.sub_cam_info = self.create_subscription(CameraInfo, '/camera_info', self.info_callback, 10)
        # Ground truth subscription for validation/evaluation
        self.sub_gt = self.create_subscription(Odometry, '/ground_truth/odom', self.gt_callback, 10)

        self.pub_odom = self.create_publisher(Odometry, '/vio/odom', 10)
        self.pub_path = self.create_publisher(Path, '/vio/path', 10)
        self.tf_br = TransformBroadcaster(self)

        self.get_logger().info("EKF Node Initialized")

    def info_callback(self, msg):
        # Update intrinsics matrix K
        self.K = np.array(msg.k).reshape(3,3)

    def imu_callback(self, msg):
        # Extract measurements
        a_m = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
        w_m = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])

        # --- INITIALIZATION PHASE ---
        # Collect samples while robot is stationary to estimate biases and initial orientation
        if not self.initialized:
            self.init_samples.append({'accel': a_m.copy(), 'gyro': w_m.copy()})

            if len(self.init_samples) >= self.init_sample_count:
                self.initialize_from_imu()
            else:
                if len(self.init_samples) % 20 == 0:
                    self.get_logger().info(f"Initializing... {len(self.init_samples)}/{self.init_sample_count} samples")
            return

        curr_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if self.last_imu_time is None:
            self.last_imu_time = curr_time
            return

        dt = curr_time - self.last_imu_time

        # --- TIME JUMP SAFETY ---
        # If dt is larger than 100ms (we expect ~5ms at 200Hz), skip this message
        # This prevents huge integration errors from bag file jumps or pauses
        if dt > 0.1:
            self.get_logger().warn(f"Huge time jump detected (dt={dt:.4f}s). Skipping prediction step.")
            self.last_imu_time = curr_time
            return
        if dt <= 0:
            return # Skip backwards or duplicate messages
        self.last_imu_time = curr_time

        # --- PREDICTION STEP ---
        self.predict(dt, a_m, w_m)
        self.publish_state(msg.header.stamp)

    def initialize_from_imu(self):
        """
        Initialize the EKF state from stationary IMU readings.

        1. Average gyro readings to get gyroscope bias
        2. Average accel readings to get gravity direction + accel bias
        3. Compute initial orientation from gravity vector
        """
        self.get_logger().info("Computing initial biases from IMU samples...")

        # Stack all samples
        accels = np.array([s['accel'] for s in self.init_samples])
        gyros = np.array([s['gyro'] for s in self.init_samples])

        # Average readings (should be stationary)
        accel_mean = np.mean(accels, axis=0)
        gyro_mean = np.mean(gyros, axis=0)

        # Check if readings are consistent (low variance = stationary)
        accel_std = np.std(accels, axis=0)
        gyro_std = np.std(gyros, axis=0)

        self.get_logger().info(f"Accel mean: [{accel_mean[0]:.4f}, {accel_mean[1]:.4f}, {accel_mean[2]:.4f}] m/s^2")
        self.get_logger().info(f"Accel std:  [{accel_std[0]:.4f}, {accel_std[1]:.4f}, {accel_std[2]:.4f}] m/s^2")
        self.get_logger().info(f"Gyro mean:  [{gyro_mean[0]:.6f}, {gyro_mean[1]:.6f}, {gyro_mean[2]:.6f}] rad/s")
        self.get_logger().info(f"Gyro std:   [{gyro_std[0]:.6f}, {gyro_std[1]:.6f}, {gyro_std[2]:.6f}] rad/s")

        # --- Gyroscope Bias ---
        # When stationary, gyro should read zero. Any reading is bias.
        self.x[13:16] = gyro_mean  # bg = gyro_mean

        # --- Initial Orientation from Gravity ---
        # The accelerometer measures the reaction to gravity.
        # When stationary: a_measured = R_wb^T @ [0, 0, g] (assuming world Z is up)
        # So the gravity vector in body frame tells us the orientation.

        accel_norm = np.linalg.norm(accel_mean)
        if abs(accel_norm - self.gravity_magnitude) > 1.0:
            self.get_logger().warn(f"Accel magnitude {accel_norm:.2f} differs from expected {self.gravity_magnitude:.2f}")

        # Normalize to get gravity direction in body frame
        gravity_body = accel_mean / accel_norm

        # Gravity in world frame (pointing UP, since accelerometer measures reaction)
        gravity_world = np.array([0.0, 0.0, 1.0])

        # Find rotation that aligns gravity_body with gravity_world
        # This gives us R_wb (rotation from body to world)
        # Using Rodrigues' formula: find axis-angle from cross product

        v = np.cross(gravity_body, gravity_world)
        s = np.linalg.norm(v)  # sin(angle)
        c = np.dot(gravity_body, gravity_world)  # cos(angle)

        if s < 1e-6:
            # Vectors are parallel
            if c > 0:
                # Already aligned, identity rotation
                R_init = np.eye(3)
            else:
                # Opposite direction, 180 degree rotation around X
                R_init = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        else:
            # Rodrigues formula: R = I + [v]_x + [v]_x^2 * (1-c)/s^2
            vx = np.array([[0, -v[2], v[1]],
                           [v[2], 0, -v[0]],
                           [-v[1], v[0], 0]])
            R_init = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))

        # Convert to quaternion [w, x, y, z]
        rot = R.from_matrix(R_init)
        q_scipy = rot.as_quat()  # [x, y, z, w]
        self.x[6:10] = np.array([q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]])

        # --- Update Gravity Model ---
        # Use the ACTUAL measured magnitude instead of hardcoded 9.81
        # This ensures (R @ accel - gravity) is exactly zero when stationary
        actual_gravity = np.linalg.norm(accel_mean)
        self.gravity_magnitude = actual_gravity
        self.g = np.array([0.0, 0.0, actual_gravity])
        self.get_logger().info(f"Updated Gravity Model: {actual_gravity:.4f} m/s^2")

        # --- Accelerometer Bias ---
        # CRITICAL: Set bias to ZERO!
        # The X/Y components of accel_mean are already explained by the orientation
        # we just computed from gravity. If we also subtract them as bias, we
        # double-compensate and create false acceleration.
        #
        # The filter will estimate any true sensor bias over time.
        self.x[10:13] = np.zeros(3)

        # Log the computed initial tilt for debugging
        initial_roll = np.arctan2(accel_mean[1], accel_mean[2])
        initial_pitch = np.arctan2(-accel_mean[0], np.sqrt(accel_mean[1]**2 + accel_mean[2]**2))
        self.get_logger().info(f"Initial tilt: Roll={np.degrees(initial_roll):.2f} deg, Pitch={np.degrees(initial_pitch):.2f} deg")
        self.get_logger().info(f"Initial orientation (quat): w={self.x[6]:.4f}, x={self.x[7]:.4f}, y={self.x[8]:.4f}, z={self.x[9]:.4f}")
        self.get_logger().info(f"Accel bias: [{self.x[10]:.4f}, {self.x[11]:.4f}, {self.x[12]:.4f}] m/s^2 (all zero)")
        self.get_logger().info(f"Gyro bias:  [{gyro_mean[0]:.6f}, {gyro_mean[1]:.6f}, {gyro_mean[2]:.6f}] rad/s")

        # Clear init samples
        self.init_samples = []
        self.initialized = True
        self.get_logger().info("=== EKF INITIALIZED - Starting state estimation ===")

    def predict(self, dt, a_m, w_m):
        # Unpack State
        p = self.x[0:3]
        v = self.x[3:6]
        q = self.x[6:10]
        ba = self.x[10:13]
        bg = self.x[13:16]

        # Rotation Matrix R_wb
        rot = R.from_quat([q[1], q[2], q[3], q[0]]) # Scipy uses [x, y, z, w]
        R_wb = rot.as_matrix()

        # 1. Nominal State Propagation
        # Bias corrected measurements
        a_unbiased = a_m - ba
        w_unbiased = w_m - bg
        # Use calibrated gravity from initialization (not hardcoded 9.81)

        # Position
        p_new = p + v * dt + 0.5 * (R_wb @ a_unbiased - self.g) * dt**2
        # Velocity
        v_new = v + (R_wb @ a_unbiased - self.g) * dt
        # Orientation (Quaternion integration)
        # exp_map for rotation vector w * dt
        w_norm = np.linalg.norm(w_unbiased)
        if w_norm > 1e-6:
            axis = w_unbiased / w_norm
            angle = w_norm * dt
            # Quaternion for small rotation [sin(a/2)u, cos(a/2)]
            dq = np.hstack([np.sin(angle/2)*axis, np.cos(angle/2)]) # [x,y,z,w]
            # Combine: q_new = q * dq
            # Note: Scipy multiplication is q2 * q1 (q2 applied after)
            q_new_obj = rot * R.from_quat(dq)
            q_new = q_new_obj.as_quat() # [x,y,z,w]
            # Store as [w, x, y, z] for our state convention
            q_new = np.array([q_new[3], q_new[0], q_new[1], q_new[2]])
        else:
            q_new = q

        # Biases (Random Walk) -> Constant in prediction mean
        ba_new = ba
        bg_new = bg

        # Update Nominal State
        self.x[0:3] = p_new
        self.x[3:6] = v_new
        self.x[6:10] = q_new
        self.x[10:13] = ba_new
        self.x[13:16] = bg_new

        # 2. Covariance Propagation (Linearization)
        # Fx (15x15) Jacobian of error state
        Fx = np.eye(15)

        # Position block
        Fx[0:3, 3:6] = np.eye(3) * dt

        # Velocity block
        # d(v)/d(theta) = -R * [a]_x * dt
        a_skew = np.array([[0, -a_unbiased[2], a_unbiased[1]],
                           [a_unbiased[2], 0, -a_unbiased[0]],
                           [-a_unbiased[1], a_unbiased[0], 0]])
        Fx[3:6, 6:9] = -R_wb @ a_skew * dt
        Fx[3:6, 9:12] = -R_wb * dt # d(v)/d(ba)

        # Orientation block
        # d(theta)/d(theta) = I - [w]_x * dt (approx for small dt)
        # d(theta)/d(bg) = -I * dt
        Fx[6:9, 12:15] = -np.eye(3) * dt

        # Noise Matrix Q (discrete)
        # Simple approx: Q = G * Q_continuous * G.T * dt
        # We assume diagonal Q for simplicity here
        Qi = np.zeros((15, 15))
        Qi[3:6, 3:6] = np.eye(3) * self.Q_a * dt**2 # v noise from accel
        Qi[6:9, 6:9] = np.eye(3) * self.Q_g * dt**2 # theta noise from gyro
        Qi[9:12, 9:12] = np.eye(3) * self.Q_ba * dt # ba random walk
        Qi[12:15, 12:15] = np.eye(3) * self.Q_bg * dt # bg random walk

        self.P = Fx @ self.P @ Fx.T + Qi

    def gt_callback(self, msg):
        """
        Store ground truth pose for later ATE (Absolute Trajectory Error) calculation.
        """
        gt_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        gt_pos = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])
        gt_quat = np.array([
            msg.pose.pose.orientation.w,
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z
        ])
        self.gt_trajectory.append({
            'time': gt_time,
            'position': gt_pos,
            'orientation': gt_quat
        })
        # Keep trajectory buffer bounded
        if len(self.gt_trajectory) > 2000:
            self.gt_trajectory.pop(0)

    def vision_callback(self, msg):
        """
        Handle visual landmark observations with time synchronization check.
        """
        # Skip if not initialized
        if not self.initialized:
            return

        # Time synchronization check
        if self.last_imu_time is None:
            # No IMU data yet, skip vision update
            self.get_logger().warn("Vision update skipped: No IMU data received yet", throttle_duration_sec=2.0)
            return

        vision_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        time_diff = abs(vision_time - self.last_imu_time)

        if time_diff > self.max_time_delay:
            self.get_logger().warn(
                f"Vision-IMU time mismatch: {time_diff:.3f}s > {self.max_time_delay}s. "
                "Consider adjusting synchronization.",
                throttle_duration_sec=5.0
            )
            # Still proceed but log the warning for debugging

        # msg is PoseArray. x=u, y=v, z=id
        for obs in msg.poses:
            lid = obs.position.z
            u_meas = obs.position.x
            v_meas = obs.position.y

            if lid in self.map:
                self.update(self.map[lid], u_meas, v_meas)

    def update(self, lm_pos_world, u_meas, v_meas):
        # --- UPDATE STEP ---
        # 1. Project Map Point to Camera
        p_w = self.x[0:3]
        q_w = self.x[6:10] # [w,x,y,z]
        R_wb = R.from_quat([q_w[1], q_w[2], q_w[3], q_w[0]]).as_matrix()

        # Transform World -> Body
        # p_b = R_wb^T * (p_lm - p_w)
        p_b = R_wb.T @ (lm_pos_world - p_w)

        # Transform Body -> Camera
        # Assuming fixed offset T_bc
        # p_c = R_bc.T * (p_b - t_bc)  <-- Wait, usually p_c = R_bc^T * p_b - R_bc^T*t_bc
        # Let's use standard homogeneous transform
        # T_wc = T_wb * T_bc
        # p_c = T_wc^-1 * p_w_lm

        # Simplified: We defined R_b_c manually in init
        # Correct Body->Camera (Rotation only for now, ignoring small translation for Jacobian approx)
        # p_c = self.R_b_c @ p_b

        # Full Transform:
        t_bc = self.T_b_c[0:3, 3]
        # Since R_b_c provided matches Optical frame
        p_c = self.R_b_c @ (p_b - t_bc)

        # Check if behind camera
        if p_c[2] < 0.1: return

        # Project to pixels
        # [u, v, 1]^T = K * p_c / z
        u_pred = self.K[0,0] * p_c[0]/p_c[2] + self.K[0,2]
        v_pred = self.K[1,1] * p_c[1]/p_c[2] + self.K[1,2]

        # Residual
        z_res = np.array([u_meas - u_pred, v_meas - v_pred])

        # 2. Calculate Jacobian H (2x15)
        # H = d(res)/d(x) = d(res)/d(pc) * d(pc)/d(x)

        # Jacobian of Projection: d(uv)/d(pc)
        fx = self.K[0,0]
        fy = self.K[1,1]
        X, Y, Z = p_c
        J_proj = np.array([
            [fx/Z, 0, -fx*X/Z**2],
            [0, fy/Z, -fy*Y/Z**2]
        ])

        # Jacobian of Transform: d(pc)/d(x)
        # Only dependence is on robot position and orientation
        # p_c = R_bc * R_wb^T * (p_lm - p_w) - ...

        # w.r.t Position p_w:
        # d(pc)/d(pw) = R_bc * R_wb^T * (-I)
        J_pos = -self.R_b_c @ R_wb.T

        # w.r.t Orientation theta (in body frame):
        # d(pc)/d(theta) = R_bc * [p_b]x (skew symmetric of point in body)
        p_b_skew = np.array([[0, -p_b[2], p_b[1]],
                             [p_b[2], 0, -p_b[0]],
                             [-p_b[1], p_b[0], 0]])
        J_rot = self.R_b_c @ p_b_skew

        # Assemble H (2x15)
        H = np.zeros((2, 15))
        H[:, 0:3] = J_proj @ J_pos
        H[:, 6:9] = J_proj @ J_rot

        # 3. Kalman Update
        # S = H P H.T + R
        S = H @ self.P @ H.T + np.eye(2) * self.R_cam

        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return

        # Error State Update
        dx = K @ z_res

        # Update Covariance
        I = np.eye(15)
        self.P = (I - K @ H) @ self.P

        # 4. Inject Error into Nominal State
        # Position
        self.x[0:3] += dx[0:3]
        # Velocity
        self.x[3:6] += dx[3:6]
        # Orientation (Quaternion multiply)
        dq_rot = R.from_rotvec(dx[6:9])
        q_curr_obj = R.from_quat([self.x[7], self.x[8], self.x[9], self.x[6]]) # [x,y,z,w]
        q_new_obj = q_curr_obj * dq_rot
        q_new = q_new_obj.as_quat()
        self.x[6:10] = np.array([q_new[3], q_new[0], q_new[1], q_new[2]])
        # Biases
        self.x[10:13] += dx[9:12]
        self.x[13:16] += dx[12:15]

        # Reset Error State (Implied dx=0 for next step)

    def publish_state(self, timestamp):
        # Publish Odometry
        odom = Odometry()
        odom.header.stamp = timestamp
        odom.header.frame_id = "map"
        odom.child_frame_id = "base_footprint"

        odom.pose.pose.position.x = self.x[0]
        odom.pose.pose.position.y = self.x[1]
        odom.pose.pose.position.z = self.x[2]

        odom.pose.pose.orientation.w = self.x[6]
        odom.pose.pose.orientation.x = self.x[7]
        odom.pose.pose.orientation.y = self.x[8]
        odom.pose.pose.orientation.z = self.x[9]

        odom.twist.twist.linear.x = self.x[3]
        odom.twist.twist.linear.y = self.x[4]
        odom.twist.twist.linear.z = self.x[5]

        self.pub_odom.publish(odom)

        # Store estimated trajectory for ATE calculation
        est_time = timestamp.sec + timestamp.nanosec * 1e-9
        self.est_trajectory.append({
            'time': est_time,
            'position': self.x[0:3].copy(),
            'orientation': self.x[6:10].copy()
        })
        # Keep trajectory buffer bounded
        if len(self.est_trajectory) > 2000:
            self.est_trajectory.pop(0)

        # Broadcast Transform
        t = TransformStamped()
        t.header = odom.header
        t.child_frame_id = odom.child_frame_id
        t.transform.translation.x = self.x[0]
        t.transform.translation.y = self.x[1]
        t.transform.translation.z = self.x[2]
        t.transform.rotation = odom.pose.pose.orientation
        self.tf_br.sendTransform(t)

        # Path Visualization
        pose = PoseStamped()
        pose.header = odom.header
        pose.pose = odom.pose.pose
        self.path_msg.header = odom.header
        self.path_msg.poses.append(pose)
        if len(self.path_msg.poses) > 500: self.path_msg.poses.pop(0)
        self.pub_path.publish(self.path_msg)

    def compute_ate(self):
        """
        Compute Absolute Trajectory Error (ATE) as RMSE between
        estimated and ground truth trajectories.

        Returns:
            float: RMSE of position error in meters, or None if insufficient data
        """
        if len(self.gt_trajectory) < 10 or len(self.est_trajectory) < 10:
            self.get_logger().warn("Insufficient trajectory data for ATE computation")
            return None

        # Match trajectories by closest timestamp
        errors = []
        for est in self.est_trajectory:
            # Find closest ground truth by time
            min_dt = float('inf')
            closest_gt = None
            for gt in self.gt_trajectory:
                dt = abs(est['time'] - gt['time'])
                if dt < min_dt:
                    min_dt = dt
                    closest_gt = gt

            # Only use if time difference is acceptable (< 50ms)
            if closest_gt is not None and min_dt < 0.05:
                pos_error = np.linalg.norm(est['position'] - closest_gt['position'])
                errors.append(pos_error ** 2)

        if len(errors) == 0:
            self.get_logger().warn("No matching trajectory points found for ATE")
            return None

        rmse = np.sqrt(np.mean(errors))
        self.get_logger().info(f"ATE (RMSE): {rmse:.4f} m over {len(errors)} points")
        return rmse

    def compute_nees(self):
        """
        Compute Normalized Estimation Error Squared (NEES) for filter consistency.
        This checks if the filter's uncertainty (covariance) matches the actual errors.

        For a consistent filter, NEES ~ chi-squared with DOF = state dimension
        Expected value: DOF (15 for our error state)
        """
        if len(self.gt_trajectory) < 1 or len(self.est_trajectory) < 1:
            return None

        # Get most recent estimates
        est = self.est_trajectory[-1]

        # Find closest ground truth
        min_dt = float('inf')
        closest_gt = None
        for gt in self.gt_trajectory:
            dt = abs(est['time'] - gt['time'])
            if dt < min_dt:
                min_dt = dt
                closest_gt = gt

        if closest_gt is None or min_dt > 0.05:
            return None

        # Position error (3 DOF)
        pos_error = est['position'] - closest_gt['position']
        P_pos = self.P[0:3, 0:3]

        try:
            nees_pos = pos_error.T @ np.linalg.inv(P_pos) @ pos_error
            return nees_pos
        except np.linalg.LinAlgError:
            return None

def main(args=None):
    rclpy.init(args=args)
    node = EKFNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()