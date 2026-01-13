#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, CameraInfo
from geometry_msgs.msg import PoseArray, PoseStamped, TransformStamped
from nav_msgs.msg import Path, Odometry
from tf2_ros import TransformBroadcaster

import numpy as np
from scipy.spatial.transform import Rotation as R


def skew_symmetric(v):
    """
    Compute the skew-symmetric (cross-product) matrix of a 3D vector.
    Used extensively in Jacobian calculations for ES-EKF.
    [v]_× such that [v]_× @ u = v × u

    Reference: MatthewHampsey/mekf util.py
    """
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ])


def quat_to_rotation_matrix(q):
    """
    Convert quaternion [w, x, y, z] to rotation matrix.
    Uses the closed-form formula for efficiency.

    Reference: MatthewHampsey/mekf util.py (quatToMatrix)
    """
    w, x, y, z = q
    # Using scipy for robustness
    rot = R.from_quat([x, y, z, w])  # scipy uses [x, y, z, w]
    return rot.as_matrix()


class EKFNode(Node):
    def __init__(self):
        super().__init__('ekf_node')

        # --- Initialization Phase ---
        # Collect IMU samples while stationary to estimate biases and initial orientation
        self.initialized = False
        self.init_samples = []
        self.init_sample_count = 2400  # Number of samples to collect (at 200Hz = 12 seconds)
        self.gravity_magnitude = 9.81  # Expected gravity magnitude (will be updated from IMU)
        self.g = np.array([0.0, 0.0, 9.81])  # Gravity vector in world frame (will be updated)

        # --- State Definitions ---
        # State: [p(3), v(3), q(4), ba(3), bg(3)] = 16 elements
        # Error State: [dp(3), dv(3), dtheta(3), dba(3), dbg(3)] = 15 elements
        self.x = np.zeros(16)
        self.x[6] = 1.0  # Initial quaternion (w=1, x=0, y=0, z=0)

        # Covariance Matrix (15x15)
        # Start with higher uncertainty - let vision corrections do their job
        self.P = np.eye(15) * 0.1
        self.P[0:3, 0:3] = np.eye(3) * 0.5   # Position uncertainty (0.5m std)
        self.P[3:6, 3:6] = np.eye(3) * 0.1   # Velocity uncertainty (0.3m/s std)
        self.P[6:9, 6:9] = np.eye(3) * 0.1   # Orientation uncertainty (~18 deg std)
        self.P[9:12, 9:12] = np.eye(3) * 0.01  # Accel bias uncertainty
        self.P[12:15, 12:15] = np.eye(3) * 0.01  # Gyro bias uncertainty

        # Noise Parameters - Tuned for Gazebo simulation
        # Higher values = trust IMU less, allow more vision correction
        self.sigma_a = 0.5      # Accel noise stddev (m/s^2) - higher for Gazebo
        self.sigma_g = 0.05     # Gyro noise stddev (rad/s) - accounts for simulation noise
        self.Q_a = self.sigma_a ** 2  # Accel noise variance
        self.Q_g = self.sigma_g ** 2  # Gyro noise variance
        self.Q_ba = 1e-5  # Accel bias random walk - faster adaptation
        self.Q_bg = 1e-5  # Gyro bias random walk - faster adaptation
        self.R_cam = 5.0   # Pixel measurement noise - trust vision more (lower = more trust)

        # Outlier rejection settings
        self.mahalanobis_threshold = 8.0  # Relaxed threshold (chi-squared 99.5% for 2 DOF)
        self.consecutive_outliers = 0
        self.max_consecutive_outliers = 10  # Faster recovery

        # Gravity correction gain (for attitude correction from accelerometer)
        # Higher values = faster convergence but more noise sensitivity
        # Typical complementary filter uses 0.02-0.1 for α
        # Lower value reduces jitter from IMU noise (recommendation from analysis)
        self.gravity_correction_gain = 0.02  # Conservative to reduce jitter

        # Time sync tolerance (max acceptable delay between IMU and vision)
        self.max_time_delay = 0.1  # 100ms tolerance (relaxed for Gazebo)

        # Landmarks (Known Map from landmarks.sdf)
        # ID 1: Red, ID 2: Green, ID 3: Blue, ID 4: Yellow, ID 5: Cyan
        self.map = {
            1.0: np.array([2.0, 2.0, 0.5]),    # Red
            2.0: np.array([4.0, -2.0, 0.5]),   # Green
            3.0: np.array([-2.0, 3.0, 0.5]),   # Blue
            4.0: np.array([5.0, 3.0, 0.5]),    # Yellow
            5.0: np.array([-3.0, -3.0, 0.5])   # Cyan
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
        self.R_b_c = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])

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

    def _reinitialize_orientation(self, a_m, w_m):
        """
        Smart re-initialization when filter diverges.

        Instead of resetting to identity quaternion (which causes infinite reset loops),
        we estimate orientation from the current accelerometer reading.
        Position is NOT reset - we keep the current estimate.
        Velocity is reset to zero (safest assumption).
        BIASES ARE RESET to prevent runaway bias from causing immediate re-divergence.

        This allows the filter to recover from bad orientation estimates
        without losing all position information.
        """
        self.get_logger().warn("Re-initializing orientation from current IMU...")

        # Estimate orientation from accelerometer (gravity direction)
        accel_norm = np.linalg.norm(a_m)
        if accel_norm < 0.1:
            self.get_logger().error("Accelerometer reading too small, cannot re-initialize")
            return

        gravity_body = a_m / accel_norm
        gravity_world = np.array([0.0, 0.0, 1.0])

        # Find rotation that aligns gravity_body with gravity_world
        v = np.cross(gravity_body, gravity_world)
        s = np.linalg.norm(v)
        c = np.dot(gravity_body, gravity_world)

        if s < 1e-6:
            if c > 0:
                R_init = np.eye(3)
            else:
                R_init = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        else:
            vx = skew_symmetric(v)
            R_init = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))

        # Convert to quaternion
        rot = R.from_matrix(R_init)
        q_scipy = rot.as_quat()  # [x, y, z, w]
        self.x[6:10] = np.array([q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]])

        # Reset velocity to zero
        self.x[3:6] = 0.0

        # Ground constraint: Z position should be 0
        self.x[2] = 0.0

        # CRITICAL: Reset accelerometer bias to zero!
        # If bias has grown to a huge value (common during divergence), keeping it
        # will cause immediate re-divergence. Realistic accel bias is < 0.5 m/s².
        self.x[10:13] = 0.0

        # Reset gyro bias to zero as well (will re-estimate during operation)
        self.x[13:16] = 0.0

        # Expand covariance to reflect uncertainty after reset
        self.P[0:3, 0:3] = np.eye(3) * 1.0    # Position uncertainty
        self.P[3:6, 3:6] = np.eye(3) * 0.5    # Velocity uncertainty
        self.P[6:9, 6:9] = np.eye(3) * 0.3    # Orientation uncertainty
        self.P[9:12, 9:12] = np.eye(3) * 0.1  # Accel bias uncertainty (allow re-estimation)
        self.P[12:15, 12:15] = np.eye(3) * 0.1  # Gyro bias uncertainty

        self.get_logger().info(f"Re-initialized quat: w={self.x[6]:.3f}, x={self.x[7]:.3f}, y={self.x[8]:.3f}, z={self.x[9]:.3f}")

    def predict(self, dt, a_m, w_m):
        """
        ES-EKF / MEKF Prediction Step (IMU Odometry Model)

        This implements the prediction step of the Error-State Extended Kalman Filter
        following the multiplicative quaternion formulation from MatthewHampsey/mekf.

        The prediction consists of two parallel processes:
        1. Nominal State Propagation - Non-linear kinematics integration
        2. Error Covariance Propagation - Linearized uncertainty propagation

        Error State Vector (15D): [δp(3), δv(3), δθ(3), δba(3), δbg(3)]
        - δθ uses minimal 3D angular representation to avoid quaternion rank deficiency

        Reference: prediction_step.md, es_ekf_handbook.md
        """
        # ===================================================================
        # STEP 0: Unpack State and Compute Bias-Corrected Measurements
        # ===================================================================
        p = self.x[0:3]      # Position
        v = self.x[3:6]      # Velocity
        q = self.x[6:10]     # Quaternion [w, x, y, z]
        ba = self.x[10:13]   # Accelerometer bias
        bg = self.x[13:16]   # Gyroscope bias

        # Rotation Matrix R_wb (body to world)
        rot = R.from_quat([q[1], q[2], q[3], q[0]])  # scipy uses [x, y, z, w]
        R_wb = rot.as_matrix()

        # Bias-corrected IMU measurements
        a_corrected = a_m - ba  # Corrected acceleration in body frame
        w_corrected = w_m - bg  # Corrected angular velocity in body frame

        # ===================================================================
        # STEP 1: Nominal State Propagation (Non-linear Kinematics)
        # ===================================================================
        # Acceleration in world frame (gravity compensated)
        acc_world = R_wb @ a_corrected - self.g

        # --- GROUND ROBOT CONSTRAINTS ---
        # For a wheeled robot on flat ground:
        # 1. Z-acceleration should be ~0 (can't fly or sink)
        # 2. Z-velocity should be ~0
        # 3. Z-position should be ~0 (ground level)

        # NOTE: Removed aggressive acceleration clipping per recommendations.md
        # Hard clipping causes divergence when robot experiences shocks/maneuvers
        # Let the EKF handle uncertainty naturally; only clip truly extreme values

        # Soft limit for extreme cases only (e.g., sensor glitches)
        MAX_ACCEL = 10.0  # m/s² - allow larger transients, only clip glitches
        acc_world_clipped = np.clip(acc_world, -MAX_ACCEL, MAX_ACCEL)

        # Zero out Z-axis acceleration for ground robot (can't accelerate vertically)
        acc_world_clipped[2] = 0.0

        acc_world_norm = np.linalg.norm(acc_world)

        # --- Divergence Watchdog ---
        if acc_world_norm > 20.0:  # Increased threshold - let filter work
            self.get_logger().error(f"Filter Divergence! Accel: {acc_world_norm:.1f} m/s^2. Re-initializing from IMU...")
            self._reinitialize_orientation(a_m, w_m)
            return

        # Debug logging (every ~2 seconds at 200Hz)
        if not hasattr(self, '_debug_counter'):
            self._debug_counter = 0
        self._debug_counter += 1
        if self._debug_counter % 400 == 1:
            self.get_logger().info(f"World Accel: [{acc_world[0]:.3f}, {acc_world[1]:.3f}, {acc_world[2]:.3f}] m/s^2")

        # --- Position Update ---
        # p_k = p_{k-1} + v_{k-1}*dt + 0.5*(R*a_corrected - g)*dt^2
        p_new = p + v * dt + 0.5 * acc_world_clipped * dt**2

        # Ground constraint: Z should stay near 0
        p_new[2] = 0.0

        # --- Velocity Update ---
        # v_k = v_{k-1} + (R*a_corrected - g)*dt
        v_new = v + acc_world_clipped * dt

        # Ground constraint: Z velocity should be 0
        v_new[2] = 0.0

        # Velocity damping: prevent unbounded velocity growth
        # Max speed for TurtleBot is ~0.26 m/s (we allow up to 1.5 m/s for safety margin)
        MAX_SPEED = 1.5  # m/s
        speed = np.linalg.norm(v_new[:2])
        if speed > MAX_SPEED:
            v_new[:2] = v_new[:2] * (MAX_SPEED / speed)
            self.get_logger().warn(f"Velocity clipped from {speed:.2f} to {MAX_SPEED} m/s")

        # --- Orientation Update (Quaternion Integration) ---
        # Using quaternion exponential map: q_k = q_{k-1} ⊗ exp(0.5 * ω_corrected * dt)
        # Reference: MatthewHampsey/mekf model.py - quaternion derivative method
        w_norm = np.linalg.norm(w_corrected)
        if w_norm > 1e-8:
            # Exact quaternion exponential for rotation vector θ = ω*dt
            # Δq = [cos(|θ|/2), sin(|θ|/2) * θ/|θ|]
            half_angle = 0.5 * w_norm * dt
            axis = w_corrected / w_norm
            dq = np.array([
                axis[0] * np.sin(half_angle),
                axis[1] * np.sin(half_angle),
                axis[2] * np.sin(half_angle),
                np.cos(half_angle)
            ])  # [x, y, z, w] for scipy

            # Quaternion multiplication: q_new = q_old ⊗ dq
            q_new_obj = rot * R.from_quat(dq)
            q_new_scipy = q_new_obj.as_quat()  # [x, y, z, w]
            q_new = np.array([q_new_scipy[3], q_new_scipy[0], q_new_scipy[1], q_new_scipy[2]])
        else:
            # For very small rotations, quaternion stays the same
            q_new = q.copy()

        # --- Bias Update (Random Walk Model) ---
        # Biases are constant in the prediction step (drift added via process noise)
        ba_new = ba.copy()
        bg_new = bg.copy()

        # --- BIAS MAGNITUDE LIMITS ---
        # Prevent biases from growing to unrealistic values
        # Realistic accelerometer bias: < 0.5 m/s² (typical MEMS IMU)
        # Realistic gyroscope bias: < 0.1 rad/s (~5 deg/s)
        MAX_ACCEL_BIAS = 0.5  # m/s²
        MAX_GYRO_BIAS = 0.1   # rad/s

        ba_new = np.clip(ba_new, -MAX_ACCEL_BIAS, MAX_ACCEL_BIAS)
        bg_new = np.clip(bg_new, -MAX_GYRO_BIAS, MAX_GYRO_BIAS)

        # NOTE: VELOCITY_DECAY removed per recommendations.md
        # Non-physical damping fights against visual updates. Let EKF handle drift naturally.

        # Update Nominal State
        self.x[0:3] = p_new
        self.x[3:6] = v_new
        self.x[6:10] = q_new
        self.x[10:13] = ba_new
        self.x[13:16] = bg_new

        # ===================================================================
        # STEP 2: Error Covariance Propagation (Linearized Uncertainty)
        # ===================================================================
        # The error state Jacobian Fx describes how errors propagate.
        # Fx ≈ I + F*dt where F is the continuous-time error dynamics matrix.
        #
        # Reference: MatthewHampsey/mekf kalman2.py lines 55-69
        #
        # Error state ordering: [δp, δv, δθ, δba, δbg]
        #                       [0:3, 3:6, 6:9, 9:12, 12:15]

        # Construct continuous-time Jacobian F
        F = np.zeros((15, 15))

        # --- Position error dynamics ---
        # δṗ = δv
        F[0:3, 3:6] = np.eye(3)

        # --- Velocity error dynamics ---
        # δv̇ = -R*[a_corrected]_× * δθ - R * δba
        # d(δv)/d(δθ): Rotation of body-frame acceleration by orientation error
        F[3:6, 6:9] = -R_wb @ skew_symmetric(a_corrected)
        # d(δv)/d(δba): Effect of accel bias error on velocity
        F[3:6, 9:12] = -R_wb

        # --- Orientation error dynamics ---
        # δθ̇ = -[ω_corrected]_× * δθ - δbg
        # d(δθ)/d(δθ): Gyro measurement coupling (MEKF key insight!)
        # Reference: MatthewHampsey/mekf kalman2.py line 66: G[0:3, 0:3] = -skewSymmetric(gyro_meas)
        F[6:9, 6:9] = -skew_symmetric(w_corrected)
        # d(δθ)/d(δbg): Effect of gyro bias error on orientation
        F[6:9, 12:15] = -np.eye(3)

        # --- Bias error dynamics ---
        # δḃa = 0, δḃg = 0 (Random walk - noise added separately)
        # F[9:12, 9:12] = 0 (already zero)
        # F[12:15, 12:15] = 0 (already zero)

        # Discrete-time state transition: Fx = I + F*dt
        Fx = np.eye(15) + F * dt

        # ===================================================================
        # STEP 3: Process Noise Covariance Q
        # ===================================================================
        # The process noise captures uncertainty from IMU sensor noise and bias drift.
        #
        # Reference: MatthewHampsey/mekf kalman2.py process_covariance()
        # Uses Van Loan method for proper discrete-time noise covariance.
        #
        # Simplified version with primary diagonal and key cross-terms:

        # Noise covariance matrices (3x3)
        Q_gyro = self.Q_g * np.eye(3)        # Gyro noise variance
        Q_gyro_bias = self.Q_bg * np.eye(3)  # Gyro bias random walk
        Q_accel = self.Q_a * np.eye(3)       # Accel noise variance
        Q_accel_bias = self.Q_ba * np.eye(3) # Accel bias random walk

        # Build discrete process noise matrix Q
        # Following the structure from MatthewHampsey/mekf kalman2.py
        Qi = np.zeros((15, 15))

        # Position noise (from velocity integration of acceleration noise)
        # Q_p = Q_a * dt^4/4 + Q_ba * dt^6/36 (approximated)
        Qi[0:3, 0:3] = Q_accel * (dt**4 / 4.0)

        # Velocity noise (from acceleration noise)
        # Q_v = Q_a * dt^2 + higher order terms
        Qi[3:6, 3:6] = Q_accel * dt + Q_accel_bias * (dt**3 / 3.0)

        # Position-Velocity cross-correlation
        Qi[0:3, 3:6] = Q_accel * (dt**3 / 2.0)
        Qi[3:6, 0:3] = Qi[0:3, 3:6].T

        # Orientation noise (from gyro noise)
        # Q_θ = Q_g * dt + Q_bg * dt^3/3
        Qi[6:9, 6:9] = Q_gyro * dt + Q_gyro_bias * (dt**3 / 3.0)

        # Orientation-Gyro bias cross-correlation
        # Reference: MatthewHampsey/mekf - Q[0:3, 9:12] = -gyro_bias_cov*(dt^2)/2
        Qi[6:9, 12:15] = -Q_gyro_bias * (dt**2 / 2.0)
        Qi[12:15, 6:9] = Qi[6:9, 12:15].T

        # Accel bias random walk
        Qi[9:12, 9:12] = Q_accel_bias * dt

        # Velocity-Accel bias cross-correlation
        Qi[3:6, 9:12] = -Q_accel_bias * (dt**2 / 2.0)
        Qi[9:12, 3:6] = Qi[3:6, 9:12].T

        # Gyro bias random walk
        Qi[12:15, 12:15] = Q_gyro_bias * dt

        # ===================================================================
        # STEP 4: Covariance Update
        # ===================================================================
        # P_{k|k-1} = Fx * P_{k-1|k-1} * Fx^T + Q
        self.P = Fx @ self.P @ Fx.T + Qi

        # Ensure symmetry (numerical stability)
        self.P = 0.5 * (self.P + self.P.T)

        # --- COVARIANCE BOUNDS ---
        # Prevent covariance from growing unbounded, which causes erratic Kalman gains
        # These limits represent our worst-case uncertainty
        MAX_POS_VAR = 10.0   # 10m² = ~3m std (if off by more, filter should reset)
        MAX_VEL_VAR = 1.0    # 1m²/s² = 1m/s std (robot max speed is 0.26 m/s)
        MAX_ROT_VAR = 0.5    # 0.5 rad² = ~40° std
        MAX_BIAS_VAR = 0.1   # Bias variance

        # Clip diagonal elements
        for i in range(3):
            self.P[i, i] = min(self.P[i, i], MAX_POS_VAR)          # Position
            self.P[3+i, 3+i] = min(self.P[3+i, 3+i], MAX_VEL_VAR)  # Velocity
            self.P[6+i, 6+i] = min(self.P[6+i, 6+i], MAX_ROT_VAR)  # Orientation
            self.P[9+i, 9+i] = min(self.P[9+i, 9+i], MAX_BIAS_VAR) # Accel bias
            self.P[12+i, 12+i] = min(self.P[12+i, 12+i], MAX_BIAS_VAR) # Gyro bias

        # Limit off-diagonal correlations to prevent wild cross-corrections
        # Correlation coefficient should stay in [-1, 1]
        for i in range(15):
            for j in range(i+1, 15):
                max_corr = np.sqrt(self.P[i,i] * self.P[j,j])
                if abs(self.P[i,j]) > max_corr:
                    sign = np.sign(self.P[i,j])
                    self.P[i,j] = sign * max_corr * 0.99  # Keep correlation < 1
                    self.P[j,i] = self.P[i,j]

        # ===================================================================
        # STEP 5: ACCELEROMETER-BASED ATTITUDE CORRECTION
        # ===================================================================
        # This is the KEY insight from AHRS EKF implementations (soarbear/imu_ekf, Mayitzin/ahrs):
        # The accelerometer provides an absolute reference for roll/pitch via gravity.
        # Without this, small gyro biases cause orientation to drift, which causes
        # gravity to "leak" into XY acceleration, causing position to fly away.
        #
        # This is essentially a complementary filter fused with the EKF:
        # - Gyro provides high-frequency orientation changes
        # - Accelerometer provides low-frequency absolute reference
        #
        # Reference: AHRS EKF correction step uses h(q) = R(q)^T @ g_world

        # Only apply correction when accelerometer magnitude is close to gravity
        # (i.e., when the robot is not accelerating significantly)
        accel_magnitude = np.linalg.norm(a_corrected)
        accel_deviation = abs(accel_magnitude - self.gravity_magnitude)

        # Threshold: only correct when within 1.0 m/s^2 of expected gravity
        if accel_deviation < 1.0:
            # Normalize accelerometer reading to get measured gravity direction in body frame
            gravity_body_measured = a_corrected / accel_magnitude

            # Expected gravity direction in body frame (using current orientation estimate)
            # If orientation is perfect: R_wb^T @ [0,0,g] / g = [0, 0, 1] in body frame
            # We compare measured vs expected gravity direction
            R_bw = R_wb.T  # World to body rotation
            gravity_body_expected = R_bw @ np.array([0.0, 0.0, 1.0])

            # Attitude error: cross product gives rotation axis, magnitude gives angle (small angle approx)
            # error = measured × expected gives the rotation to correct
            attitude_error = np.cross(gravity_body_measured, gravity_body_expected)

            # This error is in body frame. For roll/pitch correction:
            # attitude_error[0] ≈ pitch error (rotation around body X)
            # attitude_error[1] ≈ roll error (rotation around body Y)
            # attitude_error[2] ≈ yaw component (ignore - can't determine yaw from gravity)

            # Apply correction with adaptive gain based on acceleration consistency
            # Lower gain when accelerating (less trust in gravity direction)
            adaptive_gain = self.gravity_correction_gain * (1.0 - accel_deviation)

            # Zero out yaw correction (can't determine yaw from gravity alone)
            attitude_error[2] = 0.0

            correction = adaptive_gain * attitude_error

            # Apply correction as small quaternion rotation (in body frame)
            if np.linalg.norm(correction) > 1e-8:
                # Create rotation from correction vector
                dq_correction = R.from_rotvec(correction)

                # Current orientation
                q_curr = R.from_quat([self.x[7], self.x[8], self.x[9], self.x[6]])  # scipy [x,y,z,w]

                # Apply correction: q_corrected = q_curr * dq (body frame correction)
                q_corrected = q_curr * dq_correction
                q_new_corr = q_corrected.as_quat()  # [x, y, z, w]

                # Update state with corrected quaternion
                self.x[6:10] = np.array([q_new_corr[3], q_new_corr[0], q_new_corr[1], q_new_corr[2]])

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
        self.get_logger().info(f"Vision update! Landmark at {lm_pos_world}")
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
        # d(pc)/d(theta) = R_bc * [p_b]_× (skew symmetric of point in body)
        J_rot = self.R_b_c @ skew_symmetric(p_b)

        # Assemble H (2x15)
        H = np.zeros((2, 15))
        H[:, 0:3] = J_proj @ J_pos
        H[:, 6:9] = J_proj @ J_rot

        # 3. Kalman Update
        # S = H P H.T + R
        S = H @ self.P @ H.T + np.eye(2) * self.R_cam

        # --- Outlier Gating ---
        # For KNOWN landmarks, we should be more permissive
        # Large residuals after long dead-reckoning are EXPECTED and should be corrected
        try:
            S_inv = np.linalg.inv(S)
            mahalanobis_sq = z_res @ S_inv @ z_res
        except np.linalg.LinAlgError:
            self.get_logger().warn("Singular S matrix, skipping update")
            return

        pixel_residual = np.linalg.norm(z_res)
        mahal_dist = np.sqrt(mahalanobis_sq)

        # Adaptive gating: if covariance is large, allow larger corrections
        # Chi-squared 99% for 2 DOF is 9.21, 99.9% is 13.82
        # Since we KNOW landmark positions, we can be more aggressive
        adaptive_threshold = max(self.mahalanobis_threshold, 10.0)

        if mahal_dist > adaptive_threshold:
            self.consecutive_outliers += 1

            # NOTE: Removed manual covariance expansion per recommendations.md
            # Manual P expansion causes violent state jumps (high Kalman gain)
            # Instead, let the filter naturally increase uncertainty through Q
            # and use a higher threshold to accept more measurements

            if self.consecutive_outliers >= self.max_consecutive_outliers:
                # After many outliers, lower threshold to accept next measurement
                # This allows gradual correction rather than violent jumps
                self.get_logger().warn(f"FILTER RECOVERY: {self.consecutive_outliers} outliers. Lowering threshold temporarily.")
                # Don't modify P - just accept the next measurement with a warning
                self.consecutive_outliers = 0
                # Fall through to accept this measurement (don't return)
            else:
                self.get_logger().warn(f"Outlier Rejected! Residual: {pixel_residual:.1f} px, Mahal: {mahal_dist:.1f} (thresh={adaptive_threshold:.1f})")
                return

        # Good measurement - reset outlier counter
        self.consecutive_outliers = 0

        # Log successful updates for debugging
        if pixel_residual > 20.0:
            self.get_logger().info(f"Vision correction: {pixel_residual:.1f} px, Mahal: {mahal_dist:.1f}")

        try:
            K = self.P @ H.T @ S_inv
        except np.linalg.LinAlgError:
            return

        # Error State Update
        dx = K @ z_res

        # Update Covariance (Joseph form for numerical stability)
        I = np.eye(15)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ (np.eye(2) * self.R_cam) @ K.T

        # 4. Inject Error into Nominal State
        # NOTE: Removed hard-clipping per recommendations.md
        # Hard clipping prevents proper convergence and causes persistent innovations.
        # The Kalman filter is designed to handle large residuals through its math.
        # We only apply soft limits for safety (log but don't clip unless extreme)

        # Position correction (no hard clip - let EKF work)
        pos_correction = dx[0:3]
        if np.linalg.norm(pos_correction) > 1.0:
            self.get_logger().info(f"Large position correction: {np.linalg.norm(pos_correction):.2f}m", throttle_duration_sec=1.0)
        self.x[0:3] += pos_correction

        # Velocity correction (soft limit only for extreme cases)
        vel_correction = dx[3:6]
        vel_corr_norm = np.linalg.norm(vel_correction)
        if vel_corr_norm > 2.0:  # Only clip truly extreme corrections
            self.get_logger().warn(f"Extreme velocity correction clipped: {vel_corr_norm:.2f} m/s", throttle_duration_sec=1.0)
            vel_correction = vel_correction * (2.0 / vel_corr_norm)
        self.x[3:6] += vel_correction

        # Orientation (Quaternion multiply) - soft limit for extreme cases only
        rot_correction = dx[6:9]
        rot_corr_norm = np.linalg.norm(rot_correction)
        if rot_corr_norm > 0.5:  # ~30 degrees - truly extreme
            self.get_logger().warn(f"Extreme rotation correction clipped: {np.degrees(rot_corr_norm):.1f} deg", throttle_duration_sec=1.0)
            rot_correction = rot_correction * (0.5 / rot_corr_norm)
        dq_rot = R.from_rotvec(rot_correction)
        q_curr_obj = R.from_quat([self.x[7], self.x[8], self.x[9], self.x[6]]) # [x,y,z,w]
        q_new_obj = q_curr_obj * dq_rot
        q_new = q_new_obj.as_quat()
        self.x[6:10] = np.array([q_new[3], q_new[0], q_new[1], q_new[2]])

        # Biases - apply corrections (EKF handles the rate naturally through P)
        self.x[10:13] += dx[9:12]
        self.x[13:16] += dx[12:15]

        # Enforce bias magnitude limits (physical bounds, not rate limits)
        MAX_ACCEL_BIAS = 0.5  # m/s² - realistic MEMS limit
        MAX_GYRO_BIAS = 0.1   # rad/s - realistic MEMS limit
        self.x[10:13] = np.clip(self.x[10:13], -MAX_ACCEL_BIAS, MAX_ACCEL_BIAS)
        self.x[13:16] = np.clip(self.x[13:16], -MAX_GYRO_BIAS, MAX_GYRO_BIAS)

        # --- GROUND ROBOT CONSTRAINTS (post-update) ---
        self.x[2] = 0.0  # Z position = 0
        self.x[5] = 0.0  # Z velocity = 0

        # Final velocity sanity check (physical limit, not EKF limit)
        MAX_SPEED = 2.0  # m/s - slightly above TurtleBot max for safety margin
        speed = np.linalg.norm(self.x[3:5])
        if speed > MAX_SPEED:
            self.x[3:5] = self.x[3:5] * (MAX_SPEED / speed)
            self.get_logger().warn(f"Velocity exceeded {MAX_SPEED} m/s, clipped", throttle_duration_sec=2.0)

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