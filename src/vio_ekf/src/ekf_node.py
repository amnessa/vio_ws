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
        self.init_sample_count = 2400  # Reduced from 2400 (1 second at 200Hz)
        self.gravity_magnitude = 9.81  # Expected gravity magnitude (will be updated from IMU)
        self.g = np.array([0.0, 0.0, 9.81])  # Gravity vector in world frame (will be updated)

        # --- State Definitions ---
        # ROBOCENTRIC FORMULATION: Velocity is expressed in BODY frame, not world frame.
        # This decouples observable states (velocity, tilt) from unobservable global yaw.
        # Reference: "Observability-based Rules for Designing Consistent EKF SLAM Estimators"
        #
        # State: [p_w(3), v_b(3), q(4), ba(3), bg(3)] = 16 elements
        #   - p_w: Position in WORLD frame (for visualization/localization)
        #   - v_b: Velocity in BODY frame (robocentric - decoupled from yaw uncertainty)
        #   - q: Quaternion (world to body orientation)
        #   - ba, bg: IMU biases in body frame
        #
        # Error State: [dp(3), dv(3), dtheta(3), dba(3), dbg(3)] = 15 elements
        self.x = np.zeros(16)
        self.x[6] = 1.0  # Initial quaternion (w=1, x=0, y=0, z=0)

        # Covariance Matrix (15x15)
        # Start with higher uncertainty - let vision corrections do their job
        self.P = np.eye(15) * 0.1
        self.P[0:3, 0:3] = np.eye(3) * 0.5   # Position uncertainty (0.5m std)
        self.P[3:6, 3:6] = np.eye(3) * 0.1   # Velocity uncertainty (0.3m/s std)
        self.P[6:9, 6:9] = np.eye(3) * 0.1   # Orientation uncertainty (~18 deg std)
        self.P[9:12, 9:12] = np.eye(3) * 0.1  # Accel bias uncertainty - higher to allow estimation
        self.P[12:15, 12:15] = np.eye(3) * 0.01  # Gyro bias uncertainty

        # Noise Parameters - Tuned for Gazebo simulation
        # Higher values = trust IMU less, allow more vision correction
        self.sigma_a = 1.0      # Accel noise stddev (m/s^2) - higher for Gazebo
        self.sigma_g = 0.5     # Gyro noise stddev (rad/s) - accounts for simulation noise
        self.Q_a = self.sigma_a ** 2  # Accel noise variance
        self.Q_g = self.sigma_g ** 2  # Gyro noise variance
        # Per recommendation2.md: Tune bias random walk carefully.
        # Too high Q_ba causes filter to use bias to compensate for orientation errors.
        # Reduced from 5e-4 to allow slower, more stable bias estimation.
        self.Q_ba = 1e-4  # Accel bias random walk - slower adaptation
        self.Q_bg = 1e-5  # Gyro bias random walk - slower adaptation
        self.R_cam = 70.0   # Pixel measurement noise - INCREASED to trust vision less (prevents velocity explosion)

        # ZUPT (Zero-Velocity Update) parameters
        # Per recommend.md Section 5: Trigger ZUPT based solely on gyro activity
        # Ignore accelerometer deviation as it may be caused by orientation error
        self.zupt_gyro_threshold = 0.02  # rad/s - per recommend.md (<0.02 rad/s)
        self.zupt_window = []  # Rolling window of IMU samples
        self.zupt_window_size = 50  # ~250ms at 200Hz
        self.zupt_gyro_only_counter = 0  # Counter for gyro-only stationary detection
        self.zupt_gyro_only_threshold = 100  # 0.5 second at 200Hz - if gyro stable this long, definitely stationary

        # Formal ZUPT measurement noise (small = high confidence velocity is zero)
        self.R_zupt_velocity = 0.001  # (m/s)^2 - very confident velocity is zero when stationary
        self.R_zupt_gravity = 0.1  # Gravity direction measurement noise for tilt correction

        # Vision-based motion detection to prevent false ZUPT
        # If vision shows robot is moving, disable ZUPT even if gyro is quiet
        self.last_vision_correction = 0.0  # Position correction from last vision update
        self.vision_motion_threshold = 0.02  # 2cm correction = robot is moving
        self.vision_motion_cooldown = 0  # Frames since significant vision correction

        # Outlier rejection settings
        self.mahalanobis_threshold = 60.0  # Very relaxed - accept most measurements
        self.consecutive_outliers = 0
        self.max_consecutive_outliers = 5  # Quick recovery after just 3 rejections

        # Gravity correction gain (for attitude correction from accelerometer)
        # Higher values = faster convergence but more noise sensitivity
        # Typical complementary filter uses 0.02-0.1 for α
        # Per recommendation.md: Reduce further or disable to prevent jitter
        # The EKF should handle orientation through proper measurement updates
        self.gravity_correction_gain = 0.005  # Very conservative - minimal jitter

        # Time sync tolerance (max acceptable delay between IMU and vision)
        self.max_time_delay = 0.15  # 150ms tolerance (relaxed for Gazebo)

        # State buffer for time delay compensation (per recommend.md)
        # Store recent states to match vision measurements with correct pose
        self.state_buffer = []  # List of {'time': t, 'x': state, 'P': covariance}
        self.state_buffer_size = 50  # ~250ms at 200Hz

        # Landmarks (Known Map - ArUco markers in circular arrangement)
        # Generated by scripts/generate_aruco_simulation.py
        # 3 rings at 1.4m, 2.8m, 4.2m radius (30% smaller), 8 markers each, facing center
        # Camera should always see 3+ markers regardless of heading
        self.map = {
            0.0: np.array([2.50, 0.00, 0.30]),
            1.0: np.array([2.41, 0.65, 0.30]),
            2.0: np.array([2.17, 1.25, 0.30]),
            3.0: np.array([1.77, 1.77, 0.30]),
            4.0: np.array([1.25, 2.17, 0.30]),
            5.0: np.array([0.65, 2.41, 0.30]),
            6.0: np.array([0.00, 2.50, 0.30]),
            7.0: np.array([-0.65, 2.41, 0.30]),
            8.0: np.array([-1.25, 2.17, 0.30]),
            9.0: np.array([-1.77, 1.77, 0.30]),
            10.0: np.array([-2.17, 1.25, 0.30]),
            11.0: np.array([-2.41, 0.65, 0.30]),
            12.0: np.array([-2.50, 0.00, 0.30]),
            13.0: np.array([-2.41, -0.65, 0.30]),
            14.0: np.array([-2.17, -1.25, 0.30]),
            15.0: np.array([-1.77, -1.77, 0.30]),
            16.0: np.array([-1.25, -2.17, 0.30]),
            17.0: np.array([-0.65, -2.41, 0.30]),
            18.0: np.array([-0.00, -2.50, 0.30]),
            19.0: np.array([0.65, -2.41, 0.30]),
            20.0: np.array([1.25, -2.17, 0.30]),
            21.0: np.array([1.77, -1.77, 0.30]),
            22.0: np.array([2.17, -1.25, 0.30]),
            23.0: np.array([2.41, -0.65, 0.30]),
        }

        # First-Estimate Jacobians (FEJ) storage
        # Stores the robot pose (position, orientation) when each landmark was first observed.
        # Using these "first estimates" for Jacobian computation prevents spurious observability
        # of global yaw, which causes "gravity leakage" where yaw errors contaminate velocity.
        # Reference: Huang, Mourikis, Roumeliotis - "Observability-based Rules for Designing
        # Consistent EKF SLAM Estimators" (IJRR 2010)
        self.landmark_first_estimates = {}  # lm_id -> {'p': position, 'R': rotation_matrix}

        # Calibration (Extrinsics: Base -> Camera Optical Frame)
        # Camera moved forward to 0.10m to avoid robot body blocking view
        # Total translation: [0.10, 0.0, 0.093]
        self.t_b_c = np.array([0.10, 0.0, 0.093])

        # Rotation from body frame to camera optical frame
        # Body frame (ROS): X=forward, Y=left, Z=up
        # Optical frame (OpenCV): X=right, Y=down, Z=forward (depth)
        #
        # Verified transformation:
        #   Body X [1,0,0] -> Camera [0,0,1] (depth)
        #   Body Y [0,1,0] -> Camera [-1,0,0] (negative right = left)
        #   Body Z [0,0,1] -> Camera [0,-1,0] (negative down = up)
        #
        # This gives R_b_c such that p_camera = R_b_c @ p_body
        self.R_b_c = np.array([
            [0, -1, 0],   # Camera X = -Body Y
            [0, 0, -1],   # Camera Y = -Body Z
            [1, 0, 0]     # Camera Z = Body X (depth = forward)
        ], dtype=np.float64)

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
        # Per recommendation.md: Allow the filter to estimate accelerometer bias
        # rather than forcing it to zero. This lets the EKF learn constant sensor
        # offsets that cause velocity drift.
        #
        # After aligning orientation with gravity, the RESIDUAL acceleration
        # (in world frame) should be zero. Any non-zero residual is likely bias.
        R_init_mat = rot.as_matrix()
        accel_world_residual = R_init_mat @ accel_mean - self.g

        # The residual in world frame is what the filter sees as "phantom acceleration"
        # Transform back to body frame to get initial bias estimate
        initial_accel_bias = R_init_mat.T @ accel_world_residual

        # Only set small initial bias - let filter refine it
        # Limit to realistic MEMS bias range (< 0.3 m/s²)
        initial_accel_bias = np.clip(initial_accel_bias, -0.3, 0.3)
        self.x[10:13] = initial_accel_bias

        self.get_logger().info(f"Initial accel world residual: [{accel_world_residual[0]:.4f}, {accel_world_residual[1]:.4f}, {accel_world_residual[2]:.4f}] m/s^2")

        # Log the computed initial tilt for debugging
        initial_roll = np.arctan2(accel_mean[1], accel_mean[2])
        initial_pitch = np.arctan2(-accel_mean[0], np.sqrt(accel_mean[1]**2 + accel_mean[2]**2))
        self.get_logger().info(f"Initial tilt: Roll={np.degrees(initial_roll):.2f} deg, Pitch={np.degrees(initial_pitch):.2f} deg")
        self.get_logger().info(f"Initial orientation (quat): w={self.x[6]:.4f}, x={self.x[7]:.4f}, y={self.x[8]:.4f}, z={self.x[9]:.4f}")
        self.get_logger().info(f"Accel bias: [{self.x[10]:.4f}, {self.x[11]:.4f}, {self.x[12]:.4f}] m/s^2 (estimated from residual)")
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

        # CRITICAL: Preserve current yaw! Gravity only gives roll/pitch, not yaw.
        # Extract current yaw before re-initialization
        q_current = self.x[6:10]  # [w, x, y, z]
        rot_current = R.from_quat([q_current[1], q_current[2], q_current[3], q_current[0]])  # scipy format
        current_euler = rot_current.as_euler('xyz')  # [roll, pitch, yaw]
        current_yaw = current_euler[2]
        self.get_logger().info(f"Preserving yaw: {np.degrees(current_yaw):.1f} deg")

        # Estimate roll/pitch from accelerometer (gravity direction)
        accel_norm = np.linalg.norm(a_m)
        if accel_norm < 0.1:
            self.get_logger().error("Accelerometer reading too small, cannot re-initialize")
            return

        gravity_body = a_m / accel_norm

        # Compute roll and pitch from gravity direction
        # gravity_body should be [0, 0, 1] when level
        # roll = atan2(gy, gz), pitch = atan2(-gx, sqrt(gy^2 + gz^2))
        new_roll = np.arctan2(gravity_body[1], gravity_body[2])
        new_pitch = np.arctan2(-gravity_body[0], np.sqrt(gravity_body[1]**2 + gravity_body[2]**2))

        # Clamp roll/pitch for ground robot
        MAX_TILT = np.radians(10.0)
        new_roll = np.clip(new_roll, -MAX_TILT, MAX_TILT)
        new_pitch = np.clip(new_pitch, -MAX_TILT, MAX_TILT)

        self.get_logger().info(f"New roll={np.degrees(new_roll):.1f}°, pitch={np.degrees(new_pitch):.1f}°, yaw={np.degrees(current_yaw):.1f}° (preserved)")

        # Build quaternion from euler angles, preserving yaw
        rot_new = R.from_euler('xyz', [new_roll, new_pitch, current_yaw])
        q_scipy = rot_new.as_quat()  # [x, y, z, w]
        self.x[6:10] = np.array([q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]])

        # Reset velocity to zero
        self.x[3:6] = 0.0

        # Ground constraint: Z position should be 0
        self.x[2] = 0.0

        # CRITICAL FIX: Reset biases when they're saturated!
        # If biases hit their limits, the filter was already diverging and the
        # bias estimates are garbage. Using saturated biases corrupts the
        # gravity-based orientation estimate, causing immediate re-divergence.
        MAX_ACCEL_BIAS = 0.5
        MAX_GYRO_BIAS = 0.1

        # Check if biases are near saturation (within 10% of limit)
        accel_bias_saturated = np.any(np.abs(self.x[10:13]) > MAX_ACCEL_BIAS * 0.9)
        gyro_bias_saturated = np.any(np.abs(self.x[13:16]) > MAX_GYRO_BIAS * 0.9)

        if accel_bias_saturated or gyro_bias_saturated:
            self.get_logger().warn("Biases saturated! Resetting to zero for clean recovery.")
            self.x[10:13] = 0.0  # Reset accel bias
            self.x[13:16] = 0.0  # Reset gyro bias
        else:
            # Keep reasonable biases but clip just in case
            self.x[10:13] = np.clip(self.x[10:13], -MAX_ACCEL_BIAS, MAX_ACCEL_BIAS)
            self.x[13:16] = np.clip(self.x[13:16], -MAX_GYRO_BIAS, MAX_GYRO_BIAS)
            self.get_logger().info(f"Keeping biases: accel=[{self.x[10]:.3f}, {self.x[11]:.3f}, {self.x[12]:.3f}], "
                                   f"gyro=[{self.x[13]:.4f}, {self.x[14]:.4f}, {self.x[15]:.4f}]")

        # Expand covariance to reflect uncertainty after reset
        self.P[0:3, 0:3] = np.eye(3) * 1.0    # Position uncertainty
        self.P[3:6, 3:6] = np.eye(3) * 0.5    # Velocity uncertainty
        self.P[6:9, 6:9] = np.eye(3) * 0.3    # Orientation uncertainty
        self.P[9:12, 9:12] = np.eye(3) * 0.2  # Accel bias - higher uncertainty to re-learn
        self.P[12:15, 12:15] = np.eye(3) * 0.1  # Gyro bias uncertainty

        # Reset First-Estimate Jacobian storage
        # After reinitialization, the old first-estimates are stale and would cause
        # inconsistent linearization. Clear them so landmarks get fresh first-estimates.
        self.landmark_first_estimates.clear()
        self.get_logger().info("FEJ: Cleared first-estimate storage after reinitialization")

        self.get_logger().info(f"Re-initialized quat: w={self.x[6]:.3f}, x={self.x[7]:.3f}, y={self.x[8]:.3f}, z={self.x[9]:.3f}")

    def _zupt_velocity_update(self):
        """
        Formal EKF velocity update during stationary periods.

        Per recommend.md Section 5: Instead of directly overwriting velocity to zero,
        perform a formal EKF measurement update where the measurement is z=0 (zero velocity)
        with small measurement noise. This collapses velocity covariance and stops
        quadratic position error buildup.

        ROBOCENTRIC: Velocity is in body frame, so ZUPT directly measures v_b = 0.
        This is simpler and more natural than world-frame velocity.

        Measurement model: z = v_b = H @ x where H selects velocity states
        Measurement: z_meas = [0, 0, 0]
        """
        # Current body-frame velocity estimate
        v_b = self.x[3:6]

        # Skip if velocity already near zero (avoid unnecessary updates)
        if np.linalg.norm(v_b) < 0.01:
            return

        # Measurement: body velocity should be zero when stationary
        z_meas = np.array([0.0, 0.0, 0.0])
        z_pred = v_b

        # Residual
        z_res = z_meas - z_pred  # Should be -v_b

        # Measurement Jacobian H (3x15): selects velocity states
        H = np.zeros((3, 15))
        H[0:3, 3:6] = np.eye(3)  # Body velocity is states 3:6 in error state

        # Measurement noise
        R_zupt = np.eye(3) * self.R_zupt_velocity

        # Innovation covariance
        S = H @ self.P @ H.T + R_zupt

        # Kalman gain
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            self.get_logger().warn("ZUPT velocity update: Singular S matrix")
            return

        # Error state correction
        dx = K @ z_res

        # Update covariance (Joseph form for numerical stability)
        I = np.eye(15)
        IKH = I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R_zupt @ K.T
        self.P = 0.5 * (self.P + self.P.T)  # Ensure symmetry

        # Apply correction to nominal state
        # Only velocity is affected significantly
        self.x[3:6] += dx[3:6]

        # Ground constraint: body Z velocity should be 0
        self.x[5] = 0.0

        # Log only when meaningful correction applied
        if np.linalg.norm(dx[3:6]) > 0.01:
            self.get_logger().info(
                f"ZUPT Velocity: Corrected by [{dx[3]:.3f}, {dx[4]:.3f}] m/s",
                throttle_duration_sec=1.0
            )

    def _zupt_gravity_update(self, a_corrected):
        """
        Formal EKF gravity measurement update for tilt correction.

        Per recommend.md Section 1: Treat accelerometer as a TILT SENSOR during
        stationary periods. When stationary, accelerometer should measure exactly
        [0, 0, g] in world frame. The residual between measured and predicted
        gravity direction provides orientation correction.

        This updates BOTH orientation AND covariance consistently, "locking" the
        tilt and stopping gravity leakage before it integrates into velocity.

        Measurement model: The accelerometer reading should equal R^T @ g when stationary
        where R is body-to-world rotation and g is gravity in world frame.
        """
        # Current orientation
        q = self.x[6:10]
        rot = R.from_quat([q[1], q[2], q[3], q[0]])
        R_wb = rot.as_matrix()

        # Expected accelerometer reading if orientation is correct:
        # a_expected = R_wb^T @ g (gravity in body frame)
        g_world = self.g  # [0, 0, g_magnitude]
        a_expected = R_wb.T @ g_world

        # Actual reading (bias-corrected)
        a_measured = a_corrected

        # Residual: difference between measured and expected
        z_res = a_measured - a_expected

        # Skip if residual is very small (already aligned)
        if np.linalg.norm(z_res) < 0.05:
            return

        # Measurement Jacobian H (3x15)
        # The measurement is a_meas = R^T @ g
        # Derivative w.r.t. orientation error dθ:
        #   d(R^T @ g)/d(dθ) = -[R^T @ g]_× = -skew(a_expected)
        # This is because small rotation: R_new^T ≈ (I - [dθ]_×) @ R^T
        # So: a_new ≈ (I - [dθ]_×) @ a_expected = a_expected - [dθ]_× @ a_expected
        #           = a_expected + [a_expected]_× @ dθ
        # Therefore: da/d(dθ) = [a_expected]_× = skew(a_expected)
        #
        # Also derivative w.r.t. accel bias: d(a_meas)/d(ba) = -I
        # (since a_corrected = a_raw - ba)

        H = np.zeros((3, 15))
        H[0:3, 6:9] = skew_symmetric(a_expected)  # d/d(dθ)
        H[0:3, 9:12] = -np.eye(3)  # d/d(dba) - accelerometer bias affects measurement

        # Measurement noise
        R_gravity = np.eye(3) * self.R_zupt_gravity

        # Innovation covariance
        S = H @ self.P @ H.T + R_gravity

        # Check for reasonable innovation
        try:
            S_inv = np.linalg.inv(S)
            mahal_sq = z_res @ S_inv @ z_res
        except np.linalg.LinAlgError:
            self.get_logger().warn("ZUPT gravity update: Singular S matrix")
            return

        # Outlier rejection (very high residual indicates something wrong)
        # BUT: Don't reject too aggressively - orientation NEEDS correction!
        # If we reject, the filter can never recover from bad orientation.
        # Instead, apply the correction but limit its magnitude.
        MAHAL_HARD_REJECT = 1000.0  # Only reject truly crazy values
        MAHAL_LIMIT_CORRECTION = 100.0  # Limit correction above this

        if mahal_sq > MAHAL_HARD_REJECT:
            self.get_logger().warn(
                f"ZUPT gravity: Extreme residual rejected (Mahal={np.sqrt(mahal_sq):.1f})",
                throttle_duration_sec=2.0
            )
            return

        # For moderate outliers, proceed but limit correction magnitude
        correction_scale = 1.0
        if mahal_sq > MAHAL_LIMIT_CORRECTION:
            correction_scale = MAHAL_LIMIT_CORRECTION / mahal_sq
            self.get_logger().warn(
                f"ZUPT gravity: Large residual, scaling correction by {correction_scale:.2f}",
                throttle_duration_sec=2.0
            )

        # Kalman gain
        K = self.P @ H.T @ S_inv

        # Error state correction (apply scaling for outliers)
        dx = correction_scale * (K @ z_res)

        # Update covariance (Joseph form) - always update P even for scaled corrections
        I = np.eye(15)
        IKH = I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R_gravity @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        # Apply orientation correction via quaternion multiplication
        # Error state dθ (3D rotation vector) -> quaternion
        rot_correction = dx[6:9]
        rot_corr_norm = np.linalg.norm(rot_correction)

        if rot_corr_norm > 1e-8:
            # Limit large corrections - but allow bigger than before to recover faster
            MAX_ROT_CORR = 0.2  # ~12 degrees max per update (was 0.1)
            if rot_corr_norm > MAX_ROT_CORR:
                rot_correction = rot_correction * (MAX_ROT_CORR / rot_corr_norm)

            dq_rot = R.from_rotvec(rot_correction)
            q_new_obj = rot * dq_rot  # Apply rotation correction
            q_new = q_new_obj.as_quat()  # [x, y, z, w]
            self.x[6:10] = np.array([q_new[3], q_new[0], q_new[1], q_new[2]])

        # Apply bias correction
        self.x[10:13] += dx[9:12]

        # Clip biases to reasonable range
        MAX_ACCEL_BIAS = 0.5
        self.x[10:13] = np.clip(self.x[10:13], -MAX_ACCEL_BIAS, MAX_ACCEL_BIAS)

        # Log significant corrections
        if rot_corr_norm > 0.01:
            self.get_logger().info(
                f"ZUPT Gravity: Tilt corrected by {np.degrees(rot_corr_norm):.2f} deg",
                throttle_duration_sec=1.0
            )

    def predict(self, dt, a_m, w_m):
        """
        ES-EKF / MEKF Prediction Step (IMU Odometry Model) - ROBOCENTRIC FORMULATION

        CRITICAL CHANGE: Velocity is now in BODY FRAME (v_b), not world frame.
        This decouples observable local motion from unobservable global yaw.

        Body-frame velocity dynamics:
          v_b_dot = a_b - ba + R_wb^T @ g - ω × v_b

        Position update:
          p_w_dot = R_wb @ v_b

        Error State Vector (15D): [δp(3), δv_b(3), δθ(3), δba(3), δbg(3)]
        """
        # ===================================================================
        # STEP 0: Unpack State and Compute Bias-Corrected Measurements
        # ===================================================================
        p = self.x[0:3]      # Position (world frame)
        v_b = self.x[3:6]    # Velocity (BODY frame - robocentric!)
        q = self.x[6:10]     # Quaternion [w, x, y, z]
        ba = self.x[10:13]   # Accelerometer bias
        bg = self.x[13:16]   # Gyroscope bias

        # Rotation Matrix R_wb (body to world)
        rot = R.from_quat([q[1], q[2], q[3], q[0]])  # scipy uses [x, y, z, w]
        R_wb = rot.as_matrix()

        # Bias-corrected IMU measurements (both in body frame)
        a_corrected = a_m - ba  # Corrected acceleration in body frame
        w_corrected = w_m - bg  # Corrected angular velocity in body frame

        # ===================================================================
        # ZUPT: Zero-Velocity Update (Stationary Detection)
        # ===================================================================
        # Per recommend.md Section 5: Trigger ZUPT based SOLELY on gyro activity.
        # Ignore accelerometer deviation as orientation errors cause accel to exceed thresholds.
        # Per recommend.md Section 1: Apply FORMAL EKF gravity measurement update when stationary.

        gyro_norm = np.linalg.norm(w_corrected)

        # Add to rolling window (gyro only for stationary detection)
        self.zupt_window.append({'gyro': gyro_norm, 'accel': a_corrected.copy()})
        if len(self.zupt_window) > self.zupt_window_size:
            self.zupt_window.pop(0)

        # --- Gyro-only stationary detection (per recommend.md Section 5) ---
        # Ignore accelerometer when deciding if stationary!
        if gyro_norm < self.zupt_gyro_threshold:
            self.zupt_gyro_only_counter += 1
        else:
            self.zupt_gyro_only_counter = 0

        # Check if stationary based on gyro AND vision
        is_stationary = False

        # Decay vision motion cooldown
        if self.vision_motion_cooldown > 0:
            self.vision_motion_cooldown -= 1

        if len(self.zupt_window) >= self.zupt_window_size:
            avg_gyro = np.mean([s['gyro'] for s in self.zupt_window])

            # Per recommend.md: Use gyro-only detection, BUT also check vision
            # If vision recently showed motion (position correction), don't ZUPT!
            gyro_says_stationary = (avg_gyro < self.zupt_gyro_threshold or
                                    self.zupt_gyro_only_counter >= self.zupt_gyro_only_threshold)
            vision_says_moving = self.vision_motion_cooldown > 0

            if gyro_says_stationary and not vision_says_moving:
                is_stationary = True

                # --- ZUPT Part 1: Formal Velocity Update (per recommend.md Section 5) ---
                # Perform a FORMAL EKF update with z=0, v=0 instead of direct state override.
                # This updates both state AND covariance consistently.
                self._zupt_velocity_update()

                # --- ZUPT Part 2: Formal Gravity Measurement Update (per recommend.md Section 1) ---
                # Treat accelerometer as a TILT SENSOR during stationary periods.
                # This corrects orientation drift with proper covariance update.
                self._zupt_gravity_update(a_corrected)

        # ===================================================================
        # STEP 1: Nominal State Propagation (Non-linear Kinematics)
        # ROBOCENTRIC: v_b is in body frame, gravity must be transformed to body
        # ===================================================================

        # Gravity in body frame: g_b = R_wb^T @ g_w
        g_body = R_wb.T @ self.g

        # Body-frame acceleration: a_b = a_measured - ba - g_b - ω × v_b
        # The accelerometer measures: a_m = a_true - g (specific force)
        # So: a_true = a_m + g, but we want gravity-compensated acceleration
        # In body frame: acc_body = a_corrected - g_body (SUBTRACT, not add!)
        # The Coriolis term (ω × v_b) accounts for rotating reference frame
        coriolis = np.cross(w_corrected, v_b)
        acc_body = a_corrected - g_body - coriolis  # FIXED: subtract gravity

        # --- GROUND ROBOT CONSTRAINTS ---
        # For body-frame: Z-body acceleration should be ~0 (robot doesn't jump)
        # But we need to be careful - body Z is not world Z when tilted

        # Soft limit for extreme cases only (e.g., sensor glitches)
        MAX_ACCEL = 10.0  # m/s² - allow larger transients, only clip glitches
        acc_body_clipped = np.clip(acc_body, -MAX_ACCEL, MAX_ACCEL)

        acc_body_norm = np.linalg.norm(acc_body)

        # --- Divergence Watchdog ---
        if acc_body_norm > 20.0:  # Increased threshold - let filter work
            self.get_logger().error(f"Filter Divergence! Body Accel: {acc_body_norm:.1f} m/s^2. Re-initializing from IMU...")
            self._reinitialize_orientation(a_m, w_m)
            return

        # Debug logging (every ~2 seconds at 200Hz)
        if not hasattr(self, '_debug_counter'):
            self._debug_counter = 0
        self._debug_counter += 1
        if self._debug_counter % 400 == 1:
            self.get_logger().info(f"Body Accel: [{acc_body[0]:.3f}, {acc_body[1]:.3f}, {acc_body[2]:.3f}] m/s^2")

        # --- Velocity Update (Body Frame) ---
        # v_b_new = v_b + acc_body * dt
        v_b_new = v_b + acc_body_clipped * dt

        # For ground robot: body-frame Z velocity should be ~0
        # (robot doesn't move up/down relative to its own frame)
        v_b_new[2] = 0.0

        # Velocity safety limit - TIGHT for ground robot
        # Robot physically cannot exceed ~2 m/s, so any higher is divergence
        MAX_SPEED_PREDICT = 2.0  # m/s - physical limit for this robot
        speed = np.linalg.norm(v_b_new[:2])
        if speed > MAX_SPEED_PREDICT:
            v_b_new[:2] = v_b_new[:2] * (MAX_SPEED_PREDICT / speed)
            self.get_logger().warn(
                f"Speed limited from {speed:.2f} to {MAX_SPEED_PREDICT:.1f} m/s",
                throttle_duration_sec=1.0
            )

        # --- Position Update (World Frame) ---
        # p_w_new = p_w + R_wb @ v_b * dt
        # Note: We use v_b (not v_b_new) for position update (first-order integration)
        v_world = R_wb @ v_b  # Transform body velocity to world frame for position update
        p_new = p + v_world * dt

        # Ground constraint: Z position should be 0
        p_new[2] = 0.0

        # --- Orientation Update (Quaternion Integration) ---
        # Using quaternion exponential map: q_k = q_{k-1} ⊗ exp(0.5 * ω_corrected * dt)
        # Reference: MatthewHampsey/mekf model.py - quaternion derivative method
        w_norm = np.linalg.norm(w_corrected)

        # Log gyro activity for debugging rotation issues
        if not hasattr(self, '_gyro_debug_counter'):
            self._gyro_debug_counter = 0
        self._gyro_debug_counter += 1
        if self._gyro_debug_counter % 400 == 1 and w_norm > 0.05:
            self.get_logger().info(
                f"Gyro: wx={w_corrected[0]:.3f}, wy={w_corrected[1]:.3f}, wz={w_corrected[2]:.3f} rad/s | "
                f"bias: [{bg[0]:.4f},{bg[1]:.4f},{bg[2]:.4f}]"
            )

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

        # --- GROUND ROBOT CONSTRAINT: Limit Roll/Pitch ---
        # A ground robot cannot physically have large roll or pitch angles.
        # Clamp to ±10 degrees to prevent divergence.
        MAX_TILT = np.radians(10.0)  # 10 degrees max tilt
        rot_new = R.from_quat([q_new[1], q_new[2], q_new[3], q_new[0]])
        euler_new = rot_new.as_euler('zyx')  # [yaw, pitch, roll]
        yaw_new, pitch_new, roll_new = euler_new

        # Clamp roll and pitch
        if abs(roll_new) > MAX_TILT or abs(pitch_new) > MAX_TILT:
            roll_clamped = np.clip(roll_new, -MAX_TILT, MAX_TILT)
            pitch_clamped = np.clip(pitch_new, -MAX_TILT, MAX_TILT)
            # Reconstruct quaternion with clamped angles
            rot_clamped = R.from_euler('zyx', [yaw_new, pitch_clamped, roll_clamped])
            q_clamped = rot_clamped.as_quat()  # [x, y, z, w]
            q_new = np.array([q_clamped[3], q_clamped[0], q_clamped[1], q_clamped[2]])
            self.get_logger().warn(
                f"Orientation clamped: roll {np.degrees(roll_new):.1f}→{np.degrees(roll_clamped):.1f}°, "
                f"pitch {np.degrees(pitch_new):.1f}→{np.degrees(pitch_clamped):.1f}°",
                throttle_duration_sec=2.0
            )

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


        # Update Nominal State
        self.x[0:3] = p_new
        self.x[3:6] = v_b_new  # Body-frame velocity
        self.x[6:10] = q_new
        self.x[10:13] = ba_new
        self.x[13:16] = bg_new

        # ===================================================================
        # STEP 2: Error Covariance Propagation (Linearized Uncertainty)
        # ROBOCENTRIC: Jacobians updated for body-frame velocity
        # ===================================================================
        # The error state Jacobian Fx describes how errors propagate.
        # Fx ≈ I + F*dt where F is the continuous-time error dynamics matrix.
        #
        # Reference: MatthewHampsey/mekf kalman2.py lines 55-69
        #
        # Error state ordering: [δp, δv_b, δθ, δba, δbg]
        #                       [0:3, 3:6, 6:9, 9:12, 12:15]

        # Construct continuous-time Jacobian F for ROBOCENTRIC formulation
        F = np.zeros((15, 15))

        # --- Position error dynamics (world frame) ---
        # p_w = ∫ R_wb @ v_b dt
        # δṗ = R_wb @ δv_b + [R_wb @ v_b]_× @ δθ
        # d(δp)/d(δv_b): Position depends on body velocity through rotation
        F[0:3, 3:6] = R_wb
        # d(δp)/d(δθ): Orientation error affects how body velocity maps to world
        F[0:3, 6:9] = -skew_symmetric(R_wb @ v_b)

        # --- Velocity error dynamics (body frame) ---
        # v_b_dot = a_b - ba - g_b - ω × v_b  (FIXED: subtract g_b)
        # δv̇_b = -[a_corrected - g_b]_× @ δθ - δba - [ω]_× @ δv_b + [v_b]_× @ δbg
        #
        # d(δv_b)/d(δv_b): Coriolis coupling -[ω]_×
        F[3:6, 3:6] = -skew_symmetric(w_corrected)
        # d(δv_b)/d(δθ): How orientation error affects gravity and acceleration
        # Since acc_body = a_corrected - g_body, the Jacobian uses this difference
        F[3:6, 6:9] = -skew_symmetric(a_corrected - g_body)  # FIXED: subtract g_body
        # d(δv_b)/d(δba): Direct effect of accel bias on body acceleration
        F[3:6, 9:12] = -np.eye(3)
        # d(δv_b)/d(δbg): Gyro bias affects Coriolis term
        F[3:6, 12:15] = skew_symmetric(v_b)

        # --- Orientation error dynamics ---
        # δθ̇ = -[ω_corrected]_× * δθ - δbg
        # d(δθ)/d(δθ): Gyro measurement coupling (MEKF key insight!)
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

        # Per recommendation2.md: Remove manual covariance clamping!
        # Forcing diagonal elements to min/max values without updating off-diagonals
        # breaks the mathematical consistency of the filter, causing P to lose
        # positive-definiteness (the "Invalid Mahalanobis distance" errors).
        #
        # Instead, use proper regularization: add small diagonal if needed
        # to ensure positive semi-definiteness.

        # Check for numerical issues and regularize if needed
        min_diag = np.min(np.diag(self.P))
        if min_diag < 1e-8:
            # Add small regularization to diagonal
            self.P += np.eye(15) * 1e-8
            self.get_logger().warn("Covariance regularization applied", throttle_duration_sec=5.0)

        # Ground robot Z-constraint: Apply as soft constraint via increased process noise
        # Per recommend.md: DON'T zero out rows/columns as that creates near-singular matrix
        # Instead, keep Z uncertainty small but mathematically valid
        # Scale down Z-related covariances while preserving positive-definiteness
        z_damping = 0.1  # Damp Z-related covariances each step
        self.P[2, 0:2] *= z_damping
        self.P[0:2, 2] *= z_damping
        self.P[2, 3:] *= z_damping
        self.P[3:, 2] *= z_damping
        self.P[5, 0:5] *= z_damping
        self.P[0:5, 5] *= z_damping
        self.P[5, 6:] *= z_damping
        self.P[6:, 5] *= z_damping
        # Keep diagonal small but non-zero
        self.P[2, 2] = max(self.P[2, 2] * z_damping, 1e-4)
        self.P[5, 5] = max(self.P[5, 5] * z_damping, 1e-4)

        # Ensure symmetry again after Z-constraint
        self.P = 0.5 * (self.P + self.P.T)

        # --- State Buffer for Time Delay Compensation ---
        # Per recommend.md: Store state history to match vision with correct pose
        if self.last_imu_time is not None:
            self.state_buffer.append({
                'time': self.last_imu_time,
                'x': self.x.copy(),
                'P': self.P.copy()
            })
            if len(self.state_buffer) > self.state_buffer_size:
                self.state_buffer.pop(0)

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
        Handle visual landmark observations with BATCH update.

        Per recommendation2.md: Instead of calling update() for each marker,
        stack all visible markers into a single batch measurement and update
        the filter once. This ensures all landmarks contribute to a single
        consistent correction, avoiding linearization errors from sequential updates.

        Per recommend.md: Use state buffer to find pose at image capture time,
        not current time, to avoid linearization errors from motion during delay.
        """
        # Skip if not initialized
        if not self.initialized:
            return

        # Time synchronization check
        if self.last_imu_time is None:
            self.get_logger().warn("Vision update skipped: No IMU data received yet", throttle_duration_sec=2.0)
            return

        vision_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        time_diff = abs(vision_time - self.last_imu_time)

        # --- Find buffered state closest to image timestamp ---
        # Per recommend.md: Calculate residual using pose at image capture time
        buffered_state = None
        if len(self.state_buffer) > 0:
            # Find state with timestamp closest to vision_time
            min_dt = float('inf')
            for state in self.state_buffer:
                dt = abs(state['time'] - vision_time)
                if dt < min_dt:
                    min_dt = dt
                    buffered_state = state

            # Only use buffer if we found a reasonably close match
            if min_dt > self.max_time_delay:
                self.get_logger().warn(
                    f"Vision-IMU time mismatch: {time_diff:.3f}s. Using current state.",
                    throttle_duration_sec=5.0
                )
                buffered_state = None

        # Collect all valid observations for batch update
        observations = []
        for obs in msg.poses:
            lid = obs.position.z
            u_meas = obs.position.x
            v_meas = obs.position.y

            if lid in self.map:
                observations.append({
                    'lm_pos': self.map[lid],
                    'u': u_meas,
                    'v': v_meas,
                    'id': lid
                })

        # Perform batch update only with 2+ markers for observability
        # Single-marker updates cause position-yaw coupling errors and are the
        # primary cause of "6km jumps" in high-drift states
        if len(observations) >= 2:
            self.batch_update(observations, buffered_state)
        elif len(observations) == 1:
            self.get_logger().info(
                f"Skipping single-marker update (ID={observations[0]['id']:.0f}) - need 2+ for observability",
                throttle_duration_sec=1.0
            )

    def batch_update(self, observations, buffered_state=None):
        """
        ITERATED EKF (IEKF) batch update using all visible landmarks.

        Standard EKF linearizes once at the prior estimate, which fails for large
        residuals (650+ pixels). IEKF iterates: re-linearize at updated estimate
        until convergence, effectively performing Gauss-Newton optimization.

        Args:
            observations: List of dicts with 'lm_pos', 'u', 'v', 'id'
            buffered_state: Optional state dict from buffer matching image timestamp
        """
        n_obs = len(observations)
        if n_obs == 0:
            return

        # Store original state for potential rollback
        x_orig = self.x.copy()
        P_orig = self.P.copy()

        # Use buffered covariance if available
        if buffered_state is not None:
            P_prior = buffered_state['P'].copy()
        else:
            P_prior = self.P.copy()

        # IEKF iteration parameters
        MAX_ITERATIONS = 4
        CONVERGENCE_THRESHOLD = 0.01  # meters for position, radians for orientation

        # Working state for iterations (start from current estimate)
        x_iter = self.x.copy()

        for iteration in range(MAX_ITERATIONS):
            # Extract current iteration state
            p_w = x_iter[0:3]
            q_w = x_iter[6:10]  # [w,x,y,z]
            R_wb = R.from_quat([q_w[1], q_w[2], q_w[3], q_w[0]]).as_matrix()

            # Build measurement model at current linearization point
            z_res_stack = []
            H_stack = []
            valid_count = 0

            for obs in observations:
                lm_pos_world = obs['lm_pos']
                u_meas = obs['u']
                v_meas = obs['v']
                lm_id = obs.get('id', -1)

                # --- First-Estimate Jacobians (FEJ) for orientation part ---
                # Store first estimate for new landmarks
                if lm_id not in self.landmark_first_estimates:
                    self.landmark_first_estimates[lm_id] = {
                        'p': p_w.copy(),
                        'R': R_wb.copy()
                    }
                    if iteration == 0:
                        self.get_logger().info(f"FEJ: First observation of LM{int(lm_id)}, storing pose")

                # Use FEJ rotation for position Jacobian (prevents yaw observability issues)
                R_fej = self.landmark_first_estimates[lm_id]['R']

                # Transform World -> Body using CURRENT iteration state
                p_b = R_wb.T @ (lm_pos_world - p_w)

                # Transform Body -> Camera Optical Frame
                p_c = self.R_b_c @ (p_b - self.t_b_c)

                # Skip if behind camera
                if p_c[2] < 0.1:
                    continue

                # Project to pixels
                u_pred = self.K[0,0] * p_c[0]/p_c[2] + self.K[0,2]
                v_pred = self.K[1,1] * p_c[1]/p_c[2] + self.K[1,2]

                # Residual
                z_res = np.array([u_meas - u_pred, v_meas - v_pred])

                # GATE large residuals - reject observations with > 150px error
                # These indicate wrong data association or severely wrong state estimate
                residual_norm = np.linalg.norm(z_res)
                if residual_norm > 150.0:
                    if iteration == 0:
                        self.get_logger().warn(
                            f"  LM{int(lm_id)}: REJECTED (res={residual_norm:.0f}px > 150px gate)",
                            throttle_duration_sec=0.5
                        )
                    continue  # Skip this observation

                # Log on first iteration only
                if iteration == 0:
                    self.get_logger().info(
                        f"  LM{int(lm_id)}: world={lm_pos_world}, p_c=[{p_c[0]:.2f},{p_c[1]:.2f},{p_c[2]:.2f}], "
                        f"pred=({u_pred:.0f},{v_pred:.0f}), meas=({u_meas:.0f},{v_meas:.0f}), "
                        f"res=[{z_res[0]:.0f},{z_res[1]:.0f}]px",
                        throttle_duration_sec=0.5
                    )

                # Jacobian of Projection
                fx = self.K[0,0]
                fy = self.K[1,1]
                X, Y, Z = p_c
                J_proj = np.array([
                    [fx/Z, 0, -fx*X/Z**2],
                    [0, fy/Z, -fy*Y/Z**2]
                ])

                # Jacobian w.r.t Position (use FEJ rotation)
                J_pos = -self.R_b_c @ R_fej.T

                # Assemble H (2x15) for this landmark - POSITION ONLY
                H = np.zeros((2, 15))
                H[:, 0:3] = J_proj @ J_pos

                z_res_stack.append(z_res)
                H_stack.append(H)
                valid_count += 1

            if valid_count == 0:
                return

            # Enforce 2+ marker requirement
            if valid_count < 2:
                self.get_logger().info(
                    f"Skipping update: only {valid_count} valid marker(s) after filtering",
                    throttle_duration_sec=1.0
                )
                return

            # Stack into matrices
            z_res_all = np.concatenate(z_res_stack)
            H_all = np.vstack(H_stack)
            R_all = np.eye(2 * valid_count) * self.R_cam

            # IEKF uses prior covariance, not updated covariance
            S = H_all @ P_prior @ H_all.T + R_all
            S = 0.5 * (S + S.T) + np.eye(S.shape[0]) * 1e-6

            try:
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                self.get_logger().warn("Singular S matrix in IEKF, skipping")
                return

            # Compute Kalman gain using PRIOR covariance
            K = P_prior @ H_all.T @ S_inv

            # Compute state correction
            dx = K @ z_res_all

            # # CRITICAL: Vision observes POSITION ONLY.
            # # Zero out all other corrections - they come from P cross-correlations,
            # # NOT from actual observability. Allowing these phantom corrections
            # # corrupts orientation → gravity cancellation fails → divergence.
            # dx[3:6] = 0.0   # Velocity - not directly observed
            # dx[6:9] = 0.0   # Orientation - not observed
            # dx[9:15] = 0.0  # Biases - not observed

            # Apply correction to iteration state (position only now)
            x_iter[0:3] += dx[0:3]  # Position

            # Check convergence
            pos_change = np.linalg.norm(dx[0:3])

            if iteration > 0 and pos_change < CONVERGENCE_THRESHOLD:
                self.get_logger().info(
                    f"IEKF converged in {iteration+1} iterations, final residual={np.linalg.norm(z_res_all)/np.sqrt(valid_count):.1f}px",
                    throttle_duration_sec=1.0
                )
                break

        # After iteration, check if result is reasonable
        total_pos_correction = np.linalg.norm(x_iter[0:3] - x_orig[0:3])
        pixel_residual = np.linalg.norm(z_res_all) / np.sqrt(valid_count)

        # Sanity check on position correction - allow larger corrections for recovery
        # Too small = filter can never recover from drift
        # Too large = bad measurements cause jumps
        MAX_TOTAL_POS = 1.5  # Allow up to 1.5m correction to enable recovery

        if total_pos_correction > MAX_TOTAL_POS:
            self.get_logger().warn(
                f"IEKF correction too large: pos={total_pos_correction:.2f}m > {MAX_TOTAL_POS}m. REJECTING update."
            )
            # Restore original state and covariance - don't apply bad update
            self.x = x_orig.copy()
            self.P = P_orig.copy()
            return

        # Track this correction for vision-based motion detection
        # If we're making significant corrections, robot is moving
        self.last_vision_correction = total_pos_correction
        if total_pos_correction > self.vision_motion_threshold:
            self.vision_motion_cooldown = 100  # ~0.5s at 200Hz - don't ZUPT for a while

        # Apply final state - POSITION ONLY
        self.x[0:3] = x_iter[0:3]
        # Velocity and orientation unchanged (vision doesn't observe them directly)

        # # Update covariance - but ONLY for position states
        # # Zero out K rows for unobserved states to prevent false certainty
        # K_masked = K.copy()
        # K_masked[3:, :] = 0.0  # Zero velocity, orientation, bias rows

        I = np.eye(15)
        # IKH = I - K_masked @ H_all
        IKH = I - K @ H_all
        # self.P = IKH @ P_prior @ IKH.T + K_masked @ R_all @ K_masked.T
        self.P = IKH @ P_prior @ IKH.T + K @ R_all @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        min_eig = np.min(np.diag(self.P))
        if min_eig < 1e-10:
            self.P += np.eye(15) * 1e-10

        # Ground constraints
        self.x[2] = 0.0  # Z position
        self.x[5] = 0.0  # Z velocity (body frame)

        # Apply soft Z-constraint to covariance
        z_damping = 0.5
        self.P[2, :] *= z_damping
        self.P[:, 2] *= z_damping
        self.P[5, :] *= z_damping
        self.P[:, 5] *= z_damping
        self.P[2, 2] = max(self.P[2, 2], 1e-4)
        self.P[5, 5] = max(self.P[5, 5], 1e-4)

        # Velocity sanity check
        MAX_SPEED = 1.0
        speed = np.linalg.norm(self.x[3:5])
        if speed > MAX_SPEED:
            self.x[3:5] = self.x[3:5] * (MAX_SPEED / speed)
            self.get_logger().warn(f"Speed clamped from {speed:.2f} to {MAX_SPEED} m/s")

        # Log update
        q = self.x[6:10]
        yaw = R.from_quat([q[1], q[2], q[3], q[0]]).as_euler('zyx')[0]
        self.get_logger().info(
            f"IEKF UPDATE ({valid_count} markers, {iteration+1} iters): pos=[{self.x[0]:.2f}, {self.x[1]:.2f}]m, "
            f"yaw={np.degrees(yaw):.1f}°, residual={pixel_residual:.1f}px",
            throttle_duration_sec=0.5
        )

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

        # Transform Body -> Camera Optical Frame
        # p_c = R_b_c @ (p_b - t_b_c)
        # This is correct: first translate to camera origin, then rotate to optical frame
        p_c = self.R_b_c @ (p_b - self.t_b_c)

        # Check if behind camera (Z in optical frame is depth)
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
        # DISABLED - vision updates position only, let IMU+ZUPT handle orientation
        # d(pc)/d(theta) = R_bc * [p_b]_× (skew symmetric of point in body)
        # J_rot = self.R_b_c @ skew_symmetric(p_b)

        # Assemble H (2x15) - POSITION ONLY
        H = np.zeros((2, 15))
        H[:, 0:3] = J_proj @ J_pos
        # H[:, 6:9] = J_proj @ J_rot  # DISABLED

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

        # Handle numerical issues with Mahalanobis distance
        if mahalanobis_sq < 0 or np.isnan(mahalanobis_sq):
            self.get_logger().warn("Invalid Mahalanobis distance, skipping update")
            return
        mahal_dist = np.sqrt(mahalanobis_sq)

        # Adaptive gating: if covariance is large, allow larger corrections
        # Chi-squared 99% for 2 DOF is 9.21, 99.9% is 13.82
        # Since we KNOW landmark positions, we can be more aggressive
        adaptive_threshold = max(self.mahalanobis_threshold, 10.0)

        if mahal_dist > adaptive_threshold:
            self.consecutive_outliers += 1

            # Per recommendation2.md: Use robust M-estimator (Huber) instead of
            # binary accept/reject. This down-weights outliers gracefully.
            if self.consecutive_outliers >= self.max_consecutive_outliers:
                # Instead of expanding covariance (which causes jumps),
                # apply Huber-weighted update with reduced gain
                huber_threshold = 3.0  # Standard Huber threshold
                huber_weight = huber_threshold / mahal_dist  # < 1 for outliers
                self.get_logger().warn(
                    f"Robust update: Mahal={mahal_dist:.1f}, Huber weight={huber_weight:.2f}"
                )
                # Scale the innovation by Huber weight
                z_res = z_res * huber_weight
                self.consecutive_outliers = 0
                # Continue with weighted update below (don't return)
            else:
                self.get_logger().warn(
                    f"Outlier Rejected! Residual: {pixel_residual:.1f} px, "
                    f"Mahal: {mahal_dist:.1f} (thresh={adaptive_threshold:.1f})"
                )
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
        # CRITICAL FIX: Since H[:,6:9] = 0 (vision doesn't observe orientation),
        # any non-zero dx[6:9] comes from P cross-correlations, NOT from actual
        # orientation observability. Zero out phantom corrections.
        dx[6:9] = 0.0    # orientation - not observed by vision
        dx[9:12] = 0.0   # accel bias - not observed by vision
        dx[12:15] = 0.0  # gyro bias - not observed by vision

        # Position correction
        pos_correction = dx[0:3]
        pos_corr_norm = np.linalg.norm(pos_correction)
        if pos_corr_norm > 0.5:  # Limit per-update correction
            pos_correction = pos_correction * (0.5 / pos_corr_norm)
            self.get_logger().warn(f"Position correction clipped: {pos_corr_norm:.2f}m", throttle_duration_sec=1.0)
        self.x[0:3] += pos_correction

        # Velocity correction
        vel_correction = dx[3:6]
        vel_corr_norm = np.linalg.norm(vel_correction)
        if vel_corr_norm > 0.3:  # Conservative limit
            vel_correction = vel_correction * (0.3 / vel_corr_norm)
            self.get_logger().warn(f"Velocity correction clipped: {vel_corr_norm:.2f}m/s", throttle_duration_sec=1.0)
        self.x[3:6] += vel_correction

        # Biases - NOT updated from vision (set to zero above)

        # Enforce bias magnitude limits (physical bounds, not rate limits)
        MAX_ACCEL_BIAS = 0.5  # m/s² - realistic MEMS limit
        MAX_GYRO_BIAS = 0.1   # rad/s - realistic MEMS limit
        self.x[10:13] = np.clip(self.x[10:13], -MAX_ACCEL_BIAS, MAX_ACCEL_BIAS)
        self.x[13:16] = np.clip(self.x[13:16], -MAX_GYRO_BIAS, MAX_GYRO_BIAS)

        # --- GROUND ROBOT CONSTRAINTS (post-update) ---
        # Per recommendation2.md: When zeroing state, also zero corresponding covariance
        # to prevent P mismatch (state forbidden to move but covariance grows)
        self.x[2] = 0.0  # Z position = 0
        self.x[5] = 0.0  # Z velocity = 0
        # Zero out Z covariance rows/columns to match state constraint
        self.P[2, :] = 0.0
        self.P[:, 2] = 0.0
        self.P[2, 2] = 1e-6  # Small non-zero for numerical stability
        self.P[5, :] = 0.0
        self.P[:, 5] = 0.0
        self.P[5, 5] = 1e-6

        # Final velocity sanity check - CONSERVATIVE for TurtleBot
        MAX_SPEED = 0.5  # Reduced from 2.0 - TurtleBot max is ~0.26 m/s
        speed = np.linalg.norm(self.x[3:5])
        if speed > MAX_SPEED:
            self.x[3:5] = self.x[3:5] * (MAX_SPEED / speed)
            self.get_logger().warn(f"Velocity clamped from {speed:.2f} to {MAX_SPEED} m/s", throttle_duration_sec=2.0)

        # --- Log Robot Belief After Measurement Update ---
        # Extract yaw from quaternion for readability
        q = self.x[6:10]  # [w, x, y, z]
        yaw = R.from_quat([q[1], q[2], q[3], q[0]]).as_euler('zyx')[0]
        # Handle potential negative diagonal (use abs to avoid sqrt warning)
        pos_diag = np.diag(self.P[0:3, 0:3])
        pos_std = np.sqrt(np.maximum(pos_diag, 0.0))
        ba = self.x[10:13]  # Accel bias
        self.get_logger().info(
            f"BELIEF: pos=[{self.x[0]:.2f}, {self.x[1]:.2f}]m, yaw={np.degrees(yaw):.1f}°, "
            f"vel=[{self.x[3]:.2f}, {self.x[4]:.2f}]m/s, ba=[{ba[0]:.3f}, {ba[1]:.3f}]m/s²",
            throttle_duration_sec=0.5
        )

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

        # Twist is in body frame (child_frame_id = base_footprint)
        # This matches ROS convention and our robocentric state representation
        odom.twist.twist.linear.x = self.x[3]  # Body-frame forward velocity
        odom.twist.twist.linear.y = self.x[4]  # Body-frame lateral velocity
        odom.twist.twist.linear.z = self.x[5]  # Body-frame vertical velocity

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