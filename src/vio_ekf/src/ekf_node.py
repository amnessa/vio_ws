#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, CameraInfo
from geometry_msgs.msg import PoseArray, PoseStamped, TransformStamped, Twist
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Float64MultiArray
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

        # =======================================================================
        # DECLARE PARAMETERS (can be set via YAML config or command line)
        # =======================================================================
        # Filter mode switches
        self.declare_parameter('enable_prediction', True)
        self.declare_parameter('enable_correction', True)
        self.declare_parameter('enable_zupt', True)

        # Process noise parameters (Q)
        self.declare_parameter('sigma_accel', 1.5)
        self.declare_parameter('sigma_gyro', 0.5)
        self.declare_parameter('sigma_accel_bias', 0.01)  # sqrt(Q_ba)
        self.declare_parameter('sigma_gyro_bias', 0.01)  # sqrt(Q_bg)

        # Measurement noise parameters (R) - Range-Bearing model
        self.declare_parameter('R_range', 0.1)      # Range noise std (m)
        self.declare_parameter('R_bearing', 0.05)   # Bearing noise std (rad)
        self.declare_parameter('R_zupt_velocity', 0.001)

        # ZUPT parameters
        self.declare_parameter('zupt_gyro_threshold', 0.02)

        # Outlier rejection
        self.declare_parameter('mahalanobis_threshold', 60.0)

        # Timing parameters
        self.declare_parameter('imu_downsample_factor', 4)
        self.declare_parameter('max_time_delay', 0.15)

        # Synthetic prediction parameters (for measurement-only testing)
        self.declare_parameter('synthetic_velocity', 0.0)  # m/s, frozen forward velocity
        self.declare_parameter('synthetic_omega', 0.0)     # rad/s, frozen yaw rate

        # =======================================================================
        # READ PARAMETERS
        # =======================================================================
        self.enable_prediction = self.get_parameter('enable_prediction').value
        self.enable_correction = self.get_parameter('enable_correction').value
        self.enable_zupt = self.get_parameter('enable_zupt').value

        # Synthetic prediction parameters
        self.synthetic_velocity = self.get_parameter('synthetic_velocity').value
        self.synthetic_omega = self.get_parameter('synthetic_omega').value

        # Log filter mode
        mode_str = []
        if self.enable_prediction:
            mode_str.append("PREDICTION(IMU)")
        else:
            mode_str.append(f"PREDICTION(SYNTHETIC v={self.synthetic_velocity:.2f}m/s, ω={self.synthetic_omega:.2f}rad/s)")
        if self.enable_correction:
            mode_str.append("CORRECTION")
        if self.enable_zupt:
            mode_str.append("ZUPT")
        self.get_logger().info(f"=== FILTER MODE: {' + '.join(mode_str)} ===")

        # --- Initialization Phase ---
        # Collect IMU samples while stationary to estimate biases and initial orientation
        self.initialized = False
        self.init_samples = []
        self.init_sample_count = 1200  # Reduced from 2400 (1 second at 200Hz)
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

        # Noise Parameters - Read from parameters
        self.sigma_a = self.get_parameter('sigma_accel').value
        self.sigma_g = self.get_parameter('sigma_gyro').value
        self.Q_a = self.sigma_a ** 2  # Accel noise variance
        self.Q_g = self.sigma_g ** 2  # Gyro noise variance
        sigma_ba = self.get_parameter('sigma_accel_bias').value
        sigma_bg = self.get_parameter('sigma_gyro_bias').value
        self.Q_ba = sigma_ba ** 2  # Accel bias random walk
        self.Q_bg = sigma_bg ** 2  # Gyro bias random walk

        # Range-Bearing measurement noise
        self.R_range = self.get_parameter('R_range').value ** 2    # Variance (m²)
        self.R_bearing = self.get_parameter('R_bearing').value ** 2  # Variance (rad²)

        self.get_logger().info(f"Process Noise: sigma_a={self.sigma_a}, sigma_g={self.sigma_g}")
        self.get_logger().info(f"Bias Walk: Q_ba={self.Q_ba:.2e}, Q_bg={self.Q_bg:.2e}")
        self.get_logger().info(f"Measurement Noise: R_range={np.sqrt(self.R_range):.3f}m, R_bearing={np.sqrt(self.R_bearing):.3f}rad")

        # ZUPT (Zero-Velocity Update) parameters
        # Per recommend.md Section 5: Trigger ZUPT based solely on gyro activity
        # Ignore accelerometer deviation as it may be caused by orientation error
        self.zupt_gyro_threshold = self.get_parameter('zupt_gyro_threshold').value
        self.zupt_window = []  # Rolling window of IMU samples
        self.zupt_window_size = 50  # ~250ms at 200Hz
        self.zupt_gyro_only_counter = 0  # Counter for gyro-only stationary detection
        self.zupt_gyro_only_threshold = 100  # 0.5 second at 200Hz - if gyro stable this long, definitely stationary

        # Formal ZUPT measurement noise (small = high confidence velocity is zero)
        self.R_zupt_velocity = self.get_parameter('R_zupt_velocity').value
        self.R_zupt_gravity = 0.1  # Gravity direction measurement noise for tilt correction

        # Vision-based motion detection to prevent false ZUPT
        # If vision shows robot is moving, disable ZUPT even if gyro is quiet
        self.last_vision_correction = 0.0  # Position correction from last vision update
        self.vision_motion_threshold = 0.02  # 2cm correction = robot is moving
        self.vision_motion_cooldown = 0  # Frames since significant vision correction

        # IMU Downsampling: 200Hz → 50Hz by averaging every 4 samples
        # This smooths out spikes and noise before EKF sees them
        # A single 50 m/s² spike becomes 12.5 m/s² after averaging with 3 normal samples
        self.imu_buffer = []  # Buffer to collect 4 samples
        self.imu_downsample_factor = self.get_parameter('imu_downsample_factor').value
        self.imu_accumulated_dt = 0.0  # Accumulated time for the batch

        # Outlier rejection settings
        self.mahalanobis_threshold = self.get_parameter('mahalanobis_threshold').value
        self.consecutive_outliers = 0
        self.max_consecutive_outliers = 5  # Quick recovery after just 3 rejections

        # Gravity correction gain (for attitude correction from accelerometer)
        # Higher values = faster convergence but more noise sensitivity
        # Typical complementary filter uses 0.02-0.1 for α
        # Per recommendation.md: Reduce further or disable to prevent jitter
        # The EKF should handle orientation through proper measurement updates
        self.gravity_correction_gain = 0.005  # Very conservative - minimal jitter

        # Time sync tolerance (max acceptable delay between IMU and vision)
        self.max_time_delay = self.get_parameter('max_time_delay').value

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

        # =====================================================================
        # IMU Extrinsics: IMU frame -> Body frame
        # =====================================================================
        # IMU position relative to body frame origin (center of rotation)
        # The IMU is mounted at [0, 0, 0.068] in body frame
        # This offset causes lever-arm effects during rotation
        self.t_b_imu = np.array([0.0, 0.0, 0.068])  # [x, y, z] in body frame

        # IMU orientation relative to body frame (identity = aligned)
        # If IMU axes are aligned with body axes, this is identity
        self.R_b_imu = np.eye(3)  # IMU frame aligned with body frame

        # Enable/disable lever-arm compensation
        self.declare_parameter('enable_imu_lever_arm', True)
        self.enable_imu_lever_arm = self.get_parameter('enable_imu_lever_arm').value

        if np.linalg.norm(self.t_b_imu) > 0.001:
            self.get_logger().info(f"IMU lever-arm: [{self.t_b_imu[0]:.3f}, {self.t_b_imu[1]:.3f}, {self.t_b_imu[2]:.3f}] m")
            if self.enable_imu_lever_arm:
                self.get_logger().info("IMU lever-arm compensation: ENABLED")
            else:
                self.get_logger().info("IMU lever-arm compensation: DISABLED")

        # =====================================================================
        # Camera Extrinsics: Body frame -> Camera Optical Frame
        # =====================================================================
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
        self.pub_diag = self.create_publisher(Float64MultiArray, '/vio/diagnostics', 10)
        self.tf_br = TransformBroadcaster(self)

        # cmd_vel publisher for synthetic prediction mode
        # When enable_prediction=False, we send cmd_vel commands matching synthetic velocity/omega
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)

        # Log synthetic mode info
        if not self.enable_prediction:
            self.get_logger().info(f"Synthetic mode: cmd_vel will be published after initialization")
            self.get_logger().info(f"  linear.x = {self.synthetic_velocity} m/s, angular.z = {self.synthetic_omega} rad/s")

        self.get_logger().info("EKF Node Initialized")

    def info_callback(self, msg):
        # Update intrinsics matrix K
        self.K = np.array(msg.k).reshape(3,3)

    def publish_synthetic_cmd_vel(self):
        """
        Publish cmd_vel commands for synthetic prediction mode.

        When enable_prediction=False, we use synthetic velocity/omega for EKF prediction.
        This function sends matching cmd_vel commands to actually move the robot,
        so the simulated motion matches our synthetic prediction model.
        """
        cmd = Twist()
        cmd.linear.x = float(self.synthetic_velocity)
        cmd.linear.y = 0.0
        cmd.linear.z = 0.0
        cmd.angular.x = 0.0
        cmd.angular.y = 0.0
        cmd.angular.z = float(self.synthetic_omega)

        self.pub_cmd_vel.publish(cmd)

    def imu_callback(self, msg):
        # Extract measurements (in IMU frame)
        a_imu = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
        w_imu = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])

        # =====================================================================
        # IMU to Body Frame Transformation
        # =====================================================================
        # Transform gyro from IMU frame to body frame
        # w_body = R_b_imu @ w_imu
        w_m = self.R_b_imu @ w_imu

        # Transform accel from IMU frame to body frame with lever-arm compensation
        # The IMU measures: a_imu = a_body + ω × (ω × r) + α × r
        # Where:
        #   - ω × (ω × r): centripetal acceleration (points toward rotation axis)
        #   - α × r: tangential acceleration (from angular acceleration)
        #   - r = t_b_imu: lever arm from body origin to IMU
        #
        # To get body-frame acceleration: a_body = R_b_imu @ a_imu - ω × (ω × r) - α × r
        # For simplicity, we ignore α × r (requires gyro derivative, typically small)

        a_m = self.R_b_imu @ a_imu

        if self.enable_imu_lever_arm and np.linalg.norm(self.t_b_imu) > 0.001:
            # Centripetal acceleration compensation: a_cent = ω × (ω × r)
            # This is the acceleration the IMU feels due to rotating around body origin
            centripetal = np.cross(w_m, np.cross(w_m, self.t_b_imu))
            a_m = a_m - centripetal

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
        if dt > 0.1:
            self.get_logger().warn(f"Huge time jump detected (dt={dt:.4f}s). Skipping prediction step.")
            self.last_imu_time = curr_time
            self.imu_buffer.clear()  # Clear buffer on time jump
            self.imu_accumulated_dt = 0.0
            return
        if dt <= 0:
            return  # Skip backwards or duplicate messages
        self.last_imu_time = curr_time

        # --- IMU DOWNSAMPLING: 200Hz → 50Hz ---
        # Accumulate samples and average every N samples
        # This smooths out spikes: a single 50 m/s² spike with 3 normal samples → 12.5 m/s²
        self.imu_buffer.append({'accel': a_m, 'gyro': w_m})
        self.imu_accumulated_dt += dt

        if len(self.imu_buffer) < self.imu_downsample_factor:
            # Not enough samples yet, wait for more
            return

        # Average the buffered samples
        avg_accel = np.mean([s['accel'] for s in self.imu_buffer], axis=0)
        avg_gyro = np.mean([s['gyro'] for s in self.imu_buffer], axis=0)
        batch_dt = self.imu_accumulated_dt

        # Clear buffer for next batch
        self.imu_buffer.clear()
        self.imu_accumulated_dt = 0.0

        # --- PREDICTION STEP ---
        if self.enable_prediction:
            # Normal IMU-based prediction
            self.predict(batch_dt, avg_accel, avg_gyro)
        else:
            # Synthetic "frozen" prediction for measurement-only testing
            # This keeps the filter alive (P stays healthy) while ignoring IMU noise
            self.predict_synthetic(batch_dt, self.synthetic_velocity, self.synthetic_omega)
            # Also send cmd_vel to actually move the robot in simulation
            self.publish_synthetic_cmd_vel()

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

        if not self.enable_prediction:
            self.get_logger().info("=== SYNTHETIC MODE: cmd_vel commands will now be published ===")

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

        # Keep current bias estimates - let the EKF covariance handle uncertainty
        # Resetting biases loses valuable calibration information
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
        PLANAR VERSION: Simplified gravity measurement for XY accel bias estimation.

        For a planar robot with roll/pitch = 0, the accelerometer z-axis should
        measure exactly g when stationary. The XY components should measure zero.

        This update estimates XY accelerometer biases only (no orientation correction
        since we enforce roll/pitch = 0 in the planar model).

        Measurement model (planar, stationary):
          a_measured_xy = 0 + bias_xy  => bias_xy = a_measured_xy
          a_measured_z = g + bias_z    => (bias_z locked to 0)
        """
        # For planar robot: XY acceleration should be zero when stationary
        # The XY readings are directly the XY biases
        a_xy = a_corrected[0:2]  # XY acceleration in body frame

        # Skip if XY residual is very small
        if np.linalg.norm(a_xy) < 0.02:
            return

        # Measurement: a_xy should be zero when stationary
        # Residual z = 0 - a_xy = -a_xy
        z_res = -a_xy

        # Measurement Jacobian H (2x15)
        # Only XY accel bias affects XY accelerometer reading
        # d(a_xy)/d(ba_xy) = -I  (since a_corrected = a_raw - ba)
        H = np.zeros((2, 15))
        H[0, 9] = -1.0   # d(ax)/d(ba_x)
        H[1, 10] = -1.0  # d(ay)/d(ba_y)

        # Measurement noise (XY gravity/bias)
        R_gravity_xy = np.eye(2) * self.R_zupt_gravity

        # Innovation covariance
        S = H @ self.P @ H.T + R_gravity_xy

        # Kalman gain
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            self.get_logger().warn("ZUPT gravity update: Singular S matrix")
            return

        K = self.P @ H.T @ S_inv

        # Error state correction
        dx = K @ z_res

        # Update covariance (Joseph form)
        I = np.eye(15)
        IKH = I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R_gravity_xy @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        # Apply XY accel bias correction only
        bias_correction = dx[9:11]  # Only ba_x, ba_y

        # Limit bias correction magnitude
        MAX_BIAS_CORR = 0.05  # Max 0.05 m/s² per update
        bias_corr_norm = np.linalg.norm(bias_correction)
        if bias_corr_norm > MAX_BIAS_CORR:
            bias_correction = bias_correction * (MAX_BIAS_CORR / bias_corr_norm)

        self.x[9:11] += bias_correction

        # Log significant corrections
        if bias_corr_norm > 0.01:
            self.get_logger().info(
                f"ZUPT Bias: XY accel bias corrected by [{bias_correction[0]:.4f}, {bias_correction[1]:.4f}] m/s²",
                throttle_duration_sec=2.0
            )

    def predict_synthetic(self, dt, fixed_vx, fixed_omega_z):
        """
        Synthetic "Frozen" Prediction for Measurement-Only Testing.

        When enable_prediction=false, we can't just skip prediction entirely because:
        1. Prediction adds uncertainty (Q), Measurement removes uncertainty (R)
        2. Without prediction, covariance P shrinks to zero ("infinite confidence")
        3. Filter stops listening to new measurements ("falls asleep")

        This function propagates the state using a "frozen" constant velocity model:
        - Ignores actual IMU measurements (no sensor noise)
        - Uses configurable fixed velocity and yaw rate
        - Maintains healthy covariance with synthetic process noise

        This allows testing the measurement update in isolation.

        Args:
            dt: Time step
            fixed_vx: Frozen forward velocity in body frame (m/s)
            fixed_omega_z: Frozen yaw rate (rad/s)
        """
        # ===================================================================
        # STEP 1: Nominal State Propagation (Perfect Frozen Model)
        # ===================================================================

        # Current state
        p = self.x[0:3]
        q = self.x[6:10]

        # Frozen velocity and angular velocity
        v_b = np.array([fixed_vx, 0.0, 0.0])  # Forward only
        w_b = np.array([0.0, 0.0, fixed_omega_z])  # Yaw only

        # Rotation matrix (body to world)
        rot = R.from_quat([q[1], q[2], q[3], q[0]])  # scipy uses [x, y, z, w]
        R_wb = rot.as_matrix()

        # Position update: p_new = p + R_wb @ v_b * dt
        self.x[0:3] = p + (R_wb @ v_b) * dt

        # Velocity update: Force body velocity to frozen value
        self.x[3:6] = v_b

        # Orientation update: integrate angular velocity
        if abs(fixed_omega_z) > 1e-6:
            # Quaternion integration for yaw rotation
            angle = fixed_omega_z * dt
            dq = R.from_rotvec([0, 0, angle])
            new_rot = rot * dq
            q_new = new_rot.as_quat()  # [x, y, z, w]
            self.x[6:10] = np.array([q_new[3], q_new[0], q_new[1], q_new[2]])

        # Biases: Freeze (no change)
        # self.x[10:16] unchanged

        # ===================================================================
        # STEP 2: Error Covariance Propagation (Frozen Jacobian)
        # ===================================================================
        # Build Jacobian F assuming frozen conditions:
        # - w = [0, 0, omega_z]
        # - a = 0 (no acceleration, constant velocity)
        # - v_b = [fixed_vx, 0, 0]

        F = np.zeros((15, 15))

        # Position-Velocity coupling: d(δp)/d(δv) = R_wb * dt
        F[0:3, 3:6] = R_wb * dt

        # Position-Orientation coupling: d(δp)/d(δθ) = -[R_wb @ v_b]_× * dt
        # This is the "frozen speed" part crucial for observability
        v_world = R_wb @ v_b
        F[0:3, 6:9] = -skew_symmetric(v_world) * dt

        # Orientation-Gyro bias coupling (minimal since we freeze w)
        # F[6:9, 12:15] = -R_wb * dt  # Commented: we're using frozen w, not measured

        # State transition matrix
        Fx = np.eye(15) + F

        # ===================================================================
        # STEP 3: Synthetic Process Noise
        # ===================================================================
        # We must add SOME noise or P collapses to zero.
        # Use small Q since the model is "perfect" (no IMU noise).

        Q_synthetic = np.zeros((15, 15))

        # Position: small uncertainty from velocity model
        Q_synthetic[0:3, 0:3] = np.eye(3) * (0.01 * dt) ** 2  # 1 cm/s position noise

        # Velocity: small uncertainty (frozen but not perfect)
        Q_synthetic[3:6, 3:6] = np.eye(3) * (0.05 * dt) ** 2  # 5 cm/s velocity noise

        # Orientation: small uncertainty from yaw rate model
        Q_synthetic[6:9, 6:9] = np.eye(3) * (0.001 * dt) ** 2  # ~0.06 deg/s orientation noise

        # Biases: very small random walk (essentially frozen)
        Q_synthetic[9:12, 9:12] = np.eye(3) * (1e-5 * dt)   # Accel bias
        Q_synthetic[12:15, 12:15] = np.eye(3) * (1e-6 * dt)  # Gyro bias

        # ===================================================================
        # STEP 4: Covariance Update
        # ===================================================================
        self.P = Fx @ self.P @ Fx.T + Q_synthetic
        self.P = 0.5 * (self.P + self.P.T)  # Ensure symmetry

        # Store state in buffer for time delay compensation
        if self.last_imu_time is not None:
            self.state_buffer.append({
                'time': self.last_imu_time,
                'x': self.x.copy(),
                'P': self.P.copy()
            })
            if len(self.state_buffer) > self.state_buffer_size:
                self.state_buffer.pop(0)

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

        # FIX: Lock ba_z to zero. For a ground robot with v_z = 0 constraint,
        # ba_z is unobservable and perfectly coupled with gravity.
        # If we don't lock it, the filter pushes ba_z to absorb model errors.
        self.x[12] = 0.0  # ba_z locked to zero

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
                if self.enable_zupt:
                    self._zupt_velocity_update()

                    # --- ZUPT Part 2: Formal Gravity Measurement Update (per recommend.md Section 1) ---
                    # Treat accelerometer as a TILT SENSOR during stationary periods.
                    # This corrects orientation drift with proper covariance update.
                    self._zupt_gravity_update(a_corrected)

        # ===================================================================
        # STEP 1: Nominal State Propagation (Non-linear Kinematics)
        # PLANAR ROBOCENTRIC: Only XY position/velocity, yaw-only orientation
        # ===================================================================

        # For planar robot, we simplify to 2D + yaw:
        # - Position: only X, Y update (Z = 0)
        # - Velocity: only X, Y in body frame (Z = 0)
        # - Orientation: only yaw (rotation around Z), roll/pitch = 0

        # Gravity in body frame: g_b = R_wb^T @ g_w
        # For planar with roll=pitch=0, g_body ≈ [0, 0, g]
        g_body = R_wb.T @ self.g

        # Body-frame acceleration (planar: only use XY components)
        # acc_body = a_corrected - g_body - ω × v_b
        # For planar robot: only consider yaw rotation (w_z) for Coriolis
        w_planar = np.array([0.0, 0.0, w_corrected[2]])  # Only yaw rate
        coriolis = np.cross(w_planar, v_b)
        acc_body = a_corrected - g_body - coriolis

        # --- PLANAR CONSTRAINT: Zero out Z acceleration ---
        # Ground robot cannot accelerate in Z direction
        acc_body[2] = 0.0

        # Soft limit for extreme cases only (e.g., sensor glitches)
        MAX_ACCEL = 10.0  # m/s² - allow larger transients, only clip glitches
        acc_body_clipped = np.clip(acc_body, -MAX_ACCEL, MAX_ACCEL)
        acc_body_clipped[2] = 0.0  # Ensure Z stays zero after clipping

        # Debug logging (every ~2 seconds at 200Hz)
        if not hasattr(self, '_debug_counter'):
            self._debug_counter = 0
        self._debug_counter += 1
        if self._debug_counter % 400 == 1:
            self.get_logger().info(f"Body Accel: [{acc_body[0]:.3f}, {acc_body[1]:.3f}, {acc_body[2]:.3f}] m/s^2")

        # --- Velocity Update (Body Frame) - PLANAR ---
        # v_b_new = v_b + acc_body * dt (only XY)
        v_b_new = v_b.copy()
        v_b_new[0:2] = v_b[0:2] + acc_body_clipped[0:2] * dt  # Only update XY
        v_b_new[2] = 0.0  # Z velocity always zero for planar

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

        # --- Position Update (World Frame) - PLANAR ---
        # p_w_new = p_w + R_wb @ v_b * dt (only XY)
        # For planar, use yaw-only rotation for position update
        yaw = R.from_quat([q[1], q[2], q[3], q[0]]).as_euler('zyx')[0]
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        # v_world_xy = R_z(yaw) @ v_b_xy
        v_world_x = cos_yaw * v_b[0] - sin_yaw * v_b[1]
        v_world_y = sin_yaw * v_b[0] + cos_yaw * v_b[1]

        p_new = p.copy()
        p_new[0] = p[0] + v_world_x * dt  # X position
        p_new[1] = p[1] + v_world_y * dt  # Y position
        p_new[2] = 0.0  # Z position always zero

        # --- Orientation Update (YAW ONLY - Planar) ---
        # For planar robot, only integrate yaw (rotation around Z axis)
        # Ignore roll/pitch gyro components - robot stays level
        w_z = w_corrected[2]  # Only yaw rate

        # Log gyro activity for debugging rotation issues
        if not hasattr(self, '_gyro_debug_counter'):
            self._gyro_debug_counter = 0
        self._gyro_debug_counter += 1
        if self._gyro_debug_counter % 400 == 1 and abs(w_z) > 0.05:
            self.get_logger().info(
                f"Gyro (planar): wz={w_z:.3f} rad/s (ignored: wx={w_corrected[0]:.3f}, wy={w_corrected[1]:.3f}) | "
                f"bias_z: {bg[2]:.4f}"
            )

        # Get current yaw from quaternion
        current_yaw = R.from_quat([q[1], q[2], q[3], q[0]]).as_euler('zyx')[0]

        # Integrate yaw only: yaw_new = yaw + w_z * dt
        yaw_new = current_yaw + w_z * dt

        # Wrap yaw to [-π, π]
        yaw_new = np.arctan2(np.sin(yaw_new), np.cos(yaw_new))

        # Construct quaternion from yaw only (roll=0, pitch=0)
        # q = [cos(yaw/2), 0, 0, sin(yaw/2)] for rotation around Z
        half_yaw = yaw_new / 2.0
        q_new = np.array([np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)])  # [w, x, y, z]

        # Normalize for safety
        q_new = q_new / np.linalg.norm(q_new)

        # --- Bias Update (Random Walk Model) ---
        # Biases are constant in the prediction step (drift added via process noise)
        # No clipping - let EKF covariance naturally bound bias estimates
        ba_new = ba.copy()
        bg_new = bg.copy()

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

        # Construct continuous-time Jacobian F for PLANAR ROBOCENTRIC formulation
        # Only XY position, XY velocity, and yaw dynamics are active
        F = np.zeros((15, 15))

        # Yaw-only rotation matrix (2D rotation in XY plane)
        R_yaw = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw,  cos_yaw, 0],
            [0,        0,       1]
        ])

        # --- Position error dynamics (world frame, XY only) ---
        # p_w = ∫ R_yaw @ v_b dt
        # δṗ = R_yaw @ δv_b + [R_yaw @ v_b]_× @ δθ
        # d(δp)/d(δv_b): Position depends on body velocity through yaw rotation
        F[0:2, 3:5] = R_yaw[0:2, 0:2]  # Only XY block
        # d(δp)/d(δθ_z): Only yaw error affects position (via velocity rotation)
        # [R_yaw @ v_b]_× @ [0,0,δθz]^T = [-vy_world, vx_world, 0]^T * δθz
        v_world_vec = R_yaw @ v_b
        F[0, 8] = -v_world_vec[1]  # d(px)/d(θz) = -vy_world
        F[1, 8] =  v_world_vec[0]  # d(py)/d(θz) =  vx_world

        # --- Velocity error dynamics (body frame, XY only) ---
        # For planar: use yaw-only Coriolis term
        # d(δv_b)/d(δv_b): Coriolis coupling from yaw rate only
        # -[0,0,wz]_× = [[0, wz, 0], [-wz, 0, 0], [0, 0, 0]]
        F[3, 4] =  w_z   # d(vx)/d(vy) = wz (Coriolis)
        F[4, 3] = -w_z   # d(vy)/d(vx) = -wz

        # d(δv_b)/d(δθ_z): How yaw error affects acceleration
        # Simplified for planar - yaw error rotates gravity/accel contribution
        # d(δv_b)/d(δba): Direct effect of accel bias on body acceleration (XY only)
        F[3:5, 9:11] = -np.eye(2)  # Only XY accel bias affects XY velocity
        # d(δv_b)/d(δbg_z): Gyro bias affects Coriolis term
        # [v_b]_× @ [0,0,δbg_z]^T = [-vy, vx, 0]^T * δbg_z
        F[3, 14] = -v_b[1]  # d(vx)/d(bg_z) = -vy
        F[4, 14] =  v_b[0]  # d(vy)/d(bg_z) = vx

        # --- Orientation error dynamics (yaw only) ---
        # δθz_dot = -δbg_z (simplified for planar)
        # d(δθz)/d(δbg_z): Effect of gyro bias error on yaw
        F[8, 14] = -1.0  # Only yaw-gyrobias coupling

        # --- Bias error dynamics ---
        # δḃa = 0, δḃg = 0 (Random walk - noise added separately)
        # F[9:12, 9:12] = 0 (already zero)
        # F[12:15, 12:15] = 0 (already zero)

        # Discrete-time state transition: Fx = I + F*dt
        Fx = np.eye(15) + F * dt

        # ===================================================================
        # STEP 3: Process Noise Covariance Q (Van Loan Method)
        # ===================================================================
        # The process noise captures uncertainty from IMU sensor noise and bias drift.
        # Using Van Loan method: Q = Fi @ Qc @ Fi.T
        # where Fi maps noise sources to error states, and Qc is continuous noise.
        #
        # Reference: es_ekf.py and MatthewHampsey/mekf kalman2.py

        # ===================================================================
        # ADAPTIVE NOISE SCALING
        # ===================================================================
        # Scale process noise based on motion intensity:
        # - High acceleration/rotation → increase Q (less confidence in prediction)
        # - Low motion → decrease Q (more confidence in prediction)

        # Compute motion intensity metrics
        accel_intensity = np.linalg.norm(acc_body_clipped[0:2])  # XY acceleration magnitude
        gyro_intensity = abs(w_z)  # Yaw rate magnitude

        # Adaptive scaling factors (1.0 = nominal, higher = more noise)
        accel_scale = 1.0 + 2.0 * min(accel_intensity / 2.0, 2.0)  # Range: 1.0 to 5.0
        gyro_scale = 1.0 + 2.0 * min(gyro_intensity / 1.0, 2.0)    # Range: 1.0 to 5.0

        # Apply scaling to base noise parameters (continuous-time variances)
        sigma_a2 = self.Q_a * accel_scale   # Accel noise variance
        sigma_g2 = self.Q_g * gyro_scale    # Gyro noise variance
        sigma_ba2 = self.Q_ba * dt          # Accel bias random walk (discrete)
        sigma_bg2 = self.Q_bg * dt          # Gyro bias random walk (discrete)

        # ===================================================================
        # Van Loan Noise Injection: Q = Fi @ Qc @ Fi.T
        # ===================================================================
        # Fi (15x12): Maps noise sources [n_a(3), n_g(3), n_ba(3), n_bg(3)] to error states
        # For PLANAR robot: only XY accel, yaw gyro, XY accel bias, Z gyro bias active

        Fi = np.zeros((15, 12), dtype=float)

        # Accel noise → position and velocity (via rotation)
        # δp += 0.5 * R_yaw * n_a * dt^2
        # δv += R_yaw * n_a * dt (but we're in body frame, so just I*dt for body velocity)
        Fi[0:2, 0:2] = 0.5 * R_yaw[0:2, 0:2] * (dt ** 2)  # n_a → δp (XY only)
        Fi[3:5, 0:2] = np.eye(2) * dt                      # n_a → δv_b (XY only, body frame)

        # Gyro noise → orientation (yaw only)
        # δθz += n_gz * dt
        Fi[8, 5] = dt  # n_gz → δθz (index 5 in noise vector is gz)

        # Bias random walk (already discrete: sigma^2 * dt baked in)
        Fi[9:11, 6:8] = np.eye(2)    # n_ba_xy → δba_xy
        Fi[14, 11] = 1.0             # n_bg_z → δbg_z

        # Continuous noise covariance Qc (12x12)
        # Order: [n_ax, n_ay, n_az, n_gx, n_gy, n_gz, n_ba_x, n_ba_y, n_ba_z, n_bg_x, n_bg_y, n_bg_z]
        Qc = np.diag([
            sigma_a2, sigma_a2, 1e-12,     # Accel noise (XY active, Z tiny)
            1e-12, 1e-12, sigma_g2,        # Gyro noise (Z active for yaw)
            sigma_ba2, sigma_ba2, 1e-12,   # Accel bias RW (XY active, Z locked)
            1e-12, 1e-12, sigma_bg2        # Gyro bias RW (Z active for yaw)
        ]).astype(float)

        # Process noise via Van Loan method
        Qi = Fi @ Qc @ Fi.T

        # Ensure tiny noise on constrained states for numerical stability
        for idx in [2, 5, 6, 7, 11, 12, 13]:  # pz, vz, roll, pitch, ba_z, bg_x, bg_y
            Qi[idx, idx] = max(Qi[idx, idx], 1e-12)

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

        # ===================================================================
        # PLANAR CONSTRAINT ENFORCEMENT ON COVARIANCE
        # ===================================================================
        # For planar robot, constrained states are:
        #   - pz (index 2), vz (index 5)
        #   - roll (index 6), pitch (index 7)
        #   - ba_z (index 11), bg_x (index 12), bg_y (index 13)
        # We damp cross-correlations and keep diagonal small but non-zero

        constrained_indices = [2, 5, 6, 7, 11, 12, 13]
        z_damping = 0.1  # Damp constrained state covariances each step

        for idx in constrained_indices:
            # Damp off-diagonal elements (cross-correlations)
            for j in range(15):
                if j != idx:
                    self.P[idx, j] *= z_damping
                    self.P[j, idx] *= z_damping
            # Keep diagonal small but non-zero for numerical stability
            self.P[idx, idx] = max(self.P[idx, idx] * z_damping, 1e-6)

        # Ensure symmetry again after constraint enforcement
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
        Handle visual landmark observations using RANGE-BEARING model.

        Instead of pixel coordinates, we compute range (distance) and bearing (angle)
        to each detected ArUco marker and perform sequential EKF updates.

        The ArUco detector provides:
          - msg.poses[i].position.x = pixel u coordinate
          - msg.poses[i].position.y = pixel v coordinate
          - msg.poses[i].position.z = landmark ID

        We convert pixel coordinates to range-bearing using camera geometry.
        """
        # Skip if correction is disabled
        if not self.enable_correction:
            return

        # Skip if not initialized
        if not self.initialized:
            return

        # Time synchronization check
        if self.last_imu_time is None:
            self.get_logger().warn("Vision update skipped: No IMU data received yet", throttle_duration_sec=2.0)
            return

        # Process each detected landmark
        update_count = 0
        for obs in msg.poses:
            lid = obs.position.z
            u_meas = obs.position.x  # Pixel u
            v_meas = obs.position.y  # Pixel v

            if lid not in self.map:
                continue

            lm_world = self.map[lid]

            # Compute range and bearing from pixel measurement
            # Use camera intrinsics to get normalized image coordinates
            # Then compute bearing from camera geometry
            fx = self.K[0, 0]
            fy = self.K[1, 1]
            cx = self.K[0, 2]
            cy = self.K[1, 2]

            # Normalized image coordinates (ray direction in camera frame)
            x_norm = (u_meas - cx) / fx
            y_norm = (v_meas - cy) / fy

            # Bearing in camera frame: angle from optical axis (Z) to ray in XZ plane
            # Camera frame: Z = forward (depth), X = right, Y = down
            # Bearing = atan2(X, Z) where Z = 1 for normalized coords
            bearing_camera = np.arctan2(x_norm, 1.0)

            # To get range, we need the actual 3D position from the known map
            # Transform landmark to body frame to compute true range
            p_w = self.x[0:3]
            q_w = self.x[6:10]  # [w, x, y, z]
            R_wb = R.from_quat([q_w[1], q_w[2], q_w[3], q_w[0]]).as_matrix()

            # Landmark in body frame
            lm_body = R_wb.T @ (lm_world - p_w)

            # Transform to camera frame (account for camera extrinsics)
            lm_cam = self.R_b_c @ (lm_body - self.t_b_c)

            # Check if landmark is in front of camera
            if lm_cam[2] < 0.1:
                continue

            # Predicted range and bearing
            # For planar: use XY distance (ignore Z)
            pred_range = np.linalg.norm(lm_body[0:2])
            if pred_range < 0.3:
                continue  # Too close

            # Bearing in body frame (angle from X axis to landmark in XY plane)
            pred_bearing = np.arctan2(lm_body[1], lm_body[0])

            # Measured bearing: transform from camera frame to body frame
            # Camera X = -Body Y, Camera Z = Body X
            # So bearing_body = atan2(-x_norm, 1) in body coords
            # Simplification: bearing in body = -bearing in camera (for forward-facing camera)
            meas_bearing = -bearing_camera

            # For range measurement, use the known landmark position and bearing
            # to triangulate. Since we have the map, we use predicted range
            # but with added measurement noise based on pixel uncertainty.
            #
            # Alternative: Use depth from stereo or known marker size
            # For now, we trust the predicted range (from map) but the bearing
            # comes from the actual pixel measurement
            meas_range = pred_range  # Use map distance (could be from marker size)

            # Perform range-bearing update
            self.range_bearing_update(lm_world, meas_range, meas_bearing, lid)
            update_count += 1

        if update_count > 0:
            self.get_logger().info(
                f"Vision: {update_count} landmarks updated",
                throttle_duration_sec=0.5
            )

    def range_bearing_update(self, lm_world, meas_range, meas_bearing, lm_id):
        """
        EKF measurement update using range-bearing model.

        Measurement model:
          z = [range, bearing]
          range = ||lm_body[0:2]||  (planar distance)
          bearing = atan2(lm_body[1], lm_body[0])  (angle in body XY plane)

        where lm_body = R_wb^T @ (lm_world - p_w)

        Args:
            lm_world: 3D landmark position in world frame
            meas_range: Measured range to landmark (m)
            meas_bearing: Measured bearing to landmark (rad)
            lm_id: Landmark ID for logging
        """
        # Current state
        p_w = self.x[0:3]
        q_w = self.x[6:10]  # [w, x, y, z]
        R_wb = R.from_quat([q_w[1], q_w[2], q_w[3], q_w[0]]).as_matrix()

        # Transform landmark to body frame
        lm_body = R_wb.T @ (lm_world - p_w)

        # Predicted range and bearing (planar)
        pred_range = np.linalg.norm(lm_body[0:2])
        if pred_range < 0.3:
            return  # Too close, skip

        pred_bearing = np.arctan2(lm_body[1], lm_body[0])

        # Measurement residual
        range_res = meas_range - pred_range
        bearing_res = meas_bearing - pred_bearing
        # Wrap bearing residual to [-π, π]
        bearing_res = np.arctan2(np.sin(bearing_res), np.cos(bearing_res))

        y = np.array([range_res, bearing_res])

        # ===================================================================
        # Jacobian H (2x15) for range-bearing measurement
        # ===================================================================
        # Measurement is function of lm_body = R_wb^T @ (lm_world - p_w)
        #
        # d(lm_body)/d(p_w) = -R_wb^T
        # d(lm_body)/d(θ) = [lm_body]× (skew of landmark in body frame)
        #
        # For range r = ||lm_body[0:2]||:
        #   dr/d(lm_body) = [lm_body[0]/r, lm_body[1]/r, 0]
        #
        # For bearing b = atan2(lm_body[1], lm_body[0]):
        #   db/d(lm_body) = [-lm_body[1]/r², lm_body[0]/r², 0]

        r = pred_range
        lx, ly = lm_body[0], lm_body[1]

        # Jacobian of measurements w.r.t. lm_body (in 3D)
        dr_dlm = np.array([lx / r, ly / r, 0.0])
        db_dlm = np.array([-ly / (r**2), lx / (r**2), 0.0])

        # Jacobian w.r.t. position: d(lm_body)/d(p_w) = -R_wb^T
        H_pos_range = dr_dlm @ (-R_wb.T)
        H_pos_bearing = db_dlm @ (-R_wb.T)

        # Jacobian w.r.t. orientation: d(lm_body)/d(θ) = [lm_body]×
        # Using right-multiplicative error convention
        lm_body_skew = skew_symmetric(lm_body)
        H_ori_range = dr_dlm @ lm_body_skew
        H_ori_bearing = db_dlm @ lm_body_skew

        # Assemble H (2x15)
        # Error state: [δp(3), δv(3), δθ(3), δba(3), δbg(3)]
        H = np.zeros((2, 15))
        H[0, 0:3] = H_pos_range      # Range w.r.t. position
        H[0, 6:9] = H_ori_range      # Range w.r.t. orientation
        H[1, 0:3] = H_pos_bearing    # Bearing w.r.t. position
        H[1, 6:9] = H_ori_bearing    # Bearing w.r.t. orientation

        # ===================================================================
        # Kalman Update
        # ===================================================================
        # Measurement noise
        Rm = np.diag([self.R_range, self.R_bearing])

        # Innovation covariance
        S = H @ self.P @ H.T + Rm

        # Kalman gain
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            self.get_logger().warn("Singular S matrix, skipping update")
            return

        K = self.P @ H.T @ S_inv

        # Error state correction
        dx = K @ y

        # Update covariance (Joseph form for numerical stability)
        I = np.eye(15)
        IKH = I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ Rm @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        # ===================================================================
        # Inject error into nominal state
        # ===================================================================

        # Position correction (with limit for safety)
        pos_correction = dx[0:3]
        pos_corr_norm = np.linalg.norm(pos_correction)
        if pos_corr_norm > 0.5:
            pos_correction = pos_correction * (0.5 / pos_corr_norm)
        self.x[0:3] += pos_correction

        # Velocity correction (with limit)
        vel_correction = dx[3:6]
        vel_corr_norm = np.linalg.norm(vel_correction)
        if vel_corr_norm > 0.3:
            vel_correction = vel_correction * (0.3 / vel_corr_norm)
        self.x[3:6] += vel_correction

        # Orientation correction - apply as quaternion update
        ori_correction = dx[6:9]
        ori_corr_norm = np.linalg.norm(ori_correction)
        if ori_corr_norm > 0.1:  # Limit to ~6 degrees
            ori_correction = ori_correction * (0.1 / ori_corr_norm)
        if ori_corr_norm > 1e-8:
            q = self.x[6:10]
            delta_q = R.from_rotvec(ori_correction)
            q_updated = (R.from_quat([q[1], q[2], q[3], q[0]]) * delta_q).as_quat()
            self.x[6:10] = np.array([q_updated[3], q_updated[0], q_updated[1], q_updated[2]])

        # Bias corrections
        self.x[10:13] += dx[9:12]   # Accel bias
        self.x[13:16] += dx[12:15]  # Gyro bias

        # Keep ba_z locked (unobservable for planar)
        self.x[12] = 0.0

        # ===================================================================
        # G·P·G^T COVARIANCE RESET AFTER ERROR INJECTION
        # ===================================================================
        G = np.eye(15)
        if ori_corr_norm > 1e-8:
            G[6:9, 6:9] = np.eye(3) - 0.5 * skew_symmetric(ori_correction)
        self.P = G @ self.P @ G.T
        self.P = 0.5 * (self.P + self.P.T)

        # ===================================================================
        # PLANAR CONSTRAINTS
        # ===================================================================
        self.x[2] = 0.0   # Z position = 0
        self.x[5] = 0.0   # Z velocity = 0
        self.x[12] = 0.0  # ba_z = 0

        # Zero constrained state covariances
        for idx in [2, 5, 11]:  # pz, vz, ba_z
            self.P[idx, :] = 0.0
            self.P[:, idx] = 0.0
            self.P[idx, idx] = 1e-6

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

        # Publish Diagnostics for tuning/debugging
        diag = Float64MultiArray()
        P_diag = np.diag(self.P)
        speed = np.linalg.norm(self.x[3:5])
        diag.data = [
            # Biases (6 values)
            float(self.x[10]), float(self.x[11]), float(self.x[12]),  # accel bias
            float(self.x[13]), float(self.x[14]), float(self.x[15]),  # gyro bias
            # Covariance diagonal - position (3 values)
            float(P_diag[0]), float(P_diag[1]), float(P_diag[2]),
            # Covariance diagonal - velocity (3 values)
            float(P_diag[3]), float(P_diag[4]), float(P_diag[5]),
            # Covariance diagonal - orientation (3 values)
            float(P_diag[6]), float(P_diag[7]), float(P_diag[8]),
            # Covariance diagonal - biases (6 values)
            float(P_diag[9]), float(P_diag[10]), float(P_diag[11]),
            float(P_diag[12]), float(P_diag[13]), float(P_diag[14]),
            # Velocity + motion indicator (4 values)
            float(self.x[3]), float(self.x[4]), float(self.x[5]),
            float(speed), float(self.last_vision_correction)
        ]
        self.pub_diag.publish(diag)

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
