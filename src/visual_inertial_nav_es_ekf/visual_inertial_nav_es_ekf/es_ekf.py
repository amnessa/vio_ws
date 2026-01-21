"""
ES-EKF with Range + Bearing Visual Measurements + ZUPT
FIXED: Jacobian signs, Covariance Tuning, and Zero-Velocity Updates
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import Path, Odometry
import numpy as np
from scipy.spatial.transform import Rotation as R
class ES_EKF(Node):
    def __init__(self):
        super().__init__('es_ekf_node')

        # --- Initialization ---
        self.is_initialized = False
        self.init_count = 0
        self.init_sum_a = np.zeros(3)
        self.g_body = None  # To be calibrated

        # Camera offset
        self.cam_offset = np.array([1.0, 0.0, 0.0])

        # Map Landmarks
        self.landmarks = {
            1: np.array([6.0, 6.0, 0.5]),   # Red
            2: np.array([6.0, -6.0, 0.5]),  # Green
            3: np.array([10.0, 0.0, 0.5]),  # Blue
            4: np.array([0.0, 8.0, 0.5])    # Yellow
        }

        # --- State Vector ---
        self.p = np.array([0.0, 0.0, 0.4])
        self.v = np.zeros(3)
        self.q = np.array([0., 0., 0., 1.])
        self.ab = np.zeros(3)
        self.wb = np.zeros(3)
        self.fixed_z = 0.4

        # --- Covariance P ---
        self.P = np.diag([
            0.01, 0.01, 0.0001,    # Pos
            0.01, 0.01, 0.0001,    # Vel
            0.001, 0.001, 0.01,    # Angle
            1e-4, 1e-4, 1e-4,      # Accel Bias
            1e-5, 1e-5, 1e-5       # Gyro Bias
        ])

        # --- Process Noise ---
        self.sigma_a = 0.05      # Accel noise
        self.sigma_w = 0.005     # Gyro noise
        self.sigma_ab = 1e-5
        self.sigma_wb = 1e-6

        # --- Measurement Noise ---
        self.R_bearing = 0.05**2
        self.R_range = 0.1**2
        self.last_imu_time = None
        self.imu_count = 0
        self.vis_count = 0

        # --- ROS ---
        self.sub_imu = self.create_subscription(
            Imu, '/vehicle_blue/imu', self.imu_callback, 10)
        self.sub_vis = self.create_subscription(
            PointStamped, '/visual_measurement', self.visual_callback, 10)
        self.pub_odom = self.create_publisher(Odometry, '/ekf_odom', 10)
        self.pub_path = self.create_publisher(Path, '/ekf_path', 10)
        self.path_msg = Path()
        self.path_msg.header.frame_id = 'world'

        self.create_timer(1.0, self.debug_callback)
        self.get_logger().info("ES-EKF (ZUPT Enabled) Started")
    def debug_callback(self):
        if self.is_initialized:
            rpy = R.from_quat(self.q).as_euler('xyz', degrees=True)
            self.get_logger().info(
                f"p=[{self.p[0]:.2f}, {self.p[1]:.2f}] "
                f"v=[{self.v[0]:.2f}, {self.v[1]:.2f}] "
                f"yaw={rpy[2]:.1f} P_tr={np.trace(self.P):.3f} VIS={self.vis_count}"
            )
    def _enforce_2d_constraints(self):
        self.p[2] = self.fixed_z
        self.v[2] = 0.0
        rpy = R.from_quat(self.q).as_euler('xyz')
        self.q = R.from_euler('xyz', [0.0, 0.0, rpy[2]]).as_quat()
    def imu_callback(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if self.last_imu_time is None:
            self.last_imu_time = t
            return
        dt = t - self.last_imu_time
        self.last_imu_time = t
        if dt <= 0 or dt > 0.1: return
        a_imu = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
        w_imu = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        # Calibration
        if not self.is_initialized:
            self.init_sum_a += a_imu
            self.init_count += 1
            if self.init_count >= 50:
                self.g_body = self.init_sum_a / 50.0
                self.is_initialized = True
                self.get_logger().info(f"Calibrated g_body: {self.g_body}")
            return
        self.imu_count += 1
        # --- PREDICTION ---
        rot = R.from_quat(self.q).as_matrix()

        a_corrected = a_imu - self.ab
        w_corrected = w_imu - self.wb
        # --- ZUPT LOGIC (Zero Velocity Update) ---
        # 1. Calculate magnitude of dynamic acceleration (removing gravity)
        a_dynamic = a_corrected - self.g_body
        acc_mag = np.linalg.norm(a_dynamic)
        gyro_mag = np.linalg.norm(w_corrected)
        # 2. Check Stationary Condition
        # Thresholds: Accel < 0.05 m/s^2 AND Gyro < 0.01 rad/s
        is_stationary = (acc_mag < 0.05) and (gyro_mag < 0.01)
        if is_stationary:
            # Force velocity to zero to kill drift
            self.v = np.zeros(3)
            a_lin_body = np.zeros(3) # Assume no acceleration
            w_corrected = np.zeros(3) # Assume no rotation
        else:
            # Use calculated acceleration
            a_lin_body = a_corrected - self.g_body
        # Rotate to world frame
        acc_world = rot @ a_lin_body
        self.p = self.p + self.v * dt + 0.5 * acc_world * dt**2

        # Only update velocity if moving (redundant but safe)
        if not is_stationary:
            self.v = self.v + acc_world * dt
        # Yaw-only integration
        d_angle = np.array([0.0, 0.0, w_corrected[2]]) * dt
        if np.linalg.norm(d_angle) > 1e-10:
            dq = R.from_rotvec(d_angle)
            self.q = (R.from_quat(self.q) * dq).as_quat()
            self.q = self.q / np.linalg.norm(self.q)
        self._enforce_2d_constraints()
        # Covariance
        Fx = np.eye(15)
        Fx[0:3, 3:6] = np.eye(3) * dt
        a_skew = self._skew(a_corrected)
        Fx[3:6, 6:9] = -rot @ a_skew * dt
        Fx[3:6, 9:12] = -rot * dt
        Fx[6:9, 12:15] = -np.eye(3) * dt
        # Reduce noise injection if stationary to prevent covariance explosion
        noise_scale = 0.01 if is_stationary else 1.0

        Qi = np.diag([self.sigma_a]*3 + [self.sigma_w]*3 + [self.sigma_ab]*3 + [self.sigma_wb]*3)**2 * dt * noise_scale

        Fi = np.zeros((15, 12))
        Fi[3:6, 0:3] = rot
        Fi[6:9, 3:6] = np.eye(3)
        Fi[9:12, 6:9] = np.eye(3)
        Fi[12:15, 9:12] = np.eye(3)
        self.P = Fx @ self.P @ Fx.T + Fi @ Qi @ Fi.T
        self.P = 0.5 * (self.P + self.P.T)
        self.publish_state(msg.header.stamp)
    def visual_callback(self, msg):
        if not self.is_initialized: return
        lm_id = int(msg.point.z)
        if lm_id not in self.landmarks: return
        z_bearing = msg.point.x
        z_range = msg.point.y
        lm_world = self.landmarks[lm_id]
        rot = R.from_quat(self.q).as_matrix()
        rot_T = rot.T

        lm_body = rot_T @ (lm_world - self.p)
        lm_cam = lm_body - self.cam_offset

        pred_range = np.linalg.norm(lm_cam[:2])
        pred_bearing = np.arctan2(lm_cam[1], lm_cam[0])
        if pred_range < 0.5: return
        self.vis_count += 1
        y_range = z_range - pred_range
        y_bearing = z_bearing - pred_bearing
        y_bearing = (y_bearing + np.pi) % (2*np.pi) - np.pi
        dr_dcam = np.array([lm_cam[0]/pred_range, lm_cam[1]/pred_range, 0.0])
        db_dcam = np.array([-lm_cam[1]/(pred_range**2), lm_cam[0]/(pred_range**2), 0.0])
        H = np.zeros((2, 15))
        H[0, 0:3] = dr_dcam @ (-rot_T)
        H[1, 0:3] = db_dcam @ (-rot_T)

        # Standard Jacobian Sign
        H[0, 6:9] = dr_dcam @ self._skew(lm_body)
        H[1, 6:9] = db_dcam @ self._skew(lm_body)
        R_meas = np.diag([self.R_range, self.R_bearing])
        S = H @ self.P @ H.T + R_meas
        K = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ np.array([y_range, y_bearing])
        self.p += dx[0:3]
        self.v += dx[3:6]

        dtheta = dx[6:9]
        if np.linalg.norm(dtheta) > 1e-10:
            dq = R.from_rotvec(dtheta)
            self.q = (R.from_quat(self.q) * dq).as_quat()
            self.q = self.q / np.linalg.norm(self.q)

        self.ab += dx[9:12]
        self.wb += dx[12:15]
        I_KH = np.eye(15) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_meas @ K.T
        self._enforce_2d_constraints()
    def _skew(self, v):
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
    def publish_state(self, stamp):
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = 'world'
        odom.pose.pose.position.x = float(self.p[0])
        odom.pose.pose.position.y = float(self.p[1])
        odom.pose.pose.position.z = float(self.p[2])
        odom.pose.pose.orientation.x = float(self.q[0])
        odom.pose.pose.orientation.y = float(self.q[1])
        odom.pose.pose.orientation.z = float(self.q[2])
        odom.pose.pose.orientation.w = float(self.q[3])
        self.pub_odom.publish(odom)
        pose = PoseStamped()
        pose.header = odom.header
        pose.pose = odom.pose.pose
        self.path_msg.poses.append(pose)
        if len(self.path_msg.poses) > 5000:
            self.path_msg.poses.pop(0)
        self.path_msg.header.stamp = stamp
        self.pub_path.publish(self.path_msg)
def main():
    rclpy.init()
    node = ES_EKF()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
if __name__ == '__main__':
    main()