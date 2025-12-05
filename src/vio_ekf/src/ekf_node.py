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

        # --- State Definitions ---
        # State: [p(3), v(3), q(4), ba(3), bg(3)] = 16 elements
        # Error State: [dp(3), dv(3), dtheta(3), dba(3), dbg(3)] = 15 elements
        self.x = np.zeros(16)
        self.x[6] = 1.0  # Initial quaternion (w=1, x=0, y=0, z=0)

        # Covariance Matrix (15x15)
        self.P = np.eye(15) * 0.1
        self.P[0:3, 0:3] *= 0.0  # Known initial position
        self.P[6:9, 6:9] *= 0.01 # Initial orientation uncertainty
        self.P[9:12, 9:12] *= 0.01 # Accel bias uncertainty
        self.P[12:15, 12:15] *= 0.001 # Gyro bias uncertainty

        # Noise Parameters
        self.Q_a = 0.01  # Accel noise density
        self.Q_g = 0.005 # Gyro noise density
        self.Q_ba = 0.0001 # Accel bias random walk
        self.Q_bg = 0.00001 # Gyro bias random walk
        self.R_cam = 5.0 # Pixel measurement noise (variance)

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

        self.sub_imu = self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.sub_vision = self.create_subscription(PoseArray, '/vio/landmarks', self.vision_callback, 10)
        self.sub_cam_info = self.create_subscription(CameraInfo, '/camera_info', self.info_callback, 10)

        self.pub_odom = self.create_publisher(Odometry, '/vio/odom', 10)
        self.pub_path = self.create_publisher(Path, '/vio/path', 10)
        self.tf_br = TransformBroadcaster(self)

        self.get_logger().info("EKF Node Initialized")

    def info_callback(self, msg):
        # Update intrinsics matrix K
        self.K = np.array(msg.k).reshape(3,3)

    def imu_callback(self, msg):
        curr_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if self.last_imu_time is None:
            self.last_imu_time = curr_time
            return

        dt = curr_time - self.last_imu_time
        if dt <= 0: return # Skip backwards or duplicate messages
        self.last_imu_time = curr_time

        # Extract measurements
        a_m = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
        w_m = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])

        # --- PREDICTION STEP ---
        self.predict(dt, a_m, w_m)
        self.publish_state(msg.header.stamp)

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
        g = np.array([0, 0, 9.81]) # Gravity in World

        # Position
        p_new = p + v * dt + 0.5 * (R_wb @ a_unbiased - g) * dt**2
        # Velocity
        v_new = v + (R_wb @ a_unbiased - g) * dt
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

    def vision_callback(self, msg):
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

def main(args=None):
    rclpy.init(args=args)
    node = EKFNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()