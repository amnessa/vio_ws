import math
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
import tf2_ros
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Float64

class TrajectoryMonitor(Node):
    def __init__(self):
        super().__init__('trajectory_monitor')

        # --- Parameters ---
        self.declare_parameter('target_frame', 'world')
        self.declare_parameter('gt_frame', 'vehicle_blue/chassis')
        self.declare_parameter('ekf_topic', '/ekf_odom')
        self.declare_parameter('path_max_len', 5000)

        self.target_frame = self.get_parameter('target_frame').value
        self.gt_frame = self.get_parameter('gt_frame').value
        self.path_max_len = int(self.get_parameter('path_max_len').value)

        # --- TF Listener ---
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=20.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # --- Subscribers ---
        self.sub_ekf = self.create_subscription(
            Odometry, 
            self.get_parameter('ekf_topic').value, 
            self.ekf_callback, 
            10
        )

        # --- Publishers ---
        self.pub_gt_path = self.create_publisher(Path, '/ground_truth/path', 10)
        self.pub_error = self.create_publisher(Float64, '/ekf/pose_error', 10)
        self.pub_rmse = self.create_publisher(Float64, '/ekf/rmse', 10)

        # --- State ---
        self.gt_path_msg = Path()
        self.gt_path_msg.header.frame_id = self.target_frame
        
        self.sq_error_sum = 0.0
        self.sample_count = 0

        self.get_logger().info("Trajectory Monitor initialized. Waiting for EKF data...")

    def ekf_callback(self, odom_msg: Odometry):
        # 1. Get EKF Position
        pos_est = odom_msg.pose.pose.position
        
        # 2. Get Ground Truth Position via TF
        try:
            # FIX: We use Time() (which means "0") to get the LATEST available transform.
            # This handles cases where Ground Truth is slower (1Hz) than EKF (100Hz).
            t = self.tf_buffer.lookup_transform(
                self.target_frame, 
                self.gt_frame, 
                Time(), # <--- This gets the latest transform available
                timeout=Duration(seconds=0.1)
            )
        except tf2_ros.TransformException as ex:
            # Log strictly once per second to avoid spamming console
            self.get_logger().warn(f'Waiting for TF connection: {ex}', throttle_duration_sec=1.0)
            return

        # Extract GT Position
        gt_x = t.transform.translation.x
        gt_y = t.transform.translation.y
        gt_z = t.transform.translation.z

        # 3. Calculate Error (Euclidean Distance)
        dx = pos_est.x - gt_x
        dy = pos_est.y - gt_y
        dz = pos_est.z - gt_z
        
        error_sq = dx*dx + dy*dy + dz*dz
        error_dist = math.sqrt(error_sq)

        # 4. Update RMSE
        self.sq_error_sum += error_sq
        self.sample_count += 1
        rmse = math.sqrt(self.sq_error_sum / self.sample_count)

        # 5. Publish Metrics
        err_msg = Float64()
        err_msg.data = error_dist
        self.pub_error.publish(err_msg)

        rmse_msg = Float64()
        rmse_msg.data = rmse
        self.pub_rmse.publish(rmse_msg)

        # 6. Build and Publish GT Path
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = odom_msg.header.stamp # Use EKF time for visualization sync
        pose_stamped.header.frame_id = self.target_frame
        pose_stamped.pose.position.x = gt_x
        pose_stamped.pose.position.y = gt_y
        pose_stamped.pose.position.z = gt_z
        pose_stamped.pose.orientation = t.transform.rotation

        self.gt_path_msg.poses.append(pose_stamped)
        
        if len(self.gt_path_msg.poses) > self.path_max_len:
            self.gt_path_msg.poses.pop(0)
            
        self.gt_path_msg.header.stamp = odom_msg.header.stamp
        self.pub_gt_path.publish(self.gt_path_msg)

def main():
    rclpy.init()
    node = TrajectoryMonitor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()