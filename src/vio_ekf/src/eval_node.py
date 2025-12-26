#!/usr/bin/env python3
"""
Evaluation Node for VIO EKF
Computes RMSE between EKF estimate and ground truth, generates trajectory plots.
"""
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import numpy as np
import message_filters
import os

class EvaluationNode(Node):
    def __init__(self):
        super().__init__('eval_node')

        # Use separate subscribers instead of time sync (more robust)
        self.sub_est = self.create_subscription(Odometry, '/vio/odom', self.est_callback, 10)
        self.sub_gt = self.create_subscription(Odometry, '/ground_truth/odom', self.gt_callback, 10)

        # Store latest messages
        self.latest_gt = None
        self.latest_gt_time = None

        self.errors = []
        self.errors_x = []
        self.errors_y = []
        self.errors_z = []
        self.path_est = {'x': [], 'y': [], 'z': []}
        self.path_gt = {'x': [], 'y': [], 'z': []}
        self.timestamps = []
        self.time_start = None

        self.get_logger().info("=" * 50)
        self.get_logger().info("Evaluation Node Started. Recording Data...")
        self.get_logger().info("Drive the robot around, then press Ctrl+C to save plots.")
        self.get_logger().info("=" * 50)

    def gt_callback(self, msg):
        """Store latest ground truth for matching"""
        self.latest_gt = msg
        self.latest_gt_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def est_callback(self, est_msg):
        """Process EKF estimate and match with closest ground truth"""
        if self.latest_gt is None:
            return  # No ground truth yet

        gt_msg = self.latest_gt

        if self.time_start is None:
            self.time_start = est_msg.header.stamp.sec + est_msg.header.stamp.nanosec * 1e-9

        current_time = est_msg.header.stamp.sec + est_msg.header.stamp.nanosec * 1e-9
        elapsed = current_time - self.time_start

        # Extract Positions
        pos_est = np.array([est_msg.pose.pose.position.x,
                           est_msg.pose.pose.position.y,
                           est_msg.pose.pose.position.z])
        pos_gt = np.array([gt_msg.pose.pose.position.x,
                          gt_msg.pose.pose.position.y,
                          gt_msg.pose.pose.position.z])

        # Calculate Error
        error = np.linalg.norm(pos_est - pos_gt)
        self.errors.append(error)
        self.errors_x.append(abs(pos_est[0] - pos_gt[0]))
        self.errors_y.append(abs(pos_est[1] - pos_gt[1]))
        self.errors_z.append(abs(pos_est[2] - pos_gt[2]))
        self.timestamps.append(elapsed)

        # Store for plotting
        self.path_est['x'].append(pos_est[0])
        self.path_est['y'].append(pos_est[1])
        self.path_est['z'].append(pos_est[2])
        self.path_gt['x'].append(pos_gt[0])
        self.path_gt['y'].append(pos_gt[1])
        self.path_gt['z'].append(pos_gt[2])

        # Log RMSE every 200 samples (~1 second at 200Hz)
        if len(self.errors) % 200 == 0:
            rmse = np.sqrt(np.mean(np.array(self.errors)**2))
            self.get_logger().info(f"Samples: {len(self.errors)} | RMSE: {rmse:.4f} m | Error: {error:.4f} m")

    def save_plots(self):
        if len(self.errors) < 10:
            self.get_logger().warn("Not enough data to generate plots (need at least 10 samples)")
            return

        self.get_logger().info("=" * 50)
        self.get_logger().info("Generating evaluation plots...")

        # Calculate final statistics
        errors_arr = np.array(self.errors)
        rmse = np.sqrt(np.mean(errors_arr**2))
        mean_error = np.mean(errors_arr)
        max_error = np.max(errors_arr)
        min_error = np.min(errors_arr)

        self.get_logger().info(f"Final Statistics over {len(self.errors)} samples:")
        self.get_logger().info(f"  RMSE:      {rmse:.4f} m")
        self.get_logger().info(f"  Mean:      {mean_error:.4f} m")
        self.get_logger().info(f"  Max:       {max_error:.4f} m")
        self.get_logger().info(f"  Min:       {min_error:.4f} m")

        # Output directory
        output_dir = '/workspaces/vio_ws'

        # Plot 1: 2D Trajectory (Top-down view)
        plt.figure(figsize=(10, 8))
        plt.plot(self.path_gt['x'], self.path_gt['y'], 'g-', label='Ground Truth', linewidth=2)
        plt.plot(self.path_est['x'], self.path_est['y'], 'r--', label='EKF Estimate', linewidth=2)
        plt.plot(self.path_gt['x'][0], self.path_gt['y'][0], 'go', markersize=10, label='Start')
        plt.plot(self.path_gt['x'][-1], self.path_gt['y'][-1], 'g^', markersize=10, label='End (GT)')
        plt.title(f"Trajectory Comparison (RMSE: {rmse:.4f} m)")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        trajectory_path = os.path.join(output_dir, 'trajectory_plot.png')
        plt.savefig(trajectory_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Plot 2: Position Error over Time
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(self.timestamps, self.errors, 'b-', linewidth=1)
        plt.axhline(y=rmse, color='r', linestyle='--', label=f'RMSE = {rmse:.4f} m')
        plt.title("Position Error (3D) over Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Error (m)")
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(self.timestamps, self.errors_x, 'r-', label='X error', alpha=0.7)
        plt.plot(self.timestamps, self.errors_y, 'g-', label='Y error', alpha=0.7)
        plt.plot(self.timestamps, self.errors_z, 'b-', label='Z error', alpha=0.7)
        plt.title("Position Error per Axis")
        plt.xlabel("Time (s)")
        plt.ylabel("Error (m)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        error_path = os.path.join(output_dir, 'error_plot.png')
        plt.savefig(error_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Plot 3: X, Y, Z Comparison
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        axes[0].plot(self.timestamps, self.path_gt['x'], 'g-', label='Ground Truth', linewidth=2)
        axes[0].plot(self.timestamps, self.path_est['x'], 'r--', label='EKF Estimate', linewidth=1)
        axes[0].set_ylabel('X (m)')
        axes[0].set_title('Position Comparison')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(self.timestamps, self.path_gt['y'], 'g-', label='Ground Truth', linewidth=2)
        axes[1].plot(self.timestamps, self.path_est['y'], 'r--', label='EKF Estimate', linewidth=1)
        axes[1].set_ylabel('Y (m)')
        axes[1].legend()
        axes[1].grid(True)

        axes[2].plot(self.timestamps, self.path_gt['z'], 'g-', label='Ground Truth', linewidth=2)
        axes[2].plot(self.timestamps, self.path_est['z'], 'r--', label='EKF Estimate', linewidth=1)
        axes[2].set_ylabel('Z (m)')
        axes[2].set_xlabel('Time (s)')
        axes[2].legend()
        axes[2].grid(True)

        plt.tight_layout()
        comparison_path = os.path.join(output_dir, 'position_comparison.png')
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()

        self.get_logger().info(f"Plots saved to:")
        self.get_logger().info(f"  - {trajectory_path}")
        self.get_logger().info(f"  - {error_path}")
        self.get_logger().info(f"  - {comparison_path}")
        self.get_logger().info("=" * 50)

        # Success criteria check
        if rmse < 0.3:
            self.get_logger().info("🎉 SUCCESS! RMSE < 0.3m - Excellent VIO performance!")
        elif rmse < 1.0:
            self.get_logger().info("✓ GOOD! RMSE < 1.0m - Acceptable VIO performance.")
        else:
            self.get_logger().info("⚠ RMSE > 1.0m - Consider tuning the filter parameters.")

def main(args=None):
    rclpy.init(args=args)
    node = EvaluationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_plots()
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()
