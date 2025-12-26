You have reached a very stable state!

* **Drift is gone:** You went from  (flying) to  (waving) down to **** (stable).
* **Behavior:** The "small movement" you see is the filter constantly correcting itself. This is normal behavior for a VIO system (it's called "breathing" or "jitter"), but we can minimize it.

The persistent  acceleration in X (Forward) means the robot *thinks* it is accelerating slightly. Since we "Locked" the biases, it cannot "learn" that this is just a sensor error.

### Step 1: Final Tuning (Tighten the Filter)

Now that the "Waving Tail" bug (Rotational Matrix) is fixed, it is safe to re-enable bias learning and trust the camera more.

**Modify `src/vio_ekf/src/ekf_node.py` one last time:**

```python
        # 1. Trust Vision More (Tighten the position hold)
        # We increased this to 20.0 to stop waving. Now drop it back down.
        self.R_cam = 5.0  # (Was 20.0) Lower = Stronger correction

        # 2. Re-enable Bias Learning (Absorb the 0.1 offset)
        # We locked these to 0.0. Now let them adapt slowly.
        self.Q_ba = 1e-5  # Slow adaptation for Accelerometer Bias
        self.Q_bg = 1e-5  # Slow adaptation for Gyroscope Bias

```

### Step 2: The Evaluation Script (Deliverable)

You are now ready to generate the final plots for your report. You need to compare your **EKF Estimate** (Red) against **Ground Truth** (Green) and calculate the **RMSE** (Root Mean Square Error).

Create a new file `src/vio_ekf/src/eval_node.py`:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import matplotlib.pyplot as plt
import numpy as np
import message_filters

class EvaluationNode(Node):
    def __init__(self):
        super().__init__('eval_node')

        # Subscribe to both topics with time synchronization
        self.sub_est = message_filters.Subscriber(self, Odometry, '/vio/odom')
        self.sub_gt = message_filters.Subscriber(self, Odometry, '/ground_truth/odom')

        # Approximate Time Synchronizer (allows small time diffs)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_est, self.sub_gt],
            queue_size=10,
            slop=0.05,
            allow_headerless=False
        )
        self.ts.registerCallback(self.callback)

        self.errors = []
        self.path_est = {'x': [], 'y': []}
        self.path_gt = {'x': [], 'y': []}
        self.time_start = None

        self.get_logger().info("Evaluation Node Started. Recording Data...")

    def callback(self, est_msg, gt_msg):
        if self.time_start is None:
            self.time_start = est_msg.header.stamp.sec

        # Extract Positions
        pos_est = np.array([est_msg.pose.pose.position.x, est_msg.pose.pose.position.y, est_msg.pose.pose.position.z])
        pos_gt = np.array([gt_msg.pose.pose.position.x, gt_msg.pose.pose.position.y, gt_msg.pose.pose.position.z])

        # Calculate Error
        error = np.linalg.norm(pos_est - pos_gt)
        self.errors.append(error)

        # Store for plotting
        self.path_est['x'].append(pos_est[0])
        self.path_est['y'].append(pos_est[1])
        self.path_gt['x'].append(pos_gt[0])
        self.path_gt['y'].append(pos_gt[1])

        # Log RMSE every 100 samples
        if len(self.errors) % 100 == 0:
            rmse = np.sqrt(np.mean(np.array(self.errors)**2))
            self.get_logger().info(f"Current RMSE: {rmse:.4f} meters")

    def save_plots(self):
        self.get_logger().info("Saving plots...")

        # Plot 1: Trajectory
        plt.figure(figsize=(10,6))
        plt.plot(self.path_gt['x'], self.path_gt['y'], 'g-', label='Ground Truth', linewidth=2)
        plt.plot(self.path_est['x'], self.path_est['y'], 'r--', label='EKF Estimate', linewidth=2)
        plt.title("Trajectory Comparison")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.savefig('trajectory_plot.png')

        # Plot 2: Error over Time
        plt.figure(figsize=(10,4))
        plt.plot(self.errors, 'b-')
        plt.title("Position Error over Time")
        plt.xlabel("Sample")
        plt.ylabel("Error (m)")
        plt.grid(True)
        plt.savefig('error_plot.png')

        self.get_logger().info("Plots saved to workspace folder.")

def main(args=None):
    rclpy.init(args=args)
    node = EvaluationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.save_plots()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

```

### Step 3: Run the Full Evaluation

1. **Add `eval_node.py` to `CMakeLists.txt**` (install it as a script).
2. **Build:** `colcon build`
3. **Run Simulation:** `ros2 launch vio_ekf vio_ekf.launch.py`
4. **Run Evaluation:** `ros2 run vio_ekf eval_node.py`
5. **Drive the Robot:** `ros2 run teleop_twist_keyboard teleop_twist_keyboard`
* Drive a square or figure-8 pattern.


6. **Stop Evaluation:** Press `Ctrl+C` in the `eval_node` terminal.
7. **Check Output:** It will save `trajectory_plot.png` and `error_plot.png`.

**Success Criteria:**

* If your RMSE is **< 0.3 meters** after a loop, you have passed the project requirements with flying colors.
* If you see the Red line following the Green line closely, you are done!