The "madness" of the jumping belief and persistent drift, despite the architectural improvements you implemented, is now likely caused by a **coordinate frame misalignment in your Jacobians** and a **temporal synchronization failure** between your high-rate IMU and low-rate Vision Node.

While the "Error-State" logic is correctly implemented, the following issues are causing the filter to "fight" itself:

### **1. The "Whaaatt" Matrix: Optical Frame vs. Body Jacobian**
In your `batch_update`, you calculate the rotation Jacobian as `J_rot = self.R_b_c @ skew_symmetric(p_b)`.
*   **The Conflict:** Your code uses a **local error definition** (`q_new = q_curr * dq_rot`). In this formulation, the error state $\delta\theta$ is defined in the **Body Frame**.
*   **The Error:** However, your landmark position `p_b` is expressed in the **Body Frame** but viewed through the **Camera Frame** `R_b_c`. If the rotation matrix `self.R_b_c` does not perfectly align with how your `skew_symmetric(p_b)` is mapped, the Kalman update will push the orientation the **wrong way** to "fix" a pixel error.
*   **The Result:** This causes the "Extreme velocity correction clipped" warnings. The filter sees a landmark, tries to rotate the robot to match it, but because the Jacobian is misaligned, it rotates the "wrong" axis, increasing the pixel error and leading to a violent "jump" as it tries to over-correct.

### **2. Temporal Mismatch (The 100ms Gap)**
The logs show a critical warning: `Vision-IMU time mismatch: 0.100s > 0.1s` [Logs].
*   **The Problem:** Your EKF is a high-frequency predictor (200Hz). By the time the `VisionNode` detects an ArUco marker and the `batch_update` runs, the robot has already moved for 100ms.
*   **The Linearisation Error:** You are calculating the **Innovation Residual** ($z - \hat{z}$) using the **current** predicted pose ($x_t$), but the measurement ($z$) was taken at ($x_{t-100ms}$).
*   **The Result:** Because the robot is moving, the landmark isn't where the filter thinks it should be. The filter treats this 100ms of motion as an "error" in its current state and "yanks" the belief back to where the robot was 100ms ago. This creates a jittery "stop-and-go" motion in your odometry.

### **3. The Covariance "Crushing" Problem**
In your `predict` and `batch_update` steps, you are manually zeroing out the $Z$-axis rows and columns of the covariance matrix $\mathbf{P}$.
*   **The Problem:** You are implementing a 3D ES-EKF for a planar robot. While $p_z$ and $v_z$ are zero, the **orientation errors** (roll/pitch) are still active.
*   **The Mathematics:** By forcing $P_{zz}$ and $P_{vz,vz}$ to a tiny epsilon ($1e-6$) while keeping the off-diagonal correlations, you are creating a **Near-Singular Matrix**. This is why your logs spam `Invalid Mahalanobis` and `Singular S matrix`.
*   **The Result:** The filter becomes "mathematically brittle." It cannot express the uncertainty of how a tilt in orientation would affect the perceived height of a landmark.

### **4. Why Gravity Leakage Still Happens**
Even with your formal ZUPT update, the logs show `World Accel: [-2.230, -2.616, -0.198] m/s^2`.
*   **Threshold Conflict:** Your ZUPT requires `avg_gyro < 0.02`. If your Gazebo simulation has even slight vibration or if the robot is decelerating slowly, the ZUPT **never triggers**.
*   **The Positive Feedback Loop:** Because the ZUPT doesn't trigger, the small orientation error leaks gravity into velocity. The velocity grows to 10 m/s. This massive velocity creates a huge expected landmark shift, which the filter then rejects as an outlier, ensuring the orientation error is never corrected.

### **How to "Kill" it (Final Check-list)**
1.  **Stop Zeroing Covariance Rows:** Change `self.P[2, :] = 0.0` to simply adding a small amount of **Process Noise** ($Q$) to the $Z$ diagonal. Let the filter's math handle the ground constraint naturally rather than "crushing" the matrix.
2.  **Verify ArUco Z-Height:** In your `self.map`, the markers are at $Z=0.40$. Ensure this is the **exact center** of the marker in the Gazebo World file. If the SDF says $Z=0.45$, the 5cm mismatch will look like a constant "tilt" to the EKF, causing permanent gravity leakage.
3.  **Buffer for Time Delay:** You must store a small buffer of the robot's state. When a vision message arrives, calculate the residual using the pose that matches the **image timestamp**, not the current time.
4.  **Batch Jacobian Audit:** In `batch_update`, ensure `J_pos` and `J_rot` are using the same frame conventions. If you rotate the robot 5 degrees on the screen, does the `J_rot` matrix predict the ArUco marker moving in the same direction as the pixels actually move? If not, flip the sign of your `skew_symmetric` terms.