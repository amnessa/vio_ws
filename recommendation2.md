Your updated **ES-EKF Node** and **Vision Node** implementations now correctly handle the **data association problem** by using 50 unique ArUco markers, which eliminates landmark confusion. However, the logs show that the system is still suffering from **catastrophic divergence** and **velocity clipping**.

Based on the code and logs, the reason your odometry is "jumping around" despite these fixes is an **internal conflict between your manual ZUPT corrections and the EKF's mathematical model**.

### **1. The "ZUPT Feedback Loop" (Log Analysis)**
The logs show a persistent `ZUPT orientation correction` of ~1.15° occurring every second while stationary.
*   **The Conflict:** In Step 2 of your `predict` function, you manually overwrite the nominal quaternion $q$ based on the accelerometer's gravity vector.
*   **The Result:** In a formal ES-EKF, orientation should only be corrected via the **Error State** ($\delta\theta$) during a measurement update. By "yanking" the nominal state quaternion during the prediction step, you are changing the robot's belief without updating the **Covariance Matrix ($P$)**.
*   **The Failure:** Because $P$ is not updated to reflect this manual change, the filter becomes mathematically inconsistent. The IMU continues to integrate based on its internal noise model, while your code "fights" it by resetting the orientation. This creates the "jitter" that eventually triggers the `Filter Divergence!` error.

### **2. Flawed Re-initialization Logic**
The logs show that after the divergence (29.2 m/s²), the filter re-initializes but immediately hits the **10.0 m/s velocity limit** again.
*   **Root Cause:** When the filter diverges, your `_reinitialize_orientation` function uses a single accelerometer sample to set the new tilt. If the robot is vibrating or experiencing a shock, this "smart" re-initialization sets a **wrong initial tilt**.
*   **The Leak:** Even a 2-degree error in this new orientation causes gravity to "leak" into the world-frame acceleration calculation ($acc\_world = R \cdot a - g$) [Logs, 1718]. This leaked gravity is integrated into velocity, which grows quadratically until it hits the `MAX_SPEED_PREDICT` limit.

### **3. The "Bias Decay" Paradox**
You added a line to **decay the accelerometer bias** toward zero during ZUPT (`bias_decay_rate = 0.995`).
*   **The Problem:** This is a non-physical heuristic that actively **prevents the EKF from doing its job**. The EKF is designed to *learn* the constant sensor bias ($b_a$) to cancel out gravity leakage.
*   **The Consequence:** By forcing the bias toward zero, you are deleting the filter's learned knowledge of the sensor's offset. When the robot stops, the filter tries to estimate the bias to fix the drift; your code then "erases" that estimate, ensuring the drift starts again the moment ZUPT is deactivated.

### **4. Vision Sensor Observability with ArUco**
The **Vision Node** is correctly publishing ArUco IDs to `/vio/landmarks`. However, the **EKF Node** handles these updates sequentially.
*   **The "Yanking" Effect:** Because you have high landmark density (50 markers), the camera likely sees 5–10 markers at once. Your code performs a full EKF update for **each marker individually** within a single callback.
*   **Linearization Error:** If the robot belief has drifted (due to the gravity leakage mentioned above), the first marker update will apply a massive correction. Because the nominal state $q$ and $p$ change significantly after the first update, the **linearization point** for the second and third markers becomes invalid. This is why the belief "jumps" violently when ArUco markers are detected.

### **Recommended Fixes**
1.  **Remove Manual Orientation Overrides:** Delete the "ZUPT Part 2" orientation correction from the `predict` function. Allow the EKF to correct orientation naturally through the ArUco measurement updates.
2.  **Stop Decaying Biases:** Remove the `bias_decay_rate`. The EKF must be allowed to maintain a steady estimate of $b_a$ to counteract gravity leakage effectively.
3.  **Implement Batch Updates:** Instead of calling `self.update()` for every marker in the loop, stack all visible ArUco markers into a single **Batch Measurement Jacobian ($H$)** and update the filter once per camera frame. This ensures all landmarks are used to find a single consistent correction.
4.  **Trust the Bias Random Walk:** Increase the time the robot stays stationary during `initialize_from_imu` (or increase the sample count) to ensure the `initial_accel_bias` is highly accurate before the first movement.