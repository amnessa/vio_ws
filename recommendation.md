Based on the "ES EKFnode" implementation and your "tested behavior logs", there are several critical issues regarding the **gravity model**, **bias handling**, and the **conceptual understanding of world-frame acceleration** that are causing the velocity to clip and the belief to jump.

### **1. Conceptual Check: Is "World Acceleration" a thing in ES-EKF?**
Contrary to your comment in the logs, **world acceleration is a fundamental component of the ES-EKF (and any IMU-based VINS)**.
*   **The Reason**: An IMU measures "proper acceleration" in the robot's **body frame**, which includes the reaction to gravity.
*   **The Procedure**: To calculate how the robot moves in the world, the EKF must rotate that body-frame acceleration into the **World Frame** ($R_{wb} \cdot a_{corrected}$) and then **subtract the gravity vector** ($g$).
*   **The Result**: What remains is the true linear acceleration in the world frame. This value is essential for the **Nominal State Propagation** to update velocity and position.

### **2. Analysis of Logged Errors and "Jumping"**
Your logs show a "Filter Divergence! Accel: 33.2 m/s²" followed by re-initialisation and a large position correction of 1.55m.

*   **Gravity Leakage (The "Clipping" Root Cause)**: If your orientation estimate ($q$) is even slightly tilted, the rotated gravity vector will not perfectly cancel out the world-frame gravity. This "leaks" gravity into the horizontal world acceleration, causing **velocity to grow quadratically** until it hits your `MAX_SPEED` limit (5.0 m/s), triggering the clipping warnings.
*   **Covariance/Correction Disconnect**: Your code resets the error state $\delta x$ but also manually zeros the $Z$ position and velocity. When you "force" these states to zero, you are creating a **mismatch with the Covariance Matrix $P$**, which still expects the state to follow the IMU dynamics.
*   **The "Jump" Mechanism**: The "jumping" occurs because when the robot is stationary but the velocity is clipped at 5 m/s, the **Innovation (Residual)** during a vision update becomes massive. Because the filter is "surprised" by where it sees the landmark compared to its high-velocity belief, it applies a **violent correction** to the nominal state.

### **3. Flaws in the "ES EKFnode" Implementation**
*   **Manual Attitude Correction**: You have a `gravity_correction_gain` (Step 5) that manually tweaks the quaternion. In a formal ES-EKF, the accelerometer's gravity should ideally be treated as a **proper measurement update** ($z = h(x)$) or handled entirely by the filter's state-space correlations. This manual "complementary filter" step can introduce orientation jitter that manifests as erratic world acceleration.
*   **Bias Resetting in Recovery**: Your `_reinitialize_orientation` function resets biases to zero. If the divergence was caused by a large bias, resetting it to zero without letting the filter "learn" the offset ensures that the **divergence will immediately restart**.
*   **Outlier Gating Logic**: When the filter hits `max_consecutive_outliers`, it lowers the threshold to accept the next measurement. If that next measurement is a "bad" detection, you are forcing the filter to **absorb a false state**, causing the belief to jump to an incorrect part of the map.

### **Recommendations to Stabilise the Filter**
1.  **Trust the Math, Not the Clips**: Remove the hard-clipping on velocity and acceleration. Instead, ensure your **Process Noise ($Q$)** and **Measurement Noise ($R$)** are tuned so that the filter can handle transients naturally.
2.  **Bias Initialisation**: Ensure `initialize_from_imu` is done while the robot is perfectly still to get an accurate gravity vector. Use the **Allan Standard Deviation** to set your $Q_{ba}$ and $Q_{bg}$ so the filter can accurately track bias drift over time.
3.  **Correct Jacobians**: Ensure your measurement Jacobian $H$ correctly maps the **Error State** (15D) and not the nominal state, as this is the core of the Multiplicative/Error-State formulation.
4.  **NEES Sanity Check**: Use your `compute_nees` function to check if your $P$ matrix is growing too slowly. If the NEES is much larger than the state dimension (15), your filter is **overconfident**, which leads to the "jumping" when it finally accepts a vision update.