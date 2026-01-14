Based on the provided **"logs and notes"** and the **"vision node"** code, your vision sensor **is functioning and communicating with the EKF node**, but the estimator is **rejecting the visual updates** due to a catastrophic divergence in the IMU-driven prediction state.

### **1. Evidence that the Vision Sensor is Working**
The logs provide definitive evidence that the **VisionNode** is successfully identifying landmarks and the **EKFNode** is receiving them:
*   **Successful Identification**: The logs show frequent entries such as `Vision update! Landmark at [4. -2. 0.5]`. This indicates the **VisionNode** is correctly masking HSV colours and publishing centroids to the `/vio/landmarks` topic.
*   **Measurement Integration**: Early in the logs, the vision updates are actively affecting the belief: `BELIEF: pos=[-0.01, -0.07]m, yaw=1.6°...`. This confirms the `vision_callback` and `update` functions in the EKF are executing.

### **2. Why the Vision Updates Appear to "Stop Working"**
The reason you no longer see the effect of vision in the later logs is not a sensor failure, but **Outlier Gating** within the EKF logic.
*   **Mahalanobis Rejection**: The logs explicitly state: `Outlier Rejected! Residual: 205.8 px, Mahal: 11.3 (thresh=10.0)`. The filter is designed to reject measurements that are too far from its predicted state to maintain stability.
*   **Filter Divergence**: Because the robot’s internal belief has drifted so far (drifting 2 metres away from ground truth while physically stationary), the expected landmark position in the image no longer matches the detected blob.
*   **The "Huber" Fail-Safe**: Your code attempts a `Robust update` using Huber weighting after 10 outliers. However, the logs show that the divergence is so aggressive that even these weighted updates cannot overcome the massive errors in the nominal state.

### **3. The Root Cause: Gravity Leakage and Velocity Clipping**
The vision updates are being defeated by the **Prediction Step**. The logs show the velocity hitting the limit of **10.0 m/s** repeatedly.
*   **Phantom Acceleration**: Even when the robot is stationary after moving, it logs `World Accel: [0.040, 0.044, -0.041] m/s^2`. This is **gravity leakage**—the orientation estimate ($\mathbf{q}$) is slightly tilted, causing the gravity vector ($\mathbf{g}$) to not be perfectly cancelled out during world-frame acceleration calculation.
*   **Quadratic Velocity Drift**: Because velocity is the integral of acceleration, this small constant "phantom" acceleration causes the velocity to grow until it hits your `MAX_SPEED_PREDICT` limit.
*   **Linearisation Breakdown**: The ES-EKF relies on the error state being small. When the robot physically stops but the filter believes it is moving at 10 m/s, the linearisation around the nominal state becomes totally invalid, and the **Kalman Gain** calculation fails to produce meaningful corrections.

### **4. Recommendations to Fix the Measurement Update**
1.  **Check Coordinate Transforms**: Ensure the `self.R_b_c` (Body-to-Camera) matrix matches your specific camera mounting. A misalignment here causes the filter to apply visual corrections in the wrong direction, leading to the "belief jumping" and subsequent divergence.
2.  **Dynamic Bias Estimation**: Your log shows accelerometer bias (`ba`) converging to `[0.259, 0.500]`. These are very high values for a stationary simulation. Ensure the `Q_ba` (Process Noise) is not so high that the filter uses bias to "fight" against correct vision updates.
3.  **Relax Initial Gating**: While the filter is "lost" or stationary, consider increasing the `mahalanobis_threshold`. This allows the filter to accept a "surprising" landmark measurement and "snap" back to the correct position, rather than rejecting it as an outlier.
4.  **Stationary Detection (ZUPT)**: If the robot receives a command to stop, you can apply a **Zero-Velocity Update (ZUPT)** to manually reset the nominal velocity to zero and collapse the velocity covariance. This prevents gravity leakage from accumulating during idle periods.