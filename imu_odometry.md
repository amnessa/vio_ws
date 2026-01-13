The **IMU odometry model** for the prediction step in an **Error-State Extended Kalman Filter (ES-EKF)** involves two parallel processes: propagating the high-frequency **nominal state** (large-scale non-linear motion) and the **error state covariance** (linearised uncertainty),.

The following pseudocode outlines the implementation of this prediction step as described in the sources:

### **Pseudocode: ES-EKF IMU Prediction Step**

**Input:**
*   **$x_{k-1}$**: Previous nominal state (Position $p$, Velocity $v$, Quaternion $q$, Biases $b_a, b_g$),.
*   **$P_{k-1}$**: Previous error covariance matrix,.
*   **$u_k$**: Current IMU measurement (Linear acceleration $a_m$, Angular velocity $\omega_m$),.
*   **$\Delta t$**: Time interval between IMU samples,.

---

**1. Nominal State Propagation (Non-linear Kinematics)**
*   **Correct IMU measurements with current bias estimates:**
    *   $a_{corrected} = a_m - b_a$,.
    *   $\omega_{corrected} = \omega_m - b_g$,.
*   **Update Position ($p$):**
    *   $p_k = p_{k-1} + v_{k-1}\Delta t + \frac{1}{2}(R(q_{k-1}) \cdot a_{corrected} - g)\Delta t^2$.
*   **Update Velocity ($v$):**
    *   $v_k = v_{k-1} + (R(q_{k-1}) \cdot a_{corrected} - g)\Delta t$,.
*   **Update Orientation (Quaternion $q$):**
    *   $\Delta q = \text{exp}(\frac{1}{2} \omega_{corrected} \Delta t)$ (using quaternion exponential map),.
    *   $q_k = q_{k-1} \otimes \Delta q$,.

**2. Error State Covariance Propagation (Linearised Uncertainty)**
*   **Compute the Error State Transition Jacobian ($F_x$):**
    *   Construct $F_x$ using current nominal state values to describe how errors in position, velocity, and orientation propagate,.
    *   $F_x = I + F \Delta t$ (where $F$ is the continuous-time Jacobian),.
*   **Define Process Noise Matrix ($Q_{imu}$):**
    *   $Q_{imu}$ captures uncertainty from IMU sensor noise and bias random walks,.
*   **Propagate Covariance ($P$):**
    *   **$P_{k|k-1} = F_x P_{k-1|k-1} F_x^T + Q_{imu}$**,.

**3. Error State Mean (Internal Logic)**
*   In the ES-EKF, the predicted error state mean ($\delta \hat{x}_{k|k-1}$) is always **reset to zero** at the start of the prediction step,.
*   $\delta \hat{x}_{k|k-1} = 0$,.

**Output:**
*   Updated nominal state $x_k$ and predicted error covariance $P_{k|k-1}$,.

---

### **Key Structural Elements**
*   **Quaternion Handling**: Unlike a standard EKF that might use additive updates on quaternions (violating unit-norm constraints), the ES-EKF propagates the nominal orientation multiplicatively and maintains orientation error in a 3D minimal representation ($\delta\theta$),.
*   **Linearisation**: Because the error state maintains small values near the origin, the linearisation used in $F_x$ is much more accurate than in a standard EKF, which linearises the entire non-linear state,.
*   **Gravity Compensation**: Linear acceleration from the IMU must be rotated into the world frame using the current orientation estimate before subtracting the gravity vector $g$,.

***

**Analogy for Covariance Propagation**: Imagine you are **tracking a hiker** on a map based only on their compass and speed. The **Nominal State** is the line you draw for their path. The **Covariance $P$** is a shadow that follows the line; as the hiker walks further without checking a landmark, that shadow gets wider and longer, representing your growing uncertainty about their exact spot. The **Jacobian $F_x$** is the math that tells you exactly how much that shadow should grow based on whether the hiker is turning or running fast.