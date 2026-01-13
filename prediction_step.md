The **prediction step** in an **Error-State Extended Kalman Filter (ES-EKF)** involves the high-frequency propagation of the **nominal state** using inertial measurements and the simultaneous propagation of the **error state covariance**.

The following pseudocode outlines the implementation of the IMU odometry/kinematics model for this process:

### **Pseudocode: ES-EKF Prediction Step**

**Inputs:**
*   Previous nominal state **$x_{k-1}$** (Position $p$, Velocity $v$, Quaternion $q$, Biases $b_a, b_g$).
*   Previous error covariance matrix **$P_{k-1}$**.
*   Raw IMU measurements **$u_k$** (Linear acceleration $a_m$, Angular velocity $\omega_m$).
*   Time interval **$\Delta t$** between samples.

---

**1. Nominal State Propagation (Non-linear Kinematics)**
*   **Bias Correction**: Adjust raw measurements using current bias estimates:
    *   $a_{corrected} = a_m - b_a$.
    *   $\omega_{corrected} = \omega_m - b_g$.
*   **Update Position ($p$)**:
    *   $p_k = p_{k-1} + v_{k-1}\Delta t + \frac{1}{2}(R(q_{k-1}) \cdot a_{corrected} - g)\Delta t^2$.
*   **Update Velocity ($v$)**:
    *   $v_k = v_{k-1} + (R(q_{k-1}) \cdot a_{corrected} - g)\Delta t$.
*   **Update Orientation (Quaternion $q$)**:
    *   $\Delta q = \text{exp}(\frac{1}{2} \omega_{corrected} \Delta t)$.
    *   $q_k = q_{k-1} \otimes \Delta q$ (utilising **multiplicative updates** to maintain unit-norm constraints).

**2. Error Covariance Propagation (Linearised Uncertainty)**
*   **Compute Jacobian ($F_x$)**: Calculate the state transition Jacobian based on the current nominal state.
    *   Commonly approximated as $F_x \approx I + F\Delta t$, where $F$ is the continuous-time error-state Jacobian.
*   **Update Covariance ($P$)**: Propagate the uncertainty by incorporating **process noise ($Q_{imu}$)** from sensor fluctuations and bias random walks:
    *   **$P_{k|k-1} = F_x P_{k-1|k-1} F_x^T + Q_{imu}$**.

**3. Error State Reset**
*   In the ES-EKF framework, the **mean of the error state ($\delta \hat{x}$)** is always **reset to zero** at the start of each prediction step because the nominal state is assumed to have already absorbed previous error corrections.

---

### **Key Technical Considerations**
*   **6-DoF Scalability**: Although this specific project involves a ground vehicle, a **6-Degrees-of-Freedom** framework is used to ensure the algorithm is applicable to general 3D robotic applications.
*   **Gravity Compensation**: Acceleration must be rotated from the body frame to the world frame using the current orientation estimate before subtracting the gravity vector ($g$).
*   **Minimal Representation**: The error state uses a **3D angular vector ($\delta\theta$)** for orientation error rather than a 4D quaternion to avoid **rank deficiency** and numerical instability in the covariance matrix.

**Analogy for Covariance Propagation**: Imagine a **blindfolded hiker** (the Nominal State) walking based on their internal sense of direction. They are quite sure of their path, but they are followed by a **growing shadow** (the Covariance $P$). Every step taken without looking at a landmark makes that shadow larger and more blurred, representing the **quadratically growing noise** and uncertainty of their true position.