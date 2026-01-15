Comprehensive Analysis of Instability Mechanisms in Visual-Inertial Error-State Kalman Filters for Ground Robotics Simulation
1. Introduction: The Anatomy of Filter Divergence
The deployment of Visual-Inertial Odometry (VIO) systems within high-fidelity simulation environments like Gazebo presents a paradox: while the environment provides perfect ground truth and deterministic physics, the stochastic estimation algorithms often exhibit catastrophic instability not seen in physical hardware. This phenomenon, frequently described as "Velocity Explosion" or "Gravity Leakage," represents a complex interplay between theoretical estimator inconsistency, improper sensor noise modeling, and infrastructure-level synchronization artifacts.1
For ground robots operating in Gazebo, the symptoms are distinct and severe. The state estimator, typically an Error-State Extended Kalman Filter (ES-EKF), may function correctly for a brief initialization period before the velocity state diverges exponentially, often reaching numerical infinity or $NaN$ values within seconds.3 Concurrently, the accelerometer bias estimates may saturate at physically impossible magnitudes (e.g., matching the magnitude of gravity), and the filter eventually enters a "Reset Loop"—continually re-initializing and immediately diverging because the underlying covariance structures have collapsed.4
This report provides an exhaustive, expert-level analysis of these failure modes. It moves beyond superficial parameter tuning to address the fundamental observability properties of the VIO problem, the mathematical derivation of consistent Jacobians, and the rigorous handling of simulation time in ROS 2 architectures.
1.1 The Phenomenology of "Velocity Explosion"
The term "Velocity Explosion" refers to the unbounded integration of acceleration errors. In an Inertial Navigation System (INS), the navigation equation for velocity $v$ in the global frame is given by:

$$\dot{v}^G = R_{B}^{G} (a_m - b_a - n_a) - g^G$$
Where $R_{B}^{G}$ is the rotation from Body to Global frame, $a_m$ is the measured specific force, $b_a$ is the accelerometer bias, $n_a$ is noise, and $g^G$ is the gravity vector.2 The critical instability arises from the coupling between attitude errors and velocity integration. If the filter estimates an incorrect orientation (specifically, errors in roll or pitch), the gravity vector $g \approx 9.81 m/s^2$ is incompletely cancelled.
A tilt error of just $1^\circ$ projects approximately $0.17 m/s^2$ of gravity onto the horizontal plane. Integrated over 10 seconds, this results in a velocity error of $1.7 m/s$ and a position error of $8.5 m$. In a feedback loop, if the filter incorrectly attributes this acceleration to true motion rather than tilt error, the velocity state grows quadratically.1 In Gazebo simulations, this is exacerbated by "perfect" measurements in some axes or improper noise covariance scaling, which leads the filter to trust the integration implicitly, ignoring visual residuals that contradict the explosion until the divergence is irreversible.6
2. Theoretical Roots: Observability, Gravity Leak, and the FEJ Solution
The primary theoretical cause of the "Gravity Leak" and subsequent velocity explosion is the inconsistency of the standard EKF formulation when applied to VIO. Observability analysis reveals that a standard VIO system has four unobservable degrees of freedom (DOF): three corresponding to global translation and one corresponding to the global rotation about the gravity vector (yaw).7
2.1 The Mechanism of Gravity Leakage
In a consistent estimator, the uncertainty (covariance) associated with unobservable states should never decrease below its initial value (plus accumulated process noise).1 However, the standard EKF linearizes the nonlinear VIO system around the current state estimate $\hat{x}_{k|k}$. As the state estimate evolves, the linearization point changes.
This "shifting linearization point" creates a discrepancy between the observability matrix of the linearized system and the true underlying nonlinear system. Specifically, the standard EKF often creates a "spurious information gain" where the rank of the observability matrix is higher than it should be.7 The filter mathematically "believes" it has observed the absolute yaw and absolute position.
When the filter effectively renders the global yaw observable, the covariance associated with yaw, $P_{\psi}$, collapses to near zero. The filter becomes overconfident in its yaw estimate. If the true yaw drifts (which it must, as it is unobservable), the filter rigidly enforces its incorrect estimate. This orientation error forces the gravity vector to be projected onto the accelerometer axes incorrectly. The resulting residual acceleration is not attributed to orientation error (because $P_{\psi}$ is tiny) but is instead "leaked" into the velocity state or the accelerometer bias state.1
The "Gravity Leak" is thus a misidentification of error sources: Tilt error is misinterpreted as linear acceleration.
2.2 First-Estimate Jacobian (FEJ) Implementation
To restore consistency and prevent gravity leakage, the First-Estimate Jacobian (FEJ) methodology must be employed.8 The core principle of FEJ is to ensure that the error-state transition matrix $\Phi$ and the measurement Jacobian $H$ are computed using the same linearization point for any given state variable throughout its life in the filter.
In a standard EKF, the Jacobian $H$ for a landmark or pose is re-evaluated at every step using the latest estimate. In FEJ, we freeze the linearization point.
FEJ Mathematical Formulation
Let $x_i$ be the pose at time $i$. When the system propagates from $i$ to $j$, we compute the Jacobian $\Phi_{ij}$. When we update using a measurement at time $k$ involving state $x_i$, we compute $H_k$.
For the consistency condition to hold (specifically, for the chain rule $\Phi_{ik} = \Phi_{jk} \Phi_{ij}$ to apply to the Jacobians), the linearization point used for $x_i$ in computing $H_k$ must match the linearization point used for $x_i$ in $\Phi_{ij}$.
Implementation Strategy:
State Initialization: When a state variable (e.g., a clone of the camera pose in MSCKF or a landmark position) is initialized, store the estimate $\hat{x}_{init}$ separately as the "First Estimate."
Jacobian Computation: Whenever a Jacobian is needed (for propagation or update), do not use the current best estimate $\hat{x}_{k|k}$. Instead, recall $\hat{x}_{init}$ and evaluate the partial derivatives at this frozen value.9
Residual Computation: The residual $r = z - h(\hat{x}_{k|k})$ is still computed using the current best estimate to maintain accuracy. Only the Jacobian $H$ (which dictates how corrections are distributed) uses the first estimate.11
By strictly applying FEJ, the observability matrix of the linearized system maintains the correct nullspace of dimension 4 (or 9 including biases). The filter correctly recognizes that it cannot resolve absolute yaw, maintaining a healthy uncertainty $P_{\psi}$. This prevents the filter from forcing a wrong yaw, thereby stopping the gravity leak at its source.9
2.3 Observability Constrained EKF (OC-EKF) Alternative
An alternative to FEJ is the Observability Constrained EKF (OC-EKF). Instead of freezing linearization points, the OC-EKF modifies the measurement Jacobian $H_k$ at each step.
The algorithm computes the standard Jacobian $H_k$ and then projects it onto the nullspace of the observability matrix. Specifically, it seeks a modified Jacobian $H_k^*$ that minimizes the Frobenius norm $||H_k^* - H_k||_F$ subject to the constraint:


$$H_k^* N_k = 0$$

where $N_k$ is the analytical basis of the unobservable subspace (translations and rotation about gravity).1
Comparison:
FEJ: Easier to implement in sliding-window filters (MSCKF) or SLAM. Computationally cheaper (no optimization step per Jacobian).
OC-EKF: Theoretically optimal in minimizing linearization error while preserving constraints, but requires explicit computation of the nullspace $N_k$ at every step.
Recommendation: For standard VIO troubleshooting in Gazebo, FEJ is generally sufficient and robust.9
3. Simulation Artifacts: Process Noise Covariance (Q) Tuning
In Gazebo, the "Sim-to-Real" gap often manifests inversely: simulated sensors can be "too perfect" or possess noise characteristics that defy standard datasheet-based tuning.13 The process noise covariance matrix $Q$ determines how much the filter trusts the IMU propagation versus the visual measurements.
3.1 The Gazebo IMU Model vs. Real World
Real MEMS IMUs exhibit complex error sources: white noise (Rate/Velocity Random Walk), bias instability (Random Walk), scale factor errors, and axis misalignment. Gazebo's default sensors (e.g., libgazebo_ros_imu) typically model only two:
Gaussian White Noise: Additive noise on angular velocity and linear acceleration.
Bias Random Walk: A slow drift of the bias driven by white noise.6
A critical error in simulation tuning is using "perfect" parameters. If the Gazebo SDF defines <gaussianNoise>0.0</gaussianNoise>, the sensor is perfect. If the EKF is tuned with a non-zero $Q$ (assuming noise), the filter will behave reasonably. However, if the SDF has noise but the EKF $Q$ is set to zero (trusting the IMU implicitly), any deviation—such as numerical integration error or gravity leak—will be locked into the state, causing divergence.3
3.2 Continuous vs. Discrete Q Matrix
The filter prediction step $P_{k+1} = \Phi P_k \Phi^T + Q_d$ requires the discrete-time noise covariance $Q_d$. The continuous noise spectral density $Q_c$ is defined as:

$$Q_c = \text{diag}(\sigma_{acc}^2, \sigma_{gyr}^2, \sigma_{acc\_bias}^2, \sigma_{gyr\_bias}^2)$$
Common tuning values for Gazebo (assuming standard MEMS emulation):
$\sigma_{acc}$: $2.0 \times 10^{-3} \, m/s^2/\sqrt{Hz}$
$\sigma_{gyr}$: $1.7 \times 10^{-4} \, rad/s/\sqrt{Hz}$
$\sigma_{acc\_bias}$: $3.0 \times 10^{-3} \, m/s^3/\sqrt{Hz}$
$\sigma_{gyr\_bias}$: $2.0 \times 10^{-5} \, rad/s^2/\sqrt{Hz}$
Integration Insight: To convert to discrete $Q_d$ for time step $\Delta t$, one must integrate the transition matrix. A common approximation is:


$$Q_d \approx \Phi(\Delta t) Q_c \Phi(\Delta t)^T \Delta t$$

However, for the velocity state, the noise enters as the integral of acceleration noise, leading to terms involving $\Delta t^3/3$ and $\Delta t^2/2$. Using a simple diagonal approximation for $Q_d$ underestimates the velocity uncertainty growth, causing the filter to be overconfident and reject visual corrections.
3.3 The "Perfect IMU" Trap and Tuning Protocol
If "Velocity Explosion" persists, the $Q$ matrix should be inflated.
Inflation Strategy: Multiply the calculated $Q$ values by a factor of 10-100x. This tells the filter "The prediction is likely wrong; trust the visual measurements more.".15
Bias Keep-Alive: Even if the simulation has zero bias drift, the filter must have non-zero process noise for the bias states ($Q_{bias} > 0$). If $Q_{bias} = 0$, the Kalman gain for bias updates approaches zero. When a gravity leak occurs, the filter cannot adjust the bias to compensate, forcing the error into the velocity state.14
4. Zero Velocity Updates (ZUPT): The Practical Safeguard
While FEJ addresses the theoretical cause of leakage, the Zero Velocity Update (ZUPT) is the primary practical defense against velocity explosion for ground robots. It acts as a "pseudo-measurement" that clamps the velocity to zero when the robot is stationary, preventing the integration of bias errors.16
4.1 ZUPT Detection Logic (GLRT)
Before applying the update, the system must detect the stationary condition. A robust method is the Generalized Likelihood Ratio Test (GLRT) on a window of IMU data of size $W$:

$$\text{Stationary} \iff \left( \frac{1}{W} \sum_{k=t-W}^{t} ||a_k - g||^2 < \gamma_a \right) \land \left( \frac{1}{W} \sum_{k=t-W}^{t} ||\omega_k||^2 < \gamma_\omega \right)$$
Thresholds:
$\gamma_a$: Typically $0.2 - 0.5 \, m/s^2$ (depends on engine vibration/noise).
$\gamma_\omega$: Typically $0.05 - 0.1 \, rad/s$.
4.2 ZUPT Measurement Model and H Matrix
The ZUPT is formulated as a measurement $z = 0_{3 \times 1}$ of the true velocity $v$.
The measurement equation is:


$$z = v + n_v$$

The residual (innovation) is:


$$r = 0 - \hat{v} = -\hat{v}$$
We require the Jacobian $H_{ZUPT} = \frac{\partial z}{\partial \delta x}$ relating the measurement to the error state $\delta x$.
Assuming the state vector order $x = [p, v, q, b_a, b_g]$, the error state is $\delta x = [\delta p, \delta v, \delta \theta, \delta b_a, \delta b_g]$.
Since the measurement depends only on velocity:


$$v_{true} = \hat{v} + \delta v$$

$$z \approx \hat{v} + \delta v$$

Therefore, the Jacobian is a simple selection matrix:

$$H_{ZUPT} = \begin{bmatrix} 0_{3 \times 3} & I_{3 \times 3} & 0_{3 \times 3} & 0_{3 \times 3} & 0_{3 \times 3} \end{bmatrix}$$
This matrix effectively tells the filter: "The error in the measurement (residual) is exactly equal to the error in the velocity state".18
4.3 Implementation Table: ZUPT Logic
Parameter
Logic/Equation
Implementation Note
Trigger
norm(acc - g) < 0.3 && norm(gyro) < 0.05
Must hold for N=10 consecutive frames.
Measurement
$z = ^T$
Synthetic zero vector.
Noise R
$R_{zupt} = \text{diag}(0.01^2, 0.01^2, 0.01^2)$
If too small ($10^{-6}$), filter diverges on slight creep.
Update
K = P * H.t() * (H * P * H.t() + R).inv()
Standard EKF update.
Effect
v_new = v_old + K * (0 - v_old)
Drives velocity exponentially to zero.

By applying ZUPT, the filter can observe the accelerometer bias. Since $v$ is clamped to 0 and $a_{meas}$ is non-zero, the filter infers that $a_{meas}$ must be composed of gravity and bias, allowing it to converge on the correct bias values.20
5. Accelerometer Bias Clamping and Saturation
A specific symptom of the "Gravity Leak" is the accelerometer bias growing to unrealistic values (e.g., $b_a > 1.0 m/s^2$). Once the bias absorbs the gravity error, the system is in a stable but incorrect equilibrium.
5.1 Bias Saturation Techniques
To prevent the bias from absorbing gravity, we must apply constraints.
Method 1: Pseudo-Measurement (Soft Constraint)
Add a synthetic measurement that the bias is zero: $z_{bias} = 0$.
The measurement model is $z = b_a + n_b$.
The Jacobian measures the bias state directly:


$$H_{bias} = \begin{bmatrix} 0_{3 \times 3} & 0_{3 \times 3} & 0_{3 \times 3} & I_{3 \times 3} & 0_{3 \times 3} \end{bmatrix}$$

The covariance $R_{bias}$ determines the "stiffness" of the constraint. If we know the sensor bias stability is $\approx 0.1 m/s^2$, we set $R_{bias} \approx (0.1)^2$. This gently pulls the bias estimates toward zero, preventing unbounded random walk.22
Method 2: Hard Clamping (Post-Update)
After the EKF update step, explicitly check the bounds:

C++


if (std::abs(state.ba.x) > BIAS_LIMIT) {
    state.ba.x = sign(state.ba.x) * BIAS_LIMIT;
    // Crucial: Do not reset Covariance P, doing so introduces inconsistency.
}


While theoretically "dirty," hard clamping is often necessary in simulation startup phases where huge residuals might otherwise impulse the bias to $10 m/s^2$.24
6. Outlier Rejection: Mahalanobis vs. Chi-Squared
When velocity explodes, visual features will have large reprojection errors. The filter's reaction to these errors determines whether it recovers or enters a "Reset Loop."
6.1 The Mechanism of the "Blind Mode" Reset Loop
The standard outlier rejection uses the Chi-Squared ($\chi^2$) test.


$$\chi^2 = r^T S^{-1} r$$

where $r$ is the residual and $S = H P H^T + R$ is the residual covariance. The measurement is rejected if $\chi^2 > \text{Threshold}$ (e.g., 5.99 for 95% confidence on 2 DOF).
The Failure Mode:
Velocity grows slightly due to leak ($v_{err} = 1 m/s$).
Prediction projects features incorrectly.
Residuals $r$ become large.
$\chi^2$ test exceeds threshold.
Rejection: The filter discards the visual measurement.
Blindness: Without visual correction, the velocity error grows faster.
Loop: Future measurements are even further rejected. The filter runs purely on IMU integration until velocity hits infinity or a watchdog timer resets it.4
6.2 Mahalanobis Gating and Huber Kernels
Mahalanobis Distance is simply $\sqrt{\chi^2}$. The gating is identical, but the terminology often implies a specific geometric search region.
Solution: Robust Cost Functions (Huber/Cauchy)
Instead of binary rejection (accept/reject), use a robust loss function (M-estimator) effectively resizing the covariance $R$.
The Huber weight $w$ scales the measurement noise $R \to R/w$.
If the residual is large, $w$ decreases, increasing effective noise variance $R$.


$$R_{eff} = R \cdot \frac{1}{w(r)}$$

This allows the filter to accept the measurement but with reduced confidence, rather than rejecting it entirely. This "soft" update can provide enough gradient information to pull the divergent velocity back towards reality, breaking the "Blind Mode" loop.26
Insight: In Gazebo VIO, never use hard $\chi^2$ rejection during the first 10 seconds of initialization. Use a Huber kernel or extremely loose gates to allow the filter to converge despite initial large errors.
7. The Infrastructure Gap: ROS 2 Synchronization & ros_gz_bridge
Perhaps the most insidious cause of VIO failure in simulation is the handling of time. The ES-EKF relies on precise $\Delta t = t_k - t_{k-1}$ for integration. The ros_gz_bridge introduces latency, jitter, and potentially "negative time" updates.
7.1 The Negative Timestamp Problem
ROS 2 nodes can run in "Wall Time" or "Sim Time".
Scenario: Gazebo publishes clock at $t=100$. The bridge receives an IMU message. Due to CPU load, the bridge processes it at wall time $T=1600000000$.
If the bridge publishes the ROS message with header.stamp = Wall Time but the EKF node has use_sim_time=true (listening to /clock at $t=100$), the time delta calculation fails catastrophically.
Inverse Scenario: The bridge stamps with Sim Time, but messages arrive out of order.
$Msg_1$: $t=100.01$
$Msg_2$: $t=100.005$ (Arrives late)
$\Delta t = -0.005$.
Standard EKF integration state += derivative * dt with negative dt causes the filter to integrate backwards, destabilizing the covariance matrix (resulting in negative diagonals).28
7.2 Synchronization Best Practices
To resolve ros_gz_bridge issues:
Global Sim Time: Ensure every node (Bridge, EKF, Rviz) has the parameter use_sim_time set to True.
Dedicated Clock Bridge: Run a separate bridge process just for the /clock topic to minimize latency.
Bash
ros2 run ros_gz_bridge parameter_bridge /clock@rosgraph_msgs/msg/Clock


Monotonic Checks in Code:
C++
if (msg->header.stamp <= last_imu_time) {
    ROS_WARN("Negative/Duplicate Delta T detected. Skipping.");
    return;
}


8. Conclusion and Integration Strategy
The "Velocity Explosion" in Gazebo VIO is rarely a single bug. It is a compounding of Theoretical Inconsistency (Gravity Leak due to lack of FEJ), Practical Drift (lack of ZUPT), Bias Saturation (lack of clamping), and Infrastructure Jitter (ROS 2 time).
Action Plan for Resolution:
Step 1 (Infrastructure): Fix ROS 2 time. Verify ros2 topic echo /imu matches ros2 topic echo /clock. Enforce strict monotonic timestamping in the EKF input callback.
Step 2 (Safety): Enable ZUPT with GLRT detection. Set $H_{zupt}$ to select velocity only. Implement $R_{bias}$ pseudo-measurements to clamp bias random walk.
Step 3 (Theory): Implement First-Estimate Jacobian (FEJ) for the SLAM features or MSCKF poses to stop the unobservable yaw variance from collapsing.
Step 4 (Tuning): Inflate Process Noise $Q$ for velocity and bias by 10x over physics values. Switch from Hard $\chi^2$ rejection to Huber kernels for visual updates.
By systematically addressing these layers, the ES-EKF transitions from a fragile estimator prone to "velocity explosion" into a robust navigation solution capable of bridging the gap between idealized simulation equations and the discrete, jittery reality of robot software.
Summary Table of Mitigation Techniques
Failure Mode
Root Cause
Mathematical Fix
Implementation Action
Velocity Explosion
Tilt error $\to$ Gravity Projection
Minimize $\delta \theta$ error; ZUPT.
ZUPT ($z=0$) when stationary; FEJ to fix subspace.
Gravity Leak
Spurious observability of Yaw
$H_{FEJ}$ uses frozen linearization point.
Cache first estimates for all landmark/pose Jacobians.
Reset Loop
$\chi^2$ rejection of all features
Robust Cost Function (Huber).
Replace if chi2 > thresh with Huber weight scaling.
Bias Saturation
Bias absorbs gravity error
Pseudo-measurement $0 = b + \eta$.
Add soft constraint update step for $b_a, b_g$.
Sim Instability
Negative $\Delta t$ / Jitter
Monotonic Time check.
Check use_sim_time; Buffer IMU to interpolate exact Camera time.

