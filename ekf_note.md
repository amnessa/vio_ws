# Visual-Inertial Extended Kalman Filter Navigation (VIO-EKF)

## Project Overview
This repository contains the implementation of a **Visual-Inertial Navigation System (VINS)** for the EE585 Probabilistic Robotics course. The system utilizes an **Error-State Extended Kalman Filter (ES-EKF)** to estimate the 6-DoF state (position, velocity, orientation) of a ground vehicle in a simulated Gazebo Fortress environment.

The filter fuses high-frequency data from an Inertial Measurement Unit (IMU) with low-frequency landmark observations from a monocular camera to bound the drift inherent in dead-reckoning navigation.

## Authors
* **Çağdaş Güven**
* **Ege Uğur Aguş**
* **Umut Can Gülmez**

## System Architecture: Error-State EKF (ES-EKF)
Unlike a standard EKF, this implementation splits the state into a **Nominal State** (handling large values and non-linearities) and an **Error State** (handling noise and small perturbations).

### 1. State Vector Definitions
* **Nominal State ($x_{nom} \in \mathbb{R}^{16}$):**
    $$x_{nom} = [p_{wb}, v_{wb}^w, q_{wb}, b_a, b_g]$$
    * $p$: Position ($3 \times 1$)
    * $v$: Velocity ($3 \times 1$)
    * $q$: Unit Quaternion ($4 \times 1$)
    * $b_a$: Accelerometer Bias ($3 \times 1$)
    * $b_g$: Gyroscope Bias ($3 \times 1$)

* **Error State ($\delta x \in \mathbb{R}^{15}$):**
    $$\delta x = [\delta p, \delta v, \delta \theta, \delta b_a, \delta b_g]$$
    * Note: Orientation error $\delta \theta$ is a $3 \times 1$ angle vector, avoiding covariance rank deficiency issues associated with 4-parameter quaternions.

### 2. Filter Process
1.  **Prediction (IMU Propagation):**
    * Propagate $x_{nom}$ using non-linear kinematics with IMU measurements ($a_m, \omega_m$).
    * Propagate error covariance $P$ using the Jacobian of the error state transition function $F_x$.

2.  **Measurement Update (Visual Correction):**
    * **Residual:** Computed in the image plane between observed pixels and projected nominal state.
    * **Kalman Gain ($K$):** Calculated w.r.t the error state covariance.
    * **Error Estimation:** $\delta \hat{x} = K \cdot r$.

3.  **Injection & Reset:**
    * The estimated error $\delta \hat{x}$ is injected into the nominal state (additive for $p, v, b$; multiplicative for $q$).
    * The error state $\delta x$ is reset to zero, and covariance $P$ is updated.
## Dependencies
* ROS 2 Humble
* Gazebo Fortress
* `vio_ekf` (Custom Package)