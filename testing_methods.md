To scientifically benchmark your Visual-Inertial ES-EKF for a ground robot, you need to isolate specific error sources (bias instability, unobservable modes, linearization errors). Random driving is not a benchmark.

Here is a structured testing protocol divided by **Trajectory Type** (to test dynamics) and **Environmental Conditions** (to test robustness).

### 1. Standard Trajectory Benchmarks

These shapes are chosen not for aesthetics, but because they excite specific modes of the IMU and Camera to test observability.

#### A. The "Figure-8" (Lemniscate) - *The Gold Standard*

This is the most scientifically valuable trajectory for VIO calibration and testing.

* **Why:** It contains both Clockwise (CW) and Counter-Clockwise (CCW) turns.
* **Scientific Value:** If your Gyro Bias () is incorrect, a circle trajectory will drift infinitely in one direction. A Figure-8 forces the bias errors to cancel out or become observable because you turn both left and right. It excites the  and  accelerometers symmetrically.
* **Execution:** Drive a continuous figure-8 (~2m width). Repeat 5-10 times.
* **Success Metric:** The estimated path should stack perfectly on top of itself (minimal drift per loop).

#### B. The "Square" (Box)

* **Why:** Tests the filter's ability to handle **Transient Spikes**.
* **Scientific Value:** A square consists of long periods of low dynamics (straight lines) interrupted by sharp 90° turns (high dynamics). This tests if your covariance matrix () inflates correctly during the turn and if your measurement update can snap the heading back without "overshooting" (ringing).
* **Execution:** Drive 2m straight, stop, turn 90°, stop, repeat.
* **Success Metric:** Check the corners. Does the estimated trajectory overshoot? That indicates incorrect Process Noise ().

#### C. The "Stop-and-Go" (Jerk Test)

* **Why:** Tests **Accelerometer Bias () Observability**.
* **Scientific Value:** At constant velocity, accelerometer bias is unobservable (it looks like a tilt error). By accelerating and braking frequently, you force the accelerometer to produce signals distinct from gravity, allowing the EKF to converge on the true bias.
* **Execution:** Drive straight, but alternate between full throttle and hard braking every 2 seconds.

---

### 2. Stress Test Scenarios (Robustness)

Once the trajectories are tracking well, you introduce "Scientific Stressors" to find the breaking point.

#### Scenario 1: The "Visual Starvation" Test

* **Setup:** Run the **Figure-8** trajectory.
* **Variable:** Progressively reduce the number of ArUco markers/landmarks visible.
* *Phase A:* 50 markers (Full visibility).
* *Phase B:* 5 markers (Sparse).
* *Phase C:* 1 marker (Intermittent).


* **Hypothesis:** The covariance () should grow larger as markers decrease. If  stays small but the robot drifts, your filter is "Overconfident" (inconsistent).

#### Scenario 2: The "Kidnapped Robot" (Sensor Dropout)

* **Setup:** Drive a straight line.
* **Action:** Artificially disable the camera data stream for  seconds (e.g.,  to ), then re-enable it.
* **Scientific Goal:** Test **Dead Reckoning** and **Relocalization**.
* During the blackout, the EKF should rely purely on IMU integration.
* When the camera returns, the filter should *not* jump discontinuously (teleport). It should smoothly correct the accumulated drift.



#### Scenario 3: The "Gazebo Vibration" Test

* **Setup:** Stationary robot (0 velocity).
* **Variable:** Increase the simulator physics step max size or reduce solver iterations to induce "jitter."
* **Scientific Goal:** This tests your **ZUPT (Zero Velocity Update)** logic.
* **Success Metric:** The robot velocity should remain exactly 0.0 despite the IMU vibrating. If the position drifts while standing still, your ZUPT or noise masking is failing.

---

### 3. Quantitative Evaluation Metrics

To be scientific, you cannot just look at the plot. You must calculate these standard SLAM metrics (using `eval_node.py`):

1. **ATE (Absolute Trajectory Error):**
* Root Mean Square Error (RMSE) between estimated position and Ground Truth.
* *Goal:*  of total distance traveled.


2. **RPE (Relative Pose Error):**
* Accuracy of local motion (e.g., "How accurate is the movement over exactly 1 meter?").
* Crucial for judging IMU calibration.


3. **NEES (Normalized Estimation Error Squared):**
* The "Filter Consistency" score.
* It checks if the actual error matches the reported Covariance ().
*
* *Goal:* Average NEES should be close to the dimension of the state (e.g., 15).
* *Interpretation:*
* NEES >> 15: Filter is **Overconfident** (Optimistic). It thinks it is accurate, but it is wrong. (Dangerous).
* NEES << 15: Filter is **Underconfident** (Pessimistic). It assumes huge error, but is actually accurate. (Lazy convergence).





### Recommended Test Plan for Your Report

1. **Calibration Run:** Stationary (1 minute) to check bias convergence.
2. **Dynamics Run:** Figure-8 (3 loops) to check orientation/velocity drift.
3. **Endurance Run:** Large Square (10m x 10m) to check long-term position integration.
4. **Stress Run:** Circle trajectory with camera blackout every 5 seconds.