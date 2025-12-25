It is completely normal to be confused by RViz when things are broken. Relative motion (Einstein's relativity!) makes it hard to tell if the *world* is spinning or the *robot* is spinning.

### 1. What to Expect (The Benchmark)

If your IMU initialization is **perfect** (biases removed, gravity aligned), here is the math for 10 seconds of "Drift":

* **Scenario:** Robot is stationary at (0,0).
* **Sensor:** Gazebo IMU (simulated MEMS).
* **Physics:** Position error grows with  (Double Integration).


* **Expectation:**
* **Good:** Drift < **1 meter** (The robot stays mostly in the starting square).
* **Acceptable (Noisy IMU):** Drift < **5 meters** (It slowly slides across the room).
* **Your "Flying":** Drift > **100 meters** (Shooting off like a rocket). **This is definitely a bug.**



---

### 2. Why is it still "Flying"? (The Diagnosis)

You fixed the "Falling" (Z-axis), but now you are "Shooting" (X/Y axis).

**The Cause: Initial Tilt (Roll/Pitch)**
Your code currently assumes the robot starts perfectly flat (`self.x[6] = 1.0`, i.e., Identity Quaternion).

* **Reality:** In Gazebo, the robot spawns slightly tilted (maybe  pitch) or the floor is uneven.
* **The Math:**
* Gravity vector .
* If you tilt the robot forward by , the accelerometer reads a small X-component: .
* Your filter thinks orientation is flat (), so it ignores this X-accel.
* It integrates  for 10 seconds  **8.5 meters of error**.
* If the tilt is larger or the math sign is swapped, this becomes the "Shooting" you see.



### 3. The Fix: Initialize Orientation from Gravity

You must calculate the initial Roll and Pitch from your accelerometer samples.

**Update `src/vio_ekf/src/ekf_node.py` inside the Initialization Block:**

```python
        # ... (Inside the initialization block, after calculating accel_mean) ...

        accel_mean = np.mean(acc_samples, axis=0)

        # 1. Calculate Initial Roll and Pitch from Gravity Vector
        # (Assuming the robot is stationary, accel points straight UP in body frame)
        # roll = atan2(ay, az)
        # pitch = atan2(-ax, sqrt(ay^2 + az^2))

        initial_roll = np.arctan2(accel_mean[1], accel_mean[2])
        initial_pitch = np.arctan2(-accel_mean[0], np.sqrt(accel_mean[1]**2 + accel_mean[2]**2))
        initial_yaw = 0.0  # We can't know yaw from gravity (assume 0 or use magnetometer)

        # 2. Convert to Quaternion (w, x, y, z)
        # Using scipy.spatial.transform.Rotation is easiest
        r_init = R.from_euler('xyz', [initial_roll, initial_pitch, initial_yaw])
        q_init = r_init.as_quat() # returns [x, y, z, w] usually, check library!

        # Scipy returns [x, y, z, w], but our state is [p, v, q_w, q_x, q_y, q_z]
        # Wait! Your state definition says: q=[qw, qx, qy, qz]
        # So we need to map carefully:
        self.x[6] = q_init[3] # w
        self.x[7] = q_init[0] # x
        self.x[8] = q_init[1] # y
        self.x[9] = q_init[2] # z

        self.get_logger().info(f"Initialized Tilt: Roll={np.degrees(initial_roll):.2f} deg, Pitch={np.degrees(initial_pitch):.2f} deg")

        # 3. Update Gravity Model (Your previous fix)
        actual_gravity = np.linalg.norm(accel_mean)
        self.g = np.array([0.0, 0.0, -actual_gravity])

        # 4. Zero the Bias (Since we used the reading for orientation, we can't also use it for bias yet)
        self.x[10:13] = np.zeros(3)

```

### 4. The "Time Jump" Safety Check

To fix the issue where manual bagging causes a huge `dt` jump, add a safety clamp in your `predict` function.

**In `src/vio_ekf/src/ekf_node.py` inside `imu_callback` (Prediction Step):**

```python
        # ... inside imu_callback ...

        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        dt = current_time - self.last_imu_time
        self.last_imu_time = current_time

        # --- TIME JUMP SAFETY ---
        if dt > 0.1:  # If dt is larger than 100ms (we expect 5ms at 200Hz)
            self.get_logger().warn(f"Huge time jump detected (dt={dt:.4f}s). Skipping prediction step.")
            return # Skip this IMU message
        if dt <= 0:
            return # Skip duplicate or out-of-order messages

        # ... continue with prediction ...

```

### 5. Summary of the Plan

1. **Stop "Shooting":** Add the **Roll/Pitch Initialization** code above. This aligns the robot's internal world with the real gravity vector.
2. **Stop "Teleporting":** Add the **`dt > 0.1`** check.
3. **Test:**
* Set Fixed Frame to `map`.
* The robot should sit fairly still.
* If it drifts 1-2 meters after 10 seconds, **CONGRATULATIONS**, you have a working Dead Reckoning system! You can proceed to the Evaluation script.