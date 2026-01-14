To design a map with high landmark density that effectively eliminates **data association problems** and provides a known reference for your ES-EKF, you can implement the following strategies based on the sources:

### **1. Use Unique Signatures and Feature Descriptors**
The most effective way to eliminate data association (correspondence) ambiguity is to ensure each landmark has a **unique signature**.
*   **Unique Signatures:** Assign numerical values or categories (e.g. distinct colours) to each landmark. Your current "Vision Node" implementation already uses `landmark_id` passed through the `z` coordinate of a `Pose` message to achieve this.
*   **Serial Numbers/Descriptors:** For more complex environments, use "serial numbers" provided by feature descriptors like **ORB or BRIEF**. These descriptors allow the system to match measurements to the specific "serial number" of a landmark in your known map.

### **2. Implement Artificial Landmarks (Ceiling or Wall Markers)**
For a "known map" scenario, **artificial landmarks** (beacons or markers) are the gold standard for high-accuracy localization.
*   **Ceiling Markers:** Mounting artificial visual markers on the ceiling is a classical solution because they are easy to recognize and rarely obstructed by dynamic objects.
*   **Spatial Arrangement:** Ensure landmarks are placed **sufficiently far apart** to minimize the probability of the sensor confusing one for another, but keep them dense enough that the robot frequently encounters them to keep **pose uncertainty small**.

### **3. Solve the Observability and Rank Deficiency Problem**
Your recent "Observability Analysis" showed that observing only 1 or 2 landmarks results in a **rank-deficient measurement matrix** with a 2-dimensional nullspace. This means the filter cannot distinguish between position and orientation errors.
*   **Increase Landmark Density:** You must ensure that the camera can see **at least 3 or more landmarks** at any given time. High landmark density provides the geometric diversity needed to uncouple translation from rotation, preventing the "belief jumping" you observed [Observability Analysis file].
*   **Mutual Exclusion:** When landmarks are dense, your code should enforce the **mutual exclusion principle**, which states that two different regions in an image cannot correspond to the same physical landmark.

### **4. Design for "Coastal Navigation"**
In environments with uneven landmark distribution, the robot should follow a **coastal navigation** strategy.
*   **Active Information Gathering:** Instead of moving through open, featureless spaces where it may get lost, the robot should "hug" areas with high landmark density.
*   **Belief-Space Planning:** By planning in the **belief space**, the robot can anticipate future uncertainty and prefer longer paths that stay near known landmarks to remain localized.

### **5. Formal Mapping Requirements**
To have a "known map," you must represent it as a **feature-based map** (a list of objects and their Cartesian coordinates) rather than a location-based volumetric map.
*   **State Vector:** Your known map should define the IMU state $x_I$ (position and rotation relative to the world frame) and the fixed coordinates of every landmark $L$.
*   **Reduced Uncertainty:** Setting the world frame to the first camera frame's corresponding IMU frame allows for **zero initial pose uncertainty**, which increases the consistency of your subsequent estimates.