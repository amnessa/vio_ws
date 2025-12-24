



### 2. Task: Evaluation & Metrics (Shared Role)

**Status:** `ekf_node.py` publishes `/vio/path` and `/vio/odom`.


**Needs Creation:** The report explicitly requires **plots of pose estimates vs. ground truth** and **RMSE calculation**.

**Action:** You should create a separate Python script (e.g., `evaluation_node.py`) or a Jupyter Notebook that:

1. Subscribes to `/vio/odom` (Your Estimate).
2. Subscribes to `/ground_truth` (Gazebo Truth, often `/gazebo/model_states` or a specific TF `map -> base_footprint`).
3. Synchronizes these two streams (e.g., using `message_filters.ApproximateTimeSynchronizer`).
4. Calculates **RMSE** for Position () and Orientation.
5. Generates the **Error vs. Distance** plots mentioned in Section IV.



