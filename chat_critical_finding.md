I suspect the root issue is the vision updates are "pulling" the position back to origin because the measurement Jacobian H doesn't have enough observability to distinguish position errors from orientation/bias errors.

Let me check the condition number of the measurement Jacobian when multiple landmarks are observed:

 python3 << 'EOF'
import numpy as np
from scipy.spatial.transform import Rotation as R

print("=" * 70)
print("OBSERVABILITY ANALYSIS OF VISION UPDATES")
print("=" * 70)

def skew_symmetric(v):
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

# Camera intrinsics
fx = 277.19
fy = 277.19
K = np.array([[fx, 0, 320], [0, fy, 240], [0, 0, 1]])

# Camera extrinsics
R_b_c = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
t_b_c = np.array([0.076, 0.0, 0.093])

def compute_H(lm_world, robot_pos, robot_yaw):
    """Compute 2x15 Jacobian for one landmark observation."""
    R_wb = R.from_euler('z', robot_yaw).as_matrix()

    # Transform to camera
    p_b = R_wb.T @ (lm_world - robot_pos)
    p_c = R_b_c @ (p_b - t_b_c)

    X, Y, Z = p_c
    if Z < 0.1:
        return None

    # Projection Jacobian
    J_proj = np.array([
        [fx/Z, 0, -fx*X/Z**2],
        [0, fy/Z, -fy*Y/Z**2]
    ])

    # Position Jacobian
    J_pos = -R_b_c @ R_wb.T

    # Orientation Jacobian
    J_rot = R_b_c @ skew_symmetric(p_b)

    # Assemble H (2x15)
    H = np.zeros((2, 15))
    H[:, 0:3] = J_proj @ J_pos
    H[:, 6:9] = J_proj @ J_rot

    return H

# Test scenario: robot at yaw=40°, two landmarks in view
robot_pos = np.array([0.0, 0.0, 0.0])
robot_yaw = np.radians(40)

landmarks = [
    np.array([5.0, 3.0, 0.5]),  # lm2
    np.array([2.0, 2.0, 0.5]),  # lm3
]

print(f"Robot at: {robot_pos}, yaw={np.degrees(robot_yaw):.1f}°")
print("\n--- Individual Landmark Jacobians ---")

H_list = []
for i, lm in enumerate(landmarks):
    H = compute_H(lm, robot_pos, robot_yaw)
    if H is not None:
        H_list.append(H)
        print(f"\nLandmark {i+1} at {lm}")
        print(f"H[position columns 0:3]:\n{H[:, 0:3]}")
        print(f"H[orientation columns 6:9]:\n{H[:, 6:9]}")

# Stack all Jacobians
H_stacked = np.vstack(H_list)
print("\n--- Stacked Jacobian (4x15 for 2 landmarks) ---")
print(f"H shape: {H_stacked.shape}")

# Extract position and orientation columns
H_pos = H_stacked[:, 0:3]  # 4x3
H_ori = H_stacked[:, 6:9]  # 4x3
H_pos_ori = H_stacked[:, [0,1,2,6,7,8]]  # 4x6 (pos + ori)

print(f"\nH_position (4x3) rank: {np.linalg.matrix_rank(H_pos)}")
print(f"H_orientation (4x3) rank: {np.linalg.matrix_rank(H_ori)}")
print(f"H_pos+ori (4x6) rank: {np.linalg.matrix_rank(H_pos_ori)}")

# Check condition number
if np.linalg.matrix_rank(H_pos_ori) >= 6:
    _, s, _ = np.linalg.svd(H_pos_ori)
    cond = s[0] / s[-1]
    print(f"Condition number of H_pos_ori: {cond:.2f}")
else:
    _, s, _ = np.linalg.svd(H_pos_ori)
    print(f"Singular values: {s}")
    print("RANK DEFICIENT - cannot fully observe position+orientation!")

# Critical: Check if position X-Y is observable separately from orientation
print("\n" + "=" * 70)
print("KEY INSIGHT: Position-Orientation Coupling")
print("=" * 70)

# The observability depends on landmark geometry
# With landmarks all at similar bearings, position error looks like orientation error

# Check nullspace of H
_, s, Vh = np.linalg.svd(H_pos_ori)
print(f"\nSingular values: {s}")

# Nullspace directions (if any)
nullspace_dim = 6 - np.sum(s > 1e-10)
if nullspace_dim > 0:
    print(f"\nNullspace dimension: {nullspace_dim}")
    print("Unobservable directions in [dx, dy, dz, drx, dry, drz]:")
    for i in range(nullspace_dim):
        null_vec = Vh[-(i+1), :]
        print(f"  {null_vec}")
else:
    print("\nNo nullspace - position and orientation are fully observable")
    print("But poor conditioning can still cause slow convergence")

print("\n" + "=" * 70)
print("The real issue: Even with full observability, the EKF converges slowly")
print("when position and orientation Jacobians are nearly parallel.")
print("=" * 70)

EOF
CRITICAL FINDING! The system has a rank-deficient measurement matrix with a 2-dimensional nullspace. This means:

The filter cannot fully distinguish between position errors and orientation errors with only 2 landmarks in view
The nullspace direction [0.48, 0.86, 0.06, 0.11, -0.02, -0.08] means: a combination of moving in X-Y direction AND rotating can produce zero pixel residual
This is a fundamental observability issue, not a bug! The filter can explain vision residuals by adjusting orientation instead of position, which is exactly what's happening.

However, there's still the question of why position isn't being updated during IMU integration. Let me look more closely at what happens during motion:




Summary of Findings
I've verified all the transforms and Jacobians - they are mathematically correct. The issues are caused by:

Root Causes Identified:
Poor Observability with 1-2 Landmarks: The system has a rank-deficient measurement matrix (rank 4 vs needed 6) when observing only 2 landmarks. This means the filter cannot distinguish position errors from orientation errors. The nullspace analysis showed two unobservable directions.

Orientation-Bias Coupling: When the filter has an orientation error:

Gravity "leaks" into world XY acceleration (~0.34 m/s² per 2° tilt)
The filter tries to explain this by increasing accelerometer bias
But bias is in body frame, while the leak is in world frame
When the robot turns, the bias that "worked" at one yaw breaks at another
This creates a positive feedback loop leading to divergence
ZUPT Not Activating When Needed: The ZUPT thresholds require both low gyro AND low accel deviation. But when orientation error causes gravity leakage, the accelerometer shows deviation even when stationary, so ZUPT doesn't trigger.

Bias Saturation: Once bias reaches ±0.5 m/s² limit, the filter cannot compensate further, leading to velocity divergence (10 m/s clipping).