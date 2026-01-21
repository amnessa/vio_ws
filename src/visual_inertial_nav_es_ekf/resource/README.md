# Visual Inertial Navigation ES-EKF

## Quick Start (Single Command)

```bash
colcon build --packages-select visual_inertial_nav_es_ekf
source install/setup.bash
ros2 launch visual_inertial_nav_es_ekf full_system.launch.py
```

This launches everything automatically:
- Ignition Gazebo with the project world
- ROS-Gazebo bridge
- ES-EKF visual-inertial navigation
- Visual detector
- Trajectory monitor
- Teleop keyboard (opens in xterm)
- Foxglove bridge (port 8765)

## Manual Launch (Separate Terminals)

If you prefer running components separately:

