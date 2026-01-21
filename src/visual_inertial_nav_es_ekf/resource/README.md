colcon build --packages-select visual_inertial_nav_es_ekf

ros2 launch visual_inertial_nav_es_ekf visual_inertial_nav_es_ekf.py

ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r cmd_vel:=/vehicle_blue/cmd_vel

ign gazebo -r project_world.sdf

ros2 launch foxglove_bridge foxglove_birdge_launch.xml

