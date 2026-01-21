from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory, PackageNotFoundError
import os

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')

    # Get package paths
    pkg_share = get_package_share_directory('visual_inertial_nav_es_ekf')
    world_file = os.path.join(pkg_share, 'environment', 'project_world.sdf')

    # Check for optional packages
    try:
        get_package_share_directory('foxglove_bridge')
        foxglove_available = True
    except PackageNotFoundError:
        foxglove_available = False
        print("[WARN] foxglove_bridge not found - visualization bridge will not be started")

    try:
        get_package_share_directory('teleop_twist_keyboard')
        teleop_available = True
    except PackageNotFoundError:
        teleop_available = False
    # Note: teleop_twist_keyboard requires interactive terminal, so we don't auto-launch it
    # Run manually: ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r cmd_vel:=/vehicle_blue/cmd_vel
    print("\n" + "="*70)
    print("For manual control, run in a separate terminal:")
    print("  ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r cmd_vel:=/vehicle_blue/cmd_vel")
    print("="*70 + "\n")

    launch_items = [
        # =========================================================
        # 1. Launch Arguments
        # =========================================================
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation (Gazebo) clock if true'
        ),

        # =========================================================
        # 2. Ignition Gazebo Simulator
        # =========================================================
        ExecuteProcess(
            cmd=['ign', 'gazebo', '-r', world_file],
            output='screen',
            name='ignition_gazebo'
        ),

        # =========================================================
        # 3. Gazebo - ROS 2 Bridge
        # =========================================================
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='ros_gz_bridge',
            arguments=[
                # Clock
                '/clock@rosgraph_msgs/msg/Clock@gz.msgs.Clock',
                # Ground Truth Odometry (For Validation)
                '/model/vehicle_blue/odometry@nav_msgs/msg/Odometry@gz.msgs.Odometry',
                # TF (Internal model parts)
                '/model/vehicle_blue/tf@tf2_msgs/msg/TFMessage@gz.msgs.Pose_V',
                # IMU
                '/vehicle_blue/imu@sensor_msgs/msg/Imu@gz.msgs.IMU',
                # Camera
                '/vehicle_blue/camera/image@sensor_msgs/msg/Image@gz.msgs.Image',
                # Command Velocity
                '/vehicle_blue/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
            ],
            remappings=[
                ('/model/vehicle_blue/tf', '/tf'),
            ],
            output='screen'
        ),

        # =========================================================
        # 4. Static Transforms
        # =========================================================

        # Map -> World
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_map_to_world',
            arguments=['--frame-id', 'map', '--child-frame-id', 'world',
                       '--x', '0', '--y', '0', '--z', '0',
                       '--qx', '0', '--qy', '0', '--qz', '0', '--qw', '1'],
            output='screen',
        ),

        # World -> Odom
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_world_to_odom',
            arguments=['--frame-id', 'world', '--child-frame-id', 'vehicle_blue/odom',
                       '--x', '0', '--y', '0', '--z', '0',
                       '--qx', '0', '--qy', '0', '--qz', '0', '--qw', '1'],
            output='screen',
        ),

        # =========================================================
        # 5. Visual Detector
        # =========================================================
        Node(
            package='visual_inertial_nav_es_ekf',
            executable='visual_detector',
            name='visual_detector',
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen'
        ),

        # =========================================================
        # 6. ES-EKF Node
        # =========================================================
        Node(
            package='visual_inertial_nav_es_ekf',
            executable='es_ekf',
            name='es_ekf_node',
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen'
        ),

        # =========================================================
        # 7. Trajectory Monitor
        # =========================================================
        Node(
            package='visual_inertial_nav_es_ekf',
            executable='trajectory_monitor',
            name='trajectory_monitor',
            parameters=[{
                'use_sim_time': use_sim_time,
                'target_frame': 'world',
                'gt_frame': 'vehicle_blue/chassis',
                'ekf_topic': '/ekf_odom',
                'path_max_len': 5000,
            }],
            output='screen'
        ),
    ]

    # =========================================================
    # 8. Teleop requires interactive terminal - skip auto-launch
    # =========================================================

    # =========================================================
    # 9. Foxglove Bridge (optional - for visualization)
    # =========================================================
    if foxglove_available:
        launch_items.append(
            Node(
                package='foxglove_bridge',
                executable='foxglove_bridge',
                name='foxglove_bridge',
                parameters=[{
                    'port': 8765,
                    'address': '0.0.0.0',
                    'use_sim_time': use_sim_time,
                }],
                output='screen'
            )
        )

    return LaunchDescription(launch_items)
