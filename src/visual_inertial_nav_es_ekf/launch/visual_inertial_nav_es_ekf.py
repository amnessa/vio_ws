from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')

    return LaunchDescription([
        # 1. Launch Arguments
        DeclareLaunchArgument(
            'use_sim_time', 
            default_value='true',
            description='Use simulation (Gazebo) clock if true'
        ),

        # 2. Gazebo - ROS 2 Bridge
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

        # ---------------------------------------------------------
        # 3. Static Transforms (THE FIX IS HERE)
        # ---------------------------------------------------------
        
        # 3.1 Map -> World
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_map_to_world',
            arguments=['0','0','0',  '0','0','0','1',  'map', 'world'],
            output='screen',
        ),

        # 3.2 World -> Odom (The Missing Link!)
        # This connects the two TF trees together.
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_world_to_odom',
            arguments=['0','0','0',  '0','0','0','1',  'world', 'vehicle_blue/odom'],
            output='screen',
        ),

        # ---------------------------------------------------------

        # 4. Vehicle Driver (Disabled for Teleop)
        # Uncomment this block if you want automatic driving again
        # Node(
        #     package='visual_inertial_nav_es_ekf',
        #     executable='vehicle_driver',
        #     name='vehicle_driver',
        #     parameters=[{
        #         'use_sim_time': use_sim_time,
        #         'cmd_topic': '/vehicle_blue/cmd_vel',
        #         'linear': 0.8,
        #         'omega_amp': 0.5,
        #         'omega_freq': 0.15,
        #     }],
        #     output='screen'
        # ),

        # 5. Visual Detector
        Node(
            package='visual_inertial_nav_es_ekf',
            executable='visual_detector',
            name='visual_detector',
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen'
        ),

        # 6. ES-EKF
        Node(
            package='visual_inertial_nav_es_ekf',
            executable='es_ekf',
            name='es_ekf_node',
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen'
        ),

        # 7. Trajectory Monitor
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
    ])