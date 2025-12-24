import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
import shutil

def generate_launch_description():
    pkg_vio = get_package_share_directory('vio_ekf')
    #pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')

    # Set GZ_SIM_RESOURCE_PATH so Gazebo can find our models
    models_path = os.path.join(pkg_vio, 'models')
    gz_resource_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=models_path + ':' + os.environ.get('GZ_SIM_RESOURCE_PATH', '')
    )
    # Also set IGN_GAZEBO_RESOURCE_PATH for older Ignition versions
    ign_resource_path = SetEnvironmentVariable(
        name='IGN_GAZEBO_RESOURCE_PATH',
        value=models_path + ':' + os.environ.get('IGN_GAZEBO_RESOURCE_PATH', '')
    )

    #  Start Ignition Gazebo with our world
    sdf_path = os.path.join(pkg_vio, 'worlds', 'landmarks.sdf')

    # Use gz or ign depending on availability (like reference file)
    gz_cmd = ['gz', 'sim', '-r', sdf_path] if shutil.which('gz') \
        else ['ign', 'gazebo', '-r', sdf_path, '--force-version', '6']
    gz_sim = ExecuteProcess(cmd=gz_cmd, output='screen')

    #  Bridge ROS2 <-> Ignition (using @ for bidirectional like reference)
    bridge_args = [
        # IMU (GZ -> ROS)
        '/imu@sensor_msgs/msg/Imu@ignition.msgs.IMU',
        # Camera image and camera_info (GZ -> ROS)
        '/camera@sensor_msgs/msg/Image@ignition.msgs.Image',
        '/camera_info@sensor_msgs/msg/CameraInfo@ignition.msgs.CameraInfo',
        # Odometry (GZ -> ROS) - wheel odometry from diff_drive
        '/odom@nav_msgs/msg/Odometry@ignition.msgs.Odometry',
        # TF from diff_drive (odom -> base_footprint)
        '/tf@tf2_msgs/msg/TFMessage@ignition.msgs.Pose_V',
        # Ground truth: world dynamic pose contains all model poses in world frame
        '/world/vio_world/dynamic_pose/info@tf2_msgs/msg/TFMessage@ignition.msgs.Pose_V',
        # Cmd vel (ROS -> GZ)
        '/cmd_vel@geometry_msgs/msg/Twist@ignition.msgs.Twist',
        # Clock (GZ -> ROS) - required for use_sim_time
        '/clock@rosgraph_msgs/msg/Clock@ignition.msgs.Clock',
        # Joint states for robot_state_publisher
        '/joint_states@sensor_msgs/msg/JointState@ignition.msgs.Model',
    ]
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=bridge_args,
        output='screen',
    )

    #  Ground Truth Publisher (Our Custom Node)
    # Subscribes to world pose info and extracts robot's ground truth pose
    # Publishes to /ground_truth/odom and broadcasts TF
    ground_truth = Node(
        package='vio_ekf',
        executable='pose_tf_broadcaster',
        name='pose_tf_broadcaster',
        output='screen',
        parameters=[{'use_sim_time': True}],
        remappings=[
            ('pose', '/world/vio_world/dynamic_pose/info'),
        ]
    )

    #  Static TF: world -> map (they are the same in our simulation)
    tf_world_to_map = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_world_map',
        arguments=['--x', '0', '--y', '0', '--z', '0',
                   '--roll', '0', '--pitch', '0', '--yaw', '0',
                   '--frame-id', 'world', '--child-frame-id', 'map'],
        parameters=[{'use_sim_time': True}],
        output='screen'
    )

    # Static TF: world -> odom (for odometry messages from Gazebo diff_drive)
    tf_world_to_odom = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_world_odom',
        arguments=['--x', '0', '--y', '0', '--z', '0',
                   '--roll', '0', '--pitch', '0', '--yaw', '0',
                   '--frame-id', 'world', '--child-frame-id', 'odom'],
        parameters=[{'use_sim_time': True}],
        output='screen'
    )

    # Static TF from camera_link -> camera sensor frame (identity transform)
    cam_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0',
                   'turtlebot3/camera_link', 'turtlebot3/camera_link/camera'],
        parameters=[{'use_sim_time': True}],
        output='screen'
    )

    # Robot State Publisher - publishes robot_description for RViz visualization
    # Publishes TFs for robot links connected to base_footprint
    # The odom->base_footprint transform comes from Gazebo diff_drive plugin
    urdf_file = os.path.join(pkg_vio, 'urdf', 'turtlebot3_waffle_pi.urdf')
    with open(urdf_file, 'r') as f:
        robot_desc = f.read()

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'robot_description': robot_desc,
        }],
    )

    #  Rviz2
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', os.path.join(pkg_vio, 'rviz', 'vio.rviz')],
        parameters=[{'use_sim_time': True}],
        output='screen'
    )

    # Vision Frontend Node
    vision_node = Node(
        package='vio_ekf',
        executable='vision_node.py',
        name='vision_node',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    ekf_node = Node(
        package='vio_ekf',
        executable='ekf_node.py',
        name='ekf_node',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    return LaunchDescription([
        gz_resource_path,
        ign_resource_path,
        gz_sim,
        bridge,
        tf_world_to_map,
        tf_world_to_odom,
        robot_state_publisher,
        ekf_node,
        vision_node,
        ground_truth,
        cam_tf,
        rviz
    ])