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
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')

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

    # 1. Start Ignition Gazebo with our world
    sdf_path = os.path.join(pkg_vio, 'worlds', 'landmarks.sdf')

    # Use gz or ign depending on availability (like reference file)
    gz_cmd = ['gz', 'sim', '-r', sdf_path] if shutil.which('gz') \
        else ['ign', 'gazebo', '-r', sdf_path, '--force-version', '6']
    gz_sim = ExecuteProcess(cmd=gz_cmd, output='screen')

    # 2. Bridge ROS2 <-> Ignition (using @ for bidirectional like reference)
    bridge_args = [
        # IMU (GZ -> ROS)
        '/imu@sensor_msgs/msg/Imu@ignition.msgs.IMU',
        # Camera image and camera_info (GZ -> ROS)
        '/camera@sensor_msgs/msg/Image@ignition.msgs.Image',
        '/camera_info@sensor_msgs/msg/CameraInfo@ignition.msgs.CameraInfo',
        # Ground-truth poses (GZ -> ROS) - Bridge Pose_V to TFMessage
        '/model/vio_robot/pose@tf2_msgs/msg/TFMessage@ignition.msgs.Pose_V',
        '/model/vio_robot/pose_static@tf2_msgs/msg/TFMessage@ignition.msgs.Pose_V',
        # Clock (GZ -> ROS) - required for use_sim_time
        '/clock@rosgraph_msgs/msg/Clock@ignition.msgs.Clock',
    ]
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=bridge_args,
        output='screen',
    )

    # 3. Ground Truth Publisher (Our Custom Node)
    # Subscribes to TFMessage on 'pose' and 'pose_static' topics and broadcasts to /tf
    ground_truth = Node(
        package='vio_ekf',
        executable='pose_tf_broadcaster',
        name='pose_tf_broadcaster',
        output='screen',
        remappings=[
            ('pose', '/model/vio_robot/pose'),
            ('pose_static', '/model/vio_robot/pose_static'),
        ]
    )

    # 4. Static TF: world -> vio_robot (connects world to model root frame)
    # Using --frame-id and --child-frame-id for proper static publisher
    tf_world_to_model = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['--x', '0', '--y', '0', '--z', '0',
                   '--roll', '0', '--pitch', '0', '--yaw', '0',
                   '--frame-id', 'world', '--child-frame-id', 'vio_robot'],
        output='screen'
    )

    # 5. Rviz2
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        # arguments=['-d', os.path.join(pkg_vio, 'rviz', 'vio.rviz')], # Enable once we save a config
        parameters=[{'use_sim_time': True}],
        output='screen'
    )

    return LaunchDescription([
        gz_resource_path,
        ign_resource_path,
        gz_sim,
        bridge,
        ground_truth,
        tf_world_to_model,
        rviz
    ])