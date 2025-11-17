from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
import os

def generate_launch_description():

    config_path = os.path.join(
        get_package_share_directory('rlr_camera'),
        'config',
        'param.yaml'
    )

    calibration = os.path.join(
        get_package_share_directory('rlr_camera'),
        'calibration',
        'ost.yaml'
    )

    image_source = DeclareLaunchArgument(
        "image_source",
        default_value="/dev/video0",
        description="Maf-maf",
    )

    camera_name = DeclareLaunchArgument(
        "camera_name",
        default_value="maf_camera",
        description="Maf-maf-maf",
    )

    camera_node=Node(
	    package='v4l2_camera',
	    executable='v4l2_camera_node',
	    name='v4l2_camera',
	    parameters=[
            {'camera_name': LaunchConfiguration("camera_name")},
            {'video_device': LaunchConfiguration("image_source")},
            {'camera_info_url': f'file://{calibration}'},
            config_path],
	    output='screen'
    )

    tf = Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=['0', '0', '0', '0', '0', '0', 'base_link', 'camera']
        )

    return LaunchDescription([
    camera_name,
    image_source,
	camera_node,
    tf
    ])
