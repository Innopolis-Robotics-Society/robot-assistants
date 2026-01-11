from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    pca_cfg = PathJoinSubstitution([FindPackageShare("iros_multiview_pointcloud_scan"), "config", "default.yaml"])

    pointcloud_accumulator = Node(
        package="iros_multiview_pointcloud_scan",
        executable="pointcloud_accumulator",
        name="pointcloud_accumulator",
        output="screen",
        parameters=[
            {
                pca_cfg,
            },
        ],
    )

    pc_capture = Node(
        package="iros_multiview_pointcloud_scan",
        executable="pc_capture",
        name="pc_capture",
        output="screen",
        parameters=[],
    )

    return LaunchDescription([
        pointcloud_accumulator,
        pc_capture,
    ])
