from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    icp_cfg = PathJoinSubstitution([FindPackageShare("iros_icp_mesh_align"), "config", "default.yaml"])

    icp_server = Node(
        package="iros_icp_mesh_align",
        executable="icp_server",
        name="icp_server",
        output="screen",
        parameters=[
            {
                icp_cfg,
            },
        ],
    )

    return LaunchDescription([
        icp_server,
    ])
