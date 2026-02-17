from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    host = LaunchConfiguration("host")
    port = LaunchConfiguration("port")

    return LaunchDescription([
        DeclareLaunchArgument("host", default_value="0.0.0.0"),
        DeclareLaunchArgument("port", default_value="8000"),
        Node(
            package="iros_rest_api",
            executable="api_server",
            name="iros_rest_api_server",
            output="screen",
            arguments=["--host", host, "--port", port],
        ),
    ])
