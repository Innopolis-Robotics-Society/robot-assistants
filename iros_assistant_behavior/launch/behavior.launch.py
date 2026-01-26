from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    routines_default = PathJoinSubstitution([
        FindPackageShare("iros_assistant_behavior"),
        "config",
        "routines.yaml",
    ])

    return LaunchDescription([
        DeclareLaunchArgument("routines_file", default_value=routines_default),
        DeclareLaunchArgument("min_confidence", default_value="50.0"),
        DeclareLaunchArgument("post_step_delay_sec", default_value="0.2"),

        Node(
            package="iros_assistant_behavior",
            executable="inspector",
            name="inspector",
            output="screen",
            parameters=[{
                "routines_file": LaunchConfiguration("routines_file"),
                "min_confidence": LaunchConfiguration("min_confidence"),
                "post_step_delay_sec": LaunchConfiguration("post_step_delay_sec"),
                "command_topic": "/voice/command",
                "default_service_timeout_sec": 60.0,
                "abort_on_failure": True,
            }],
        ),
    ])
