from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_config = PathJoinSubstitution([
        FindPackageShare('iros_robopro_controller'),
        'config',
        'driver.yaml',
    ])

    config = LaunchConfiguration('config')

    return LaunchDescription([
        DeclareLaunchArgument(
            'config',
            default_value=default_config,
            description='Path to YAML config file for rc_robot_driver',
        ),
        Node(
            package='iros_robopro_controller',
            executable='rc_robot_driver',
            name='rc_robot_driver',
            output='screen',
            parameters=[config],
        ),
    ])
