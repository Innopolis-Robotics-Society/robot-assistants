from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_config = PathJoinSubstitution([
        FindPackageShare('iros_assistant_behavior'),
        'config',
        'routines.yaml',
    ])

    return LaunchDescription([
        Node(
            package='iros_assistant_behavior',
            executable='inspector',
            name='inspector',
            output='screen',
            parameters=[{
                'routines_file': default_config,
                'min_confidence': 50.0,
            }],
        ),
    ])
