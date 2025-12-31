from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from pathlib import Path
import os
def generate_launch_description():
    voice_rec = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                PathJoinSubstitution(
                    [
                        FindPackageShare("iros_ru_voice_recognision"),
                        "launch",
                        "voice_control.launch.py",
                    ]
                )
            ]
        ),
    )

    return LaunchDescription([
        voice_rec,
    ])