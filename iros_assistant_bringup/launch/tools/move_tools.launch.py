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

    go_to_frame = Node(
        package="iros_assistant_bringup",
        executable="go_to_frame",
        output="screen",
        parameters=[
            {
            },
        ],
    )

    exec_ik_move = Node(
        package="iros_assistant_bringup",
        executable="exec_ik_move",
        output="screen",
        parameters=[
            {
            },
        ],
    )

    gripper_control = Node(
        package="iros_assistant_bringup",
        executable="gripper_control",
        output="screen",
        parameters=[
            {
            },
        ],
    )

    water_proc = Node(
        package="iros_ru_voice_recognision",
        executable="water_proc",
        output="screen",
        parameters=[
            {
            },
        ],
    )

    return LaunchDescription([
        go_to_frame,
        exec_ik_move,
        gripper_control,
        #TO-DO: transfer to assistant behavior
        water_proc
    ])