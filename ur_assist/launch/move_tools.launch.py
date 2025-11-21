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


    ins_search = Node(
        package="rlr_cv",
        executable="ins_search",
        output="screen",
        parameters=[
            {
                'model_path': '/home/mobile/ros2_ws/src/rlr_cv/models/best_fixed.pt',
                "image_topic": "/image_rect",
            },
        ],
    )

    go_to_frame = Node(
        package="ur_assist",
        executable="go_to_frame",
        output="screen",
        parameters=[
            {
            },
        ],
    )

    exec_ik_move = Node(
        package="ur_assist",
        executable="exec_ik_move",
        output="screen",
        parameters=[
            {
            },
        ],
    )

    return LaunchDescription([
        ins_search,
        go_to_frame,
        exec_ik_move,
    ])