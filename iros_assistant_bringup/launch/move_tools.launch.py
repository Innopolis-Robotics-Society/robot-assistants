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
        package="iros_cv",
        executable="ins_search",
        output="screen",
        parameters=[
            {
                'model_path': '/home/mobile/ros2_ws/src/iros_cv/models/best_fixed.pt',
                "image_topic": "/image_rect",
            },
        ],
    )

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

    rlr_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                PathJoinSubstitution(
                    [
                        FindPackageShare("iros_assistant_bringup"),
                        "launch",
                        "bringup.launch.py",
                    ]
                )
            ]
        ),
        launch_arguments=[
        ]
    )

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
        ins_search,
        go_to_frame,
        exec_ik_move,
        gripper_control,
        rlr_bringup,
        voice_rec,
        water_proc
    ])