from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder
from pathlib import Path
import os
def generate_launch_description():

    ur_cfg = PathJoinSubstitution([FindPackageShare("ur_assist"), "config", "UR10e-1.yaml"])
    rviz_cfg = PathJoinSubstitution([FindPackageShare("ur_assist"), "rviz", "ur_moveit.rviz"])
    robot_ip_arg = DeclareLaunchArgument(
        'robot_ip',
        default_value='192.168.0.100')
    ur_model_arg = DeclareLaunchArgument(
        'ur_model',
        default_value='ur10e')


    robot_ip = LaunchConfiguration('robot_ip')
    ur_model = LaunchConfiguration('ur_model')
    rviz_config_file = PathJoinSubstitution(
        [FindPackageShare("ur_moveit_config"), "config", "moveit.rviz"]
    )

    ur_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                PathJoinSubstitution(
                    [
                        FindPackageShare("ur_robot_driver"),
                        "launch",
                        "ur_control.launch.py",
                    ]
                )
            ]
        ),
        launch_arguments=[
            ('ur_type', ur_model),
            ('robot_ip', robot_ip),
            ('kinematics_params_file', ur_cfg),
            ('launch_rviz', 'false'),
            ('headless_mode','true'),
            ('use_tool_communication','false'),
        ]
    )

    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                PathJoinSubstitution(
                    [
                        FindPackageShare("ur_moveit_config"),
                        "launch",
                        "ur_moveit.launch.py",
                    ]
                )
            ]
        ),
        launch_arguments=[
            ('ur_type', ur_model),
            ('launch_rviz','true'),
        ]
    )

    return LaunchDescription([
        ur_model_arg,
        robot_ip_arg,
        ur_launch,
        moveit_launch,
        #rviz
    ])