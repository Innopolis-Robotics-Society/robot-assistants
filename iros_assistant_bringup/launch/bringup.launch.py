from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from pathlib import Path

def generate_launch_description():

    ur_cfg = PathJoinSubstitution([FindPackageShare("iros_assistant_bringup"), "config", "UR10e-1.yaml"])
    rviz_cfg = PathJoinSubstitution([FindPackageShare("iros_assistant_bringup"), "rviz", "ur_moveit.rviz"])
    robot_ip_arg = DeclareLaunchArgument(
        'robot_ip',
        default_value='192.168.0.100')
    ur_model_arg = DeclareLaunchArgument(
        'ur_model',
        default_value='ur10e')

    robot_ip = LaunchConfiguration('robot_ip')
    ur_model = LaunchConfiguration('ur_model')

    ur_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                PathJoinSubstitution(
                    [
                        FindPackageShare("iros_assistant_bringup"), "launch", "robots", "ur.launch.py",
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

    move_tools = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                PathJoinSubstitution(
                    [
                        FindPackageShare("iros_assistant_bringup"), "launch", "tools", "move_tools.launch.py",
                    ]
                )
            ]
        ),
        launch_arguments=[]
    )

    voice = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                PathJoinSubstitution(
                    [
                        FindPackageShare("iros_assistant_bringup"), "launch", "tools", "voice.launch.py",
                    ]
                )
            ]
        ),
        launch_arguments=[]
    )

    assistant_cv = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                PathJoinSubstitution(
                    [
                        FindPackageShare("iros_assistant_bringup"), "launch", "perception", "assistant_cv.launch.py",
                    ]
                )
            ]
        ),
        launch_arguments=[]
    )

    voice_beh = Node(
            package='iros_assistant_behavior',
            executable='voice_beh',
        )

    tf = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                PathJoinSubstitution(
                    [
                        FindPackageShare("iros_assistant_bringup"), "launch", "world", "poses.launch.py",
                    ]
                )
            ]
        ),
        launch_arguments=[
        ]
    )



    return LaunchDescription([
        ur_model_arg,
        robot_ip_arg,
        tf,
        ur_launch,
        move_tools,
        voice,
        assistant_cv,
        voice_beh,
    ])