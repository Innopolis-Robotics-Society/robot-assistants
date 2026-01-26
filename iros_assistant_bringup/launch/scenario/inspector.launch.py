from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node


def generate_launch_description():

    ur_cfg = PathJoinSubstitution([FindPackageShare("iros_assistant_bringup"), "config", "UR10e-1.yaml"])
    rviz_cfg = PathJoinSubstitution([FindPackageShare("iros_assistant_bringup"), "rviz", "inspector.rviz"])

    video_device_arg = DeclareLaunchArgument(
        "video_device",
        default_value="/dev/video0",
    )

    robot_ip_arg = DeclareLaunchArgument(
        "robot_ip",
        default_value="192.168.0.15",
    )

    # Replaces ur_model -> robot_model (common for all robot types)
    robot_model_arg = DeclareLaunchArgument(
        "robot_model",
        default_value="robopro",  # e.g. ur10e / ur5e / ur3e / robopro
    )

    video_device = LaunchConfiguration("video_device")
    robot_ip = LaunchConfiguration("robot_ip")
    robot_model = LaunchConfiguration("robot_model")

    is_ur = PythonExpression([
        "'",
        robot_model,
        "' in ['ur3e','ur5e','ur10e','ur16e','ur20','ur30']"
    ])

    ur_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                PathJoinSubstitution(
                    [
                        FindPackageShare("iros_assistant_bringup"),
                        "launch",
                        "robots",
                        "ur.launch.py",
                    ]
                )
            ]
        ),
        condition=IfCondition(is_ur),
        launch_arguments=[
            ("ur_type", robot_model),     # robot_model is used as UR type when UR is selected
            ("robot_ip", robot_ip),
            ("kinematics_params_file", ur_cfg),
            ("launch_rviz", "false"),
            ("headless_mode", "true"),
            ("use_tool_communication", "false"),
        ],
    )

    # New manipulator controller (when robot_model is NOT UR*)
    robopro_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                PathJoinSubstitution(
                    [
                        FindPackageShare("iros_robopro_controller"),
                        "launch",
                        "driver.launch.py",
                    ]
                )
            ]
        ),
        condition=UnlessCondition(is_ur),
        launch_arguments=[
        ],
    )

    camera = Node(
        package="v4l2_camera",
        executable="v4l2_camera_node",
        parameters=[{"video_device": video_device}],
    )

    tf = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                PathJoinSubstitution(
                    [
                        FindPackageShare("iros_assistant_bringup"),
                        "launch",
                        "world",
                        "inspection_poses.launch.py",
                    ]
                )
            ]
        ),
        launch_arguments=[],
    )



    cv_pipeline = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                PathJoinSubstitution(
                    [
                        FindPackageShare("iros_cv_session_hub"),
                        "launch",
                        "cv_stack.launch.py",
                    ]
                )
            ]
        ),
        launch_arguments=[
            ("image_topic", "/image_raw"),
        ],
    )

    voice_rec = Node(
        package="iros_voice_controlled_robot",
        executable="voice_controller",
        #arguments=["-d", rviz_cfg],
    )

    behavior = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                PathJoinSubstitution(
                    [
                        FindPackageShare("iros_assistant_behavior"),
                        "launch",
                        "behavior.launch.py",
                    ]
                )
            ]
        ),
        launch_arguments=[],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", rviz_cfg],
    )

    return LaunchDescription(
        [
            robot_model_arg,
            robot_ip_arg,
            video_device_arg,
            tf,
            camera,
            ur_launch,
            robopro_launch,
            cv_pipeline,
            voice_rec,
            behavior,
            rviz,
        ]
    )
