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

    robot_ip_arg = DeclareLaunchArgument(
        "robot_ip",
        default_value="192.168.0.15",
    )

    # Replaces ur_model -> robot_model (common for all robot types)
    robot_model_arg = DeclareLaunchArgument(
        "robot_model",
        default_value="robopro",  # e.g. ur10e / ur5e / ur3e / robopro
    )

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

    azure_driver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                PathJoinSubstitution(
                    [
                        FindPackageShare("azure_kinect_ros_driver"),
                        "launch",
                        "driver.launch.py",
                    ]
                )
            ]
        ),
        launch_arguments={
            # только RGB
            "color_enabled": "true",
            "color_resolution": "3072P",   # максимум деталей (4096x3072)
            "color_format": "bgra",        # без JPEG-артефактов (тяжелее по CPU/USB)
            "fps": "15",                   # 3072P не бывает 30 FPS

            # всё остальное выключаем
            "depth_enabled": "false",
            "point_cloud": "false",
            "rgb_point_cloud": "false",
            "point_cloud_in_depth_frame": "false",

            # лишние сенсоры/фичи
            "imu_rate_target": "0",
            "wired_sync_mode": "0",
            "body_tracking_enabled": "false",
            "body_tracking_smoothing_factor": "0.0",

            # на всякий случай
            "rescale_ir_to_mono8": "false",
        }.items(),
    )

    cv_pipeline = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                PathJoinSubstitution(
                    [
                        FindPackageShare("iros_cv_algorithms"),
                        "launch",
                        "cv_algorithms.launch.py",
                    ]
                )
            ]
        ),
        launch_arguments=[
            ("mode", "trigger"),
            ("image_topic", "/rgb/image_raw"),
            ("image_timeout_s", "2.0"),
            ("result_prefix", "/cv_algorithms/result"),
            ("trigger_service", "/cv_algorithms/run"),
            ("process_period_s", "2.0"),
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
            tf,
            ur_launch,
            #robopro_launch,
            azure_driver,
            #cv_pipeline,
            voice_rec,
            behavior,
            rviz,
        ]
    )
