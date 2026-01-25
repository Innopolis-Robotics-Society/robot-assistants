from __future__ import annotations

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, OpaqueFunction, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import PushRosNamespace, SetRemap
from launch_ros.substitutions import FindPackageShare


def _as_bool(s: str) -> bool:
    return str(s).strip().lower() in ("1", "true", "yes", "on")

def _launch_setup(context, *args, **kwargs):
    namespace = LaunchConfiguration("namespace").perform(context).strip().strip("/")
    ur_type = LaunchConfiguration("ur_type").perform(context).strip()
    robot_ip = LaunchConfiguration("robot_ip").perform(context).strip()

    kinematics_params_file = LaunchConfiguration("kinematics_params_file").perform(context).strip()
    description_launchfile = LaunchConfiguration("description_launchfile").perform(context).strip()
    tf_prefix = LaunchConfiguration("tf_prefix").perform(context).strip()

    launch_rviz = LaunchConfiguration("launch_rviz").perform(context)
    headless_mode = LaunchConfiguration("headless_mode").perform(context)
    use_tool_communication = LaunchConfiguration("use_tool_communication").perform(context)
    use_mock_hardware = LaunchConfiguration("use_mock_hardware").perform(context)
    initial_joint_controller = LaunchConfiguration("initial_joint_controller").perform(context).strip()
    rviz_config_file = LaunchConfiguration("rviz_config_file").perform(context).strip()

    with_moveit = _as_bool(LaunchConfiguration("with_moveit").perform(context))
    moveit_launch_rviz = LaunchConfiguration("moveit_launch_rviz").perform(context)

    ur_control_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("ur_robot_driver"), "launch", "ur_control.launch.py"])
        ),
        launch_arguments=[
            ("ur_type", ur_type),
            ("robot_ip", robot_ip),
            ("kinematics_params_file", kinematics_params_file),
            ("use_mock_hardware", use_mock_hardware),
            ("headless_mode", headless_mode),
            ("launch_rviz", launch_rviz),
            ("initial_joint_controller", initial_joint_controller),
            ("use_tool_communication", use_tool_communication),
            # опциональные аргументы — добавляем только если задано
            *([("description_launchfile", description_launchfile)] if description_launchfile else []),
            *([("rviz_config_file", rviz_config_file)] if rviz_config_file else []),
            *([("tf_prefix", tf_prefix)] if tf_prefix else []),
        ],
    )


    move_tools = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                PathJoinSubstitution(
                    [
                        FindPackageShare("iros_assistant_bringup"),
                        "launch",
                        "tools",
                        "move_tools.launch.py",
                    ]
                )
            ]
        ),
        launch_arguments=[],
    )

    actions = [
        LogInfo(
            msg=f"[iros_assistant_bringup] UR bringup: ur_type={ur_type}, ip={robot_ip}, "
                f"ns={'/' + namespace if namespace else '(none)'}, moveit={with_moveit}"
        ),
        ur_control_launch,
        move_tools,
    ]

    if with_moveit:
        ur_moveit_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([FindPackageShare("ur_moveit_config"), "launch", "ur_moveit.launch.py"])
            ),
            launch_arguments=[
                ("ur_type", ur_type),
                ("launch_rviz", moveit_launch_rviz),
            ],
        )
        actions.append(ur_moveit_launch)

    if namespace:
        return [GroupAction([PushRosNamespace(namespace), *actions])]

    return actions



def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "namespace",
                default_value="",
                description="Robot namespace (empty = no namespace). Use unique namespaces for multi-robot setups.",
            ),
            DeclareLaunchArgument(
                "ur_type",
                default_value="ur10e",
                description="UR robot type",
            ),
            DeclareLaunchArgument(
                "robot_ip",
                default_value="192.168.0.100",
                description="Robot (or URSim) IP address.",
            ),
            DeclareLaunchArgument(
                "kinematics_params_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("iros_assistant_bringup"), "config", "UR10e-1.yaml"]
                ),
                description="Path to the kinematics_params_file (calibration / extracted kinematics).",
            ),
            DeclareLaunchArgument(
                "description_launchfile",
                default_value="",
                description="(Optional) robot description launch file.",
            ),
            DeclareLaunchArgument(
                "tf_prefix",
                default_value="",
                description="(Optional) TF prefix.",
            ),
            DeclareLaunchArgument(
                "use_mock_hardware",
                default_value="false",
                description="Use mock hardware (no physical robot required).",
            ),
            DeclareLaunchArgument(
                "headless_mode",
                default_value="false",
                description="Driver headless mode.",
            ),
            DeclareLaunchArgument(
                "use_tool_communication",
                default_value="false",
                description="Enable tool communication.",
            ),
            DeclareLaunchArgument(
                "initial_joint_controller",
                default_value="scaled_joint_trajectory_controller",
                description="Initial joint controller name.",
            ),
            DeclareLaunchArgument(
                "launch_rviz",
                default_value="false",
                description="Launch RViz from the driver.",
            ),
            DeclareLaunchArgument(
                "rviz_config_file",
                default_value="",
                description="(Optional) RViz config for the driver, if enabled and supported.",
            ),
            DeclareLaunchArgument(
                "with_moveit",
                default_value="true",
                description="Whether to bring up MoveIt (ur_moveit_config) as part of this launch.",
            ),
            DeclareLaunchArgument(
                "moveit_launch_rviz",
                default_value="true",
                description="Whether MoveIt should launch RViz.",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )