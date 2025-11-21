import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import FindExecutable


def generate_launch_description():
    ur_type = LaunchConfiguration("ur_type")
    robot_ip = LaunchConfiguration("robot_ip")

    declare_ur_type = DeclareLaunchArgument(
        "ur_type",
        default_value="ur10e",
        description="UR robot type (ur3e, ur5e, ur10e, ...)",
    )

    declare_robot_ip = DeclareLaunchArgument(
        "robot_ip",
        default_value="192.168.0.100",  # поменяй на свой IP робота при запуске
        description="Robot IP (используется только в URDF xacro)",
    )

    # --- URDF / robot_description ---
    joint_limit_params = PathJoinSubstitution(
        [FindPackageShare("ur_description"), "config", ur_type, "joint_limits.yaml"]
    )
    kinematics_params = PathJoinSubstitution(
        [FindPackageShare("ur_description"), "config", ur_type, "default_kinematics.yaml"]
    )
    physical_params = PathJoinSubstitution(
        [FindPackageShare("ur_description"), "config", ur_type, "physical_parameters.yaml"]
    )
    visual_params = PathJoinSubstitution(
        [FindPackageShare("ur_description"), "config", ur_type, "visual_parameters.yaml"]
    )

    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([FindPackageShare("ur_description"), "urdf", "ur.urdf.xacro"]),
            " ",
            "robot_ip:=", robot_ip,
            " ",
            "joint_limit_params:=", joint_limit_params,
            " ",
            "kinematics_params:=", kinematics_params,
            " ",
            "physical_params:=", physical_params,
            " ",
            "visual_params:=", visual_params,
            " ",
            "name:=", "ur",
            " ",
            "ur_type:=", ur_type,
            " ",
            "prefix:=", '""',
        ]
    )
    robot_description = {"robot_description": robot_description_content}

    # --- SRDF / robot_description_semantic ---
    robot_description_semantic_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([FindPackageShare("ur_moveit_config"), "srdf", "ur.srdf.xacro"]),
            " ",
            "name:=", "ur",
            " ",
            "prefix:=", '""',
        ]
    )
    robot_description_semantic = {
        "robot_description_semantic": robot_description_semantic_content
    }

    go_to_tf_node = Node(
        package="ur_assist",
        executable="go_to_tf",
        output="screen",
        parameters=[
            robot_description,
            robot_description_semantic,
            {
                "planning_group": "ur_manipulator",
                "base_frame": "base_link",
                "target_frame": "target_tf",
            },
        ],
    )

    return LaunchDescription(
        [
            declare_ur_type,
            declare_robot_ip,
            go_to_tf_node,
        ]
    )
