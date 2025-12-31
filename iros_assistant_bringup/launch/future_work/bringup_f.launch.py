from __future__ import annotations

import os
import yaml

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    LogInfo,
    GroupAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution
from launch_ros.actions import Node, PushRosNamespace, SetRemap
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def _split_csv(s: str) -> list[str]:
    s = (s or "").strip()
    if not s:
        return []
    if s.lower() in ("all", "*"):
        return ["all"]
    return [x.strip() for x in s.split(",") if x.strip()]


def _abs_path(pkg: str, p: str) -> str:
    if not p:
        return p
    if os.path.isabs(p):
        return p
    return os.path.join(get_package_share_directory(pkg), p)


def _launch_setup(context, *args, **kwargs):
    actions = []

    # top-level flags
    with_perception = LaunchConfiguration("with_perception")
    with_robots = LaunchConfiguration("with_robots")
    with_apps = LaunchConfiguration("with_apps")
    with_voice = LaunchConfiguration("with_voice")
    with_world_poses = LaunchConfiguration("with_world_poses")

    # arguments
    robots_arg = LaunchConfiguration("robots").perform(context)
    robots_yaml_arg = LaunchConfiguration("robots_yaml").perform(context)

    # world poses
    poses_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([FindPackageShare("iros_assistant_bringup"), "launch", "world", "poses.launch.py"])
        ]),
        condition=IfCondition(with_world_poses),
        launch_arguments={
            "poses_yaml": LaunchConfiguration("poses_yaml"),
        }.items(),
    )
    actions.append(poses_launch)

    # perception (cameras + apriltags + cv)
    perception_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([FindPackageShare("iros_assistant_bringup"), "launch", "perception", "perception.launch.py"])
        ]),
        condition=IfCondition(with_perception),
        launch_arguments={
            "cameras": LaunchConfiguration("cameras"),
            "cameras_yaml": LaunchConfiguration("cameras_yaml"),
            "use_camera_tf": LaunchConfiguration("use_camera_tf"),
            "with_rviz": LaunchConfiguration("with_perception_rviz"),
            "rviz_config": LaunchConfiguration("perception_rviz_config"),
        }.items(),
    )
    actions.append(perception_launch)

    # robots
    robots_yaml_path = _abs_path("iros_assistant_bringup", robots_yaml_arg)
    try:
        with open(robots_yaml_path, "r", encoding="utf-8") as f:
            robots_cfg = (yaml.safe_load(f) or {}).get("robots", {}) or {}
    except Exception as e:
        actions.append(LogInfo(msg=f"[bringup] Failed to read robots_yaml='{robots_yaml_path}': {e}"))
        robots_cfg = {}

    requested = _split_csv(robots_arg)

    if not robots_cfg:
        actions.append(LogInfo(msg=f"[bringup] robots_yaml has no robots: {robots_yaml_path}"))
    else:
        # resolve selection
        if not requested or requested == ["all"]:
            selected = list(robots_cfg.keys())
        else:
            selected = requested

        # validate / filter by enabled
        final_list: list[str] = []
        missing: list[str] = []

        for name in selected:
            if name not in robots_cfg:
                missing.append(name)
                continue
            if robots_cfg[name].get("enabled", True) is False:
                continue
            final_list.append(name)

        if missing:
            actions.append(LogInfo(msg=f"[bringup] Unknown robots in arg: {missing}. Available: {list(robots_cfg.keys())}"))

        if not final_list:
            actions.append(LogInfo(msg=f"[bringup] No robots to launch (robots='{robots_arg}', enabled filtered)"))
        else:
            actions.append(LogInfo(msg=f"[bringup] Launching robots: {final_list} (from {robots_yaml_path})"))

            for robot_name in final_list:
                rcfg = robots_cfg[robot_name] or {}
                rtype = str(rcfg.get("type", "")).strip()
                rns = str(rcfg.get("namespace", robot_name)).strip() or robot_name

                # launch file by type: launch/robots/<type>.launch.py
                if not rtype:
                    actions.append(LogInfo(msg=f"[bringup] Robot '{robot_name}' has no 'type' in robots.yaml"))
                    continue

                robot_launch_path = PathJoinSubstitution([
                    FindPackageShare("iros_assistant_bringup"),
                    "launch",
                    "robots",
                    f"{rtype}.launch.py",
                ])

                # common: isolate nodes/topics per robot (важно при 2+ роботах)
                group = []

                group.append(PushRosNamespace(TextSubstitution(text=rns)))

                # TO-DO
                # NOTE: это не решает коллизии frame_id (base_link и т.п.) — нужен frame_prefix/URDF-подход.
                group.append(SetRemap(src="/tf", dst="tf"))
                group.append(SetRemap(src="/tf_static", dst="tf_static"))

                # build args for known robot types
                launch_args = {}

                if rtype == "ur":
                    # per-robot overrides, with global defaults fallback
                    launch_args["robot_ip"] = str(rcfg.get("robot_ip", LaunchConfiguration("robot_ip").perform(context)))
                    launch_args["ur_model"] = str(rcfg.get("ur_model", LaunchConfiguration("ur_model").perform(context)))

                    kin = rcfg.get("kinematics_yaml", LaunchConfiguration("ur_kinematics_yaml").perform(context))
                    launch_args["kinematics_yaml"] = _abs_path("iros_assistant_bringup", str(kin))

                    wm = rcfg.get("with_moveit", LaunchConfiguration("with_moveit").perform(context))
                    launch_args["with_moveit"] = str(wm).lower() if isinstance(wm, bool) else str(wm)

                else:
                    # generic robot: pass rcfg["args"] as launch_arguments
                    extra = rcfg.get("args", {}) or {}
                    if isinstance(extra, dict):
                        for k, v in extra.items():
                            launch_args[str(k)] = str(v)

                group.append(
                    IncludeLaunchDescription(
                        PythonLaunchDescriptionSource([robot_launch_path]),
                        launch_arguments=launch_args.items(),
                        condition=IfCondition(with_robots),
                    )
                )

                actions.append(GroupAction(group))

    # tools
    assist_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([FindPackageShare("iros_assistant_bringup"), "launch", "tools", "assist.launch.py"])
        ]),
        condition=IfCondition(with_apps),
        launch_arguments={}.items(),
    )
    actions.append(assist_launch)

    voice_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([FindPackageShare("iros_assistant_bringup"), "launch", "tools", "voice.launch.py"])
        ]),
        condition=IfCondition(with_voice),
        launch_arguments={}.items(),
    )
    actions.append(voice_launch)

    return actions


def generate_launch_description():
    return LaunchDescription([
        # toggles
        DeclareLaunchArgument("with_perception", default_value="true"),
        DeclareLaunchArgument("with_robots", default_value="false"),
        DeclareLaunchArgument("with_moveit", default_value="false"),
        DeclareLaunchArgument("with_apps", default_value="false"),
        DeclareLaunchArgument("with_voice", default_value="false"),
        DeclareLaunchArgument("with_world_poses", default_value="true"),
        DeclareLaunchArgument("with_perception_rviz", default_value="true"),

        # cameras/perception
        DeclareLaunchArgument("cameras", default_value="mook_laptop_camera"),
        DeclareLaunchArgument("cameras_yaml", default_value="config/cameras.yaml"),
        DeclareLaunchArgument("use_camera_tf", default_value="true"),
        DeclareLaunchArgument(
            "perception_rviz_config",
            default_value=PathJoinSubstitution([FindPackageShare("iros_assistant_bringup"), "rviz", "hmm.rviz"]),
        ),

        # robots from YAML
        DeclareLaunchArgument("robots", default_value="all",
                             description="Robot instance names from robots.yaml: all | ur1 | ur1,robot2"),
        DeclareLaunchArgument("robots_yaml", default_value="config/robots.yaml",
                             description="Robots config YAML (relative to iros_assistant_bringup share or absolute)"),

        # defaults (used if not overridden per-robot)
        DeclareLaunchArgument("robot_ip", default_value="192.168.0.100"),
        DeclareLaunchArgument("ur_model", default_value="ur10e"),
        DeclareLaunchArgument(
            "ur_kinematics_yaml",
            default_value=PathJoinSubstitution([FindPackageShare("iros_assistant_bringup"), "config", "UR10e-1.yaml"]),
        ),

        # world poses
        DeclareLaunchArgument(
            "poses_yaml",
            default_value=PathJoinSubstitution([FindPackageShare("iros_assistant_bringup"), "config", "poses.yaml"]),
        ),

        OpaqueFunction(function=_launch_setup),
    ])
