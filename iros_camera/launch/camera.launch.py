from __future__ import annotations

import os
import yaml

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def _split_list(s: str) -> list[str]:
    s = (s or "").strip()
    if not s or s.lower() in ("all", "*"):
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def _launch_setup(context, *args, **kwargs):
    pkg_share = get_package_share_directory("iros_camera")

    cameras_arg = LaunchConfiguration("cameras").perform(context)
    cameras_yaml_arg = LaunchConfiguration("cameras_yaml").perform(context)
    use_tf = LaunchConfiguration("use_tf").perform(context).strip().lower() in ("1", "true", "yes", "on")

    cameras_yaml_path = cameras_yaml_arg
    if not os.path.isabs(cameras_yaml_path):
        cameras_yaml_path = os.path.join(pkg_share, cameras_yaml_path)

    with open(cameras_yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    cameras_cfg: dict = cfg.get("cameras", {})
    requested = _split_list(cameras_arg)  # [] => all

    if requested:
        missing = [c for c in requested if c not in cameras_cfg]
        if missing:
            return [LogInfo(msg=f"[iros_camera] Unknown cameras: {missing}. Available: {list(cameras_cfg.keys())}")]
        names = requested
    else:
        names = list(cameras_cfg.keys())

    actions = [LogInfo(msg=f"[iros_camera] Starting cameras: {names}")]

    for name in names:
        cam = cameras_cfg[name]

        video_device = cam.get("video_device", "/dev/video0")
        frame_id = cam.get("frame_id", name)
        camera_info_rel = cam.get("camera_info", "")

        params = [
            {"video_device": video_device},
            {"camera_name": name},
            {"camera_frame_id": frame_id},   # <-- ВАЖНО: теперь Image.header.frame_id станет frame_id
        ]

        if camera_info_rel:
            camera_info_path = camera_info_rel if os.path.isabs(camera_info_rel) else os.path.join(pkg_share, camera_info_rel)
            params.append({"camera_info_url": f"file://{camera_info_path}"})

        # per-camera overrides (image_size, pixel_format, output_encoding, etc.)
        extra = cam.get("params", {})
        if isinstance(extra, dict) and extra:
            params.append(extra)

        camera_node = Node(
            package="v4l2_camera",
            executable="v4l2_camera_node",
            name="v4l2_camera",
            namespace=name,
            parameters=params,
            output="screen",
        )
        actions.append(camera_node)

        if use_tf and "tf" in cam:
            tf_cfg = cam["tf"]
            parent = tf_cfg.get("parent_frame", "base_link")  # <-- исправлено под ваш YAML
            xyz = tf_cfg.get("xyz", [0.0, 0.0, 0.0])
            rpy = tf_cfg.get("rpy", [0.0, 0.0, 0.0])  # roll,pitch,yaw

            tf_node = Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name=f"static_tf_{name}",
                arguments=[
                    "--x", str(xyz[0]), "--y", str(xyz[1]), "--z", str(xyz[2]),
                    "--roll", str(rpy[0]), "--pitch", str(rpy[1]), "--yaw", str(rpy[2]),
                    "--frame-id", parent,
                    "--child-frame-id", frame_id,
                ],
                output="screen",
            )
            actions.append(tf_node)

    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "cameras",
            default_value="all",
            description="Какие камеры запускать: all | * | cam1 | cam1,cam2",
        ),
        DeclareLaunchArgument(
            "cameras_yaml",
            default_value="config/cameras.yaml",
            description="YAML с описанием камер",
        ),
        DeclareLaunchArgument(
            "use_tf",
            default_value="true",
            description="Публиковать static TF для камер",
        ),
        OpaqueFunction(function=_launch_setup),
    ])
