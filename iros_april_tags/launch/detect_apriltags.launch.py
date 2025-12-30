import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, PushRosNamespace


def generate_launch_description():
    camera = LaunchConfiguration("camera")

    image_topic = LaunchConfiguration("image_topic")
    camera_info_topic = LaunchConfiguration("camera_info_topic")
    image_rect_topic = LaunchConfiguration("image_rect_topic")

    apriltag_config = os.path.join(
        get_package_share_directory("iros_april_tags"),
        "config",
        "apriltag.yaml",
    )

    rectify_node = Node(
        package="image_proc",
        executable="rectify_node",
        name="rectify_node",
        remappings=[
            ("image", image_topic),
            ("camera_info", camera_info_topic),
            ("image_rect", image_rect_topic),
        ],
        arguments=["--ros-args", "--log-level", "error"],
        output="screen",
    )

    apriltag_node = Node(
        package="apriltag_ros",
        executable="apriltag_node",
        name="apriltag",
        parameters=[apriltag_config],
        remappings=[
            ("image_rect", image_rect_topic),
            ("camera_info", camera_info_topic),
        ],
        arguments=["--ros-args", "--log-level", "error"],
        output="screen",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "camera",
                default_value="mook_laptop_camera",
                description="Camera namespace",
            ),
            DeclareLaunchArgument(
                "image_topic",
                default_value="image_raw",
                description="Image topic in camera namespace",
            ),
            DeclareLaunchArgument(
                "camera_info_topic",
                default_value="camera_info",
                description="CameraInfo topic in camera namespace ",
            ),
            DeclareLaunchArgument(
                "image_rect_topic",
                default_value="image_rect",
                description="Rectified topic of image in camera namespace",
            ),
            GroupAction(
                [
                    PushRosNamespace(camera),
                    rectify_node,
                    apriltag_node,
                ]
            ),
        ]
    )
