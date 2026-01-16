# iros_object_scanner/launch/sphere_pose_generator.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg = get_package_share_directory("iros_object_scanner")

    cfg_pose_generator = os.path.join(pkg, "config", "sphere_pose_generator.yaml")
    cfg_cloud_shot = os.path.join(pkg, "config", "cloud_shot.yaml")


    pose_generator = Node(
            package="iros_object_scanner",
            executable="sphere_pose_generator_node",
            name="sphere_pose_generator_node",
            output="screen",
            parameters=[cfg_pose_generator],
        )

    cloud_shot = Node(
            package="iros_object_scanner",
            executable="cloud_shot",
            name="cloud_shot",
            output="screen",
            parameters=[cfg_cloud_shot],
        )

    return LaunchDescription([
        pose_generator,
        cloud_shot
    ])
