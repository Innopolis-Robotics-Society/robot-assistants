from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package="iros_rust_detect_ros",
            executable="rust_detect_node",
            name="rust_detect_node",
            output="screen",
            parameters=[{
                "weights": "/home/mobile/ros2_ws/src/iros_rust_detect_ros/models/speedup_l1_s0.35.pth",
                "arch": "linknet",
                "encoder": "mobilenet_v2",
                "classes": 2,
                "rust_class_id": 1,
                "tile": 320,
                "stride": 256,
                "thr": 0.35,
                "min_area": 200,
                "close_k": 0,
                "image_topic": "/image_raw",
                "camera_info_topic": "/camera_info",
            }],
        )


    ])