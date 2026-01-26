from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package="rust_detect_ros",
            executable="rust_detect_node",
            name="rust_detect_node",
            output="screen",
            parameters=[{
                "weights": "runs/linknet_mobilenetv2_bin/best.pt",
                "arch": "linknet",
                "encoder": "mobilenet_v2",
                "classes": 2,
                "rust_class_id": 1,
                "tile": 320,
                "stride": 256,
                "thr": 0.35,
                "min_area": 200,
                "close_k": 0,
                "image_topic": "/camera/image_raw",
                "camera_info_topic": "/camera/camera_info",
            }],
        )
    ])
