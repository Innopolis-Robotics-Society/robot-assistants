from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('rlr_tf_analysis'),
        'config',
        'params.yaml'
    )
    
    return LaunchDescription([
        Node(
            package='rlr_tf_analysis',
            executable='tf_analyzer',
            name='tf_analyzer',
            output='screen',
            parameters=[config],
            remappings=[
                ('/apriltag/detections', '/apriltag/detections')  # Явное указание топика
            ]
        )
    ])