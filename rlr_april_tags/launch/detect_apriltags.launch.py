import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    apriltag_config = os.path.join(
        get_package_share_directory('rlr_april_tags'),
        'config',
        'apriltag.yaml'
    )

    rectify_node = Node(
        package='image_proc',
        executable='rectify_node',
        name='rectify_node',
        remappings=[
            ('image', '/image_raw'),
            ('camera_info', '/camera_info'),
            ('image_rect', 'image_rect')
        ]
    )
  
    apriltag_node = Node(
        package='apriltag_ros',
        executable='apriltag_node',
        name='apriltag',
        parameters=[apriltag_config],
        remappings=[
            ('image_rect', '/image_rect'),
            ('camera_info', '/camera_info')
        ]
    )
    
    return LaunchDescription([
        rectify_node,
        apriltag_node
    ])  