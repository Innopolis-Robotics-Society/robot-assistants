from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('voice_controlled_robot'),
        'config',
        'robot_params.yaml'
    )
    
    voice_controller_node = Node(
        package='voice_controlled_robot',
        executable='voice_controller',
        name='voice_controller',
        output='screen',
        parameters=[config]
    )
    
    return LaunchDescription([
        voice_controller_node
    ])
