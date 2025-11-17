from launch import LaunchDescription
from launch_ros.actions import Node
import os

def generate_launch_description():
    # Путь к модели Vosk
    model_path = os.path.expanduser('~/vosk-models/vosk-model-small-ru-0.22')
    
    return LaunchDescription([
        Node(
            package='voice_controlled_robot',
            executable='voice_controller',
            name='voice_controller',
            output='screen',
            parameters=[{
                'model_path': model_path,
                'similarity_threshold': 0.7,
                'audio_device': 'auto',
                'sample_rate': 16000,
                'publish_tool_commands': True,
                'enable_partial_results': True
            }]
        ),
    ])
