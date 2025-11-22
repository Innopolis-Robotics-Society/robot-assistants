from launch import LaunchDescription
from launch_ros.actions import Node

def static_tf(name, x,y,z, r,p,yaw, parent, child):
    return Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name=name,
        arguments=[
            '--x', str(x), '--y', str(y), '--z', str(z),
            '--roll', str(r), '--pitch', str(p), '--yaw', str(yaw),
            '--frame-id', parent, '--child-frame-id', child
        ]
    )

def generate_launch_description():
    nad = static_tf('nad', 0.059, 0.564, 0.626, 0.0, 0.0, 1.5706, 'world', 'nad')
    vpered = static_tf('nad', 0.059, 0.564, 0.626, 1.5706, 0.0, 0.0, 'world', 'vpered')

    return LaunchDescription([
        nad,
    ])