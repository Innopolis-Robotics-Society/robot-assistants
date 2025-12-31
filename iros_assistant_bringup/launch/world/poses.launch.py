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
    nad = static_tf('nad', 0.059, 0.75, 0.60, 0.0, 0.0, 1.5706, 'world', 'pose_up')
    podat = static_tf('podat', 0.459, 0.664, 0.526, 1.5706, -1.5706, -0.86, 'world', 'pose_podat')
    forward = static_tf('forward', 0.059, 0.664, 0.40, 1.5706, -1.5706, 0.0, 'world', 'pose_forward')
    watering = static_tf('watering', 0.40, 0.60, 0.10, 1.5706, -1.5706, 0.0, 'world', 'pose_watering')

    return LaunchDescription([
        nad,
        forward,
        podat,
        watering,
    ])