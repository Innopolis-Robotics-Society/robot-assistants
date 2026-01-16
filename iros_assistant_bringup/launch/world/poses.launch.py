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
    scan = static_tf('scan', 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 'world', 'scan')
    azure_camera = static_tf('azure_camera', 0.0, 0.0, 0.0, 0.0, 1.5708, 3.1416, 'tool0_controller', 'camera_base')

    # view_0 = static_tf('scan', 0.41, 0.724, 0.277, 0.0, 0.0, 1.5706, 'world', 'view_0')
    # view_1 = static_tf('scan', 0.41, 0.535, 0.244, 0.0, -0.815, 1.520, 'world', 'view_1')
    # view_2 = static_tf('scan', 0.208, 0.643, 0.252, -0.014, -1.029, 0.410, 'world', 'view_2')
    # view_3 = static_tf('scan', 0.309, 0.719, 0.234, 0.054, -0.544, -0.170, 'world', 'view_3')
    # view_4 = static_tf('scan', 0.380, 0.682, 0.382, 0.015, -0.247, -0.441, 'world', 'view_4')

    return LaunchDescription([
        nad,
        forward,
        podat,
        watering,
        scan,
        azure_camera,
        # view_0,
        # view_1,
        # view_2,
        # view_3,
        # view_4,
    ])