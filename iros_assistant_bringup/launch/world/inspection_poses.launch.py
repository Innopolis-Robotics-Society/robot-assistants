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
    nad = static_tf('nad', 0.328, 0.491, 0.524, 0.0, 0.0, -1.5706, 'world', 'pose_up')

    poi_pcb = static_tf('poi_pcb', 0.3422, 0.570, 0.362, 0.0, 0.0, -1.5706, 'world', 'poi_pcb')
    poi_gear_common = static_tf('poi_gear_common', 0.338, 0.523, 0.335, 0.0, 0.0, -1.5706, 'world', 'poi_gear_common')
    # poi_pcb = static_tf('poi_pcb', 0.346, 0.583, 0.357, 0.0, 0.0, -1.5706, 'world', 'poi_pcb')
    # poi_pcb = static_tf('poi_pcb', 0.346, 0.583, 0.357, 0.0, 0.0, -1.5706, 'world', 'poi_pcb')

    pick_pose = static_tf('pick_pose', 0.2728, 0.478, 1.245, 0.0, 0.0, -1.5706, 'world', 'pose_pick')
    pose_ok = static_tf('pose_ok', 0.407, 0.735, 1.255, 0.0, 0.0, -1.5706, 'world', 'pose_drop_ok')
    pose_nok = static_tf('pose_nok', 0.129, 0.747, 1.255, 0.0, 0.0, -1.5706, 'world', 'pose_drop_nok')
    pick_pose_up = static_tf('pick_pose_up', 0.2728, 1.478, 0.45, 0.0, 0.0, -1.5706, 'world', 'pose_pick_up')
    pose_ok_up = static_tf('pose_ok_up', 0.407, 0.735, 1.45, 0.0, 0.0, -1.5706, 'world', 'pose_drop_ok_up')
    pose_nok_up = static_tf('pose_nok_up', 0.129, 0.747, 1.45, 0.0, 0.0, -1.5706, 'world', 'pose_drop_nok_up')
    azure_camera = static_tf('azure_camera', 0.0, 0.0, 0.0, 0.0, 1.5708, 3.1416, 'tool0_controller', 'camera_base')


    return LaunchDescription([
        nad,
        pick_pose,
        pose_ok,
        pose_nok,
        pick_pose_up,
        pose_ok_up,
        pose_nok_up,
        poi_pcb,
        poi_gear_common,
        #azure_camera,
    ])