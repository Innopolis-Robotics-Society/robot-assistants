from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from pathlib import Path

def generate_launch_description():

    ur_cfg = PathJoinSubstitution([FindPackageShare("iros_assistant_bringup"), "config", "UR10e-1.yaml"])
    azure_cfg = PathJoinSubstitution([FindPackageShare("iros_assistant_bringup"), "config", "azure_scan.yaml"])
    rviz_cfg = PathJoinSubstitution([FindPackageShare("iros_assistant_bringup"), "rviz", "inspector.rviz"])
    robot_ip_arg = DeclareLaunchArgument(
        'robot_ip',
        default_value='192.168.0.100')
    ur_model_arg = DeclareLaunchArgument(
        'ur_model',
        default_value='ur10e')

    robot_ip = LaunchConfiguration('robot_ip')
    ur_model = LaunchConfiguration('ur_model')

    ur_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                PathJoinSubstitution(
                    [
                        FindPackageShare("iros_assistant_bringup"), "launch", "robots", "ur.launch.py",
                    ]
                )
            ]
        ),
        launch_arguments=[
            ('ur_type', ur_model),
            ('robot_ip', robot_ip),
            ('kinematics_params_file', ur_cfg),
            ('launch_rviz', 'false'),
            ('headless_mode','true'),
            ('use_tool_communication','false'),
        ]
    )

    move_tools = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                PathJoinSubstitution(
                    [
                        FindPackageShare("iros_assistant_bringup"), "launch", "tools", "move_tools.launch.py",
                    ]
                )
            ]
        ),
        launch_arguments=[]
    )

    voice = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                PathJoinSubstitution(
                    [
                        FindPackageShare("iros_assistant_bringup"), "launch", "tools", "voice.launch.py",
                    ]
                )
            ]
        ),
        launch_arguments=[]
    )

    tf = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                PathJoinSubstitution(
                    [
                        FindPackageShare("iros_assistant_bringup"), "launch", "world", "poses.launch.py",
                    ]
                )
            ]
        ),
        launch_arguments=[
        ]
    )


    azure_driver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                PathJoinSubstitution(
                    [
                        FindPackageShare("azure_kinect_ros_driver"), "launch", "driver.launch.py",
                    ]
                )
            ]
        ),
        launch_arguments={
            "depth_enabled": "true",
            "color_enabled": "true",

            "point_cloud": "true",
            "rgb_point_cloud": "true",
            "point_cloud_in_depth_frame": "false",

            # best close-range detail
            "depth_mode": "WFOV_UNBINNED",
            "fps": "15",

            # meters float to avoid mm/m mismatch later
            "depth_unit": "32FC1",

            # nice looking color + lower CPU than BGRA
            "color_resolution": "1080P",
            "color_format": "bgra",

            # lower IMU traffic than full rate
            "imu_rate_target": "100",

            "wired_sync_mode": "0",
            "body_tracking_smoothing_factor": "0.0",
        }.items(),
    )

    rviz = Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', rviz_cfg],
        )

    return LaunchDescription([
        ur_model_arg,
        robot_ip_arg,
        tf,
        ur_launch,
        azure_driver,
        move_tools,
        #voice,
        rviz,
    ])