from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    rviz_cfg = PathJoinSubstitution([FindPackageShare("iros_assistant_bringup"), "rviz", "hmm.rviz"])

    camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                PathJoinSubstitution(
                    [
                        FindPackageShare("iros_camera"),
                        "launch",
                        "camera.launch.py",
                    ]
                )
            ]
        ),
        launch_arguments=[
            ('cameras', 'mook_laptop_camera'),
        ]
    )

    april_tags_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                PathJoinSubstitution(
                    [
                        FindPackageShare("iros_april_tags"),
                        "launch",
                        "detect_apriltags.launch.py",
                    ]
                )
            ]
        ),
        launch_arguments=[
            ('camera', 'mook_laptop_camera'),
        ],
    )

    ins_search = Node(
        package="iros_tool_recognition",
        executable="ins_search",
        output="screen",
        parameters=[
            {
                'model_path': '/home/mobile/ros2_ws/src/iros_tool_recognition/models/best_fixed.pt',
                "image_topic": "/image_rect",
            },
        ],
    )

    rviz = Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', rviz_cfg],
        )

    return LaunchDescription([
        camera_launch,
        april_tags_launch,
        ins_search,
        rviz,
    ])
