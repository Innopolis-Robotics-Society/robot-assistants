from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    mode = LaunchConfiguration("mode")
    image_topic = LaunchConfiguration("image_topic")
    image_timeout_s = LaunchConfiguration("image_timeout_s")
    result_prefix = LaunchConfiguration("result_prefix")
    trigger_service = LaunchConfiguration("trigger_service")
    process_period_s = LaunchConfiguration("process_period_s")

    is_timer = IfCondition(PythonExpression(["'", mode, "' == 'timer'"]))
    is_trigger = IfCondition(PythonExpression(["'", mode, "' == 'trigger'"]))

    common_params = [
        {"image_topic": image_topic},
        {"image_timeout_s": image_timeout_s},
        {"result_prefix": result_prefix},
        {"trigger_service": trigger_service},
    ]

    timer_node = Node(
        package="iros_cv_algorithms",
        executable="cv_algorithms_node",
        name="cv_algorithms_node",
        output="screen",
        parameters=common_params + [
            {"mode": "timer"},
            {"process_period_s": process_period_s},
        ],
        condition=is_timer,
    )

    trigger_node = Node(
        package="iros_cv_algorithms",
        executable="cv_algorithms_node",
        name="cv_algorithms_node",
        output="screen",
        parameters=common_params + [
            {"mode": "trigger"},
            {"process_period_s": 0.0},
        ],
        condition=is_trigger,
    )

    return LaunchDescription([
        DeclareLaunchArgument("mode", default_value="timer", description="trigger|timer"),
        DeclareLaunchArgument("image_topic", default_value="/rgb/image_raw"),
        DeclareLaunchArgument("image_timeout_s", default_value="2.0"),
        DeclareLaunchArgument("result_prefix", default_value="/cv_algorithms/result"),
        DeclareLaunchArgument("trigger_service", default_value="/cv_algorithms/run"),
        DeclareLaunchArgument("process_period_s", default_value="2.0"),

        timer_node,
        trigger_node,
    ])
