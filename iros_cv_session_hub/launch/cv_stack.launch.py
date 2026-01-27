# iros_cv_session_hub/launch/cv_stack.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node


def generate_launch_description():
    # ---------- launch args ----------
    weights = LaunchConfiguration("weights")
    image_topic = LaunchConfiguration("image_topic")

    rust_service = LaunchConfiguration("rust_service")
    pcb_infer_service = LaunchConfiguration("pcb_infer_service")
    gear_infer_service = LaunchConfiguration("gear_infer_service")

    listen_duration_s = LaunchConfiguration("listen_duration_s")
    output_prefix = LaunchConfiguration("output_prefix")

    rust_service_timeout_s = LaunchConfiguration("rust_service_timeout_s")
    pcb_service_timeout_s = LaunchConfiguration("pcb_service_timeout_s")
    gear_service_timeout_s = LaunchConfiguration("gear_service_timeout_s")

    # optional debug image publisher
    use_debug_pub = LaunchConfiguration("use_debug_publisher")
    image_dir = LaunchConfiguration("image_dir")
    debug_rate_hz = LaunchConfiguration("debug_rate_hz")
    debug_loop = LaunchConfiguration("debug_loop")

    # rust params (optional override)
    rust_thr = LaunchConfiguration("rust_thr")
    rust_min_area = LaunchConfiguration("rust_min_area")
    rust_tile = LaunchConfiguration("rust_tile")
    rust_stride = LaunchConfiguration("rust_stride")
    rust_device = LaunchConfiguration("rust_device")
    rust_amp = LaunchConfiguration("rust_amp")
    rust_prob_vis_mode = LaunchConfiguration("rust_prob_vis_mode")

    # hub input topics (configurable)
    rust_prob_topic = LaunchConfiguration("rust_prob_topic")
    rust_detected_topic = LaunchConfiguration("rust_detected_topic")

    pcb_report_topic = LaunchConfiguration("pcb_report_topic")
    pcb_annotated_topic = LaunchConfiguration("pcb_annotated_topic")

    gear_report_topic = LaunchConfiguration("gear_report_topic")
    gear_annotated_topic = LaunchConfiguration("gear_annotated_topic")

    return LaunchDescription([
        # ---------- arguments ----------
        DeclareLaunchArgument(
            "weights",
            default_value="/home/mobile/ros2_ws/src/iros_rust_detect_ros/models/speedup_l1_s0.35.pth",
            description="Path to rust segmentation checkpoint (.pth/.pt)."
        ),
        DeclareLaunchArgument(
            "image_topic",
            default_value="/image_raw",
            description="Camera topic."
        ),

        DeclareLaunchArgument(
            "rust_service",
            default_value="/rust_detect/run",
            description="Rust Trigger service name."
        ),
        DeclareLaunchArgument(
            "pcb_infer_service",
            default_value="/pcb_inspector/inference",
            description="PCB Trigger inference service name."
        ),
        DeclareLaunchArgument(
            "gear_infer_service",
            default_value="/gear_inspector/inference",
            description="Gear Trigger inference service name."
        ),

        DeclareLaunchArgument("listen_duration_s", default_value="2.0"),
        DeclareLaunchArgument("output_prefix", default_value="/cv_hub"),

        DeclareLaunchArgument("rust_service_timeout_s", default_value="10.0"),
        DeclareLaunchArgument("pcb_service_timeout_s", default_value="10.0"),
        DeclareLaunchArgument("gear_service_timeout_s", default_value="10.0"),

        # hub input topics
        DeclareLaunchArgument("rust_prob_topic", default_value="/rust/prob"),
        DeclareLaunchArgument("rust_detected_topic", default_value="/rust/detected"),
        DeclareLaunchArgument("pcb_report_topic", default_value="/pcb_inspector/report"),
        DeclareLaunchArgument("pcb_annotated_topic", default_value="/pcb_inspector/annotated"),
        DeclareLaunchArgument("gear_report_topic", default_value="/gear_inspector/report"),
        DeclareLaunchArgument("gear_annotated_topic", default_value="/gear_inspector/annotated"),

        # debug publisher
        DeclareLaunchArgument("use_debug_publisher", default_value="false"),
        DeclareLaunchArgument("image_dir", default_value=""),
        DeclareLaunchArgument("debug_rate_hz", default_value="2.0"),
        DeclareLaunchArgument("debug_loop", default_value="true"),

        # rust overrides
        DeclareLaunchArgument("rust_thr", default_value="0.35"),
        DeclareLaunchArgument("rust_min_area", default_value="200"),
        DeclareLaunchArgument("rust_tile", default_value="320"),
        DeclareLaunchArgument("rust_stride", default_value="256"),
        DeclareLaunchArgument("rust_device", default_value="auto"),
        DeclareLaunchArgument("rust_amp", default_value="true"),
        DeclareLaunchArgument("rust_prob_vis_mode", default_value="prob_x_gray"),

        # ---------- optional debug image publisher ----------
        Node(
            package="iros_cv_algorithms",
            executable="debug_image_publisher",
            name="debug_image_publisher",
            output="screen",
            condition=IfCondition(use_debug_pub),
            parameters=[{
                "image_dir": image_dir,
                "image_topic": image_topic,
                "rate_hz": debug_rate_hz,
                "loop": debug_loop,
            }],
        ),

        # ---------- PCB inspector ----------
        Node(
            package="iros_cv_algorithms",
            executable="pcb_check",
            name="pcb_inspector",
            output="screen",
            parameters=[{
                "image_topic": image_topic,
            }],
        ),

        # ---------- Gear inspector ----------
        Node(
            package="iros_cv_algorithms",
            executable="gears_check",
            name="gears_check",
            output="screen",
            parameters=[{
                "image_topic": image_topic,
            }],
        ),

        # ---------- Rust detector (service-driven) ----------
        Node(
            package="iros_rust_detect_ros",
            executable="rust_detect_node",
            name="rust_detect_node",
            output="screen",
            parameters=[{
                "weights": weights,
                "image_topic": image_topic,
                "service_name": rust_service,

                "thr": rust_thr,
                "min_area": rust_min_area,
                "tile": rust_tile,
                "stride": rust_stride,
                "device": rust_device,
                "amp": rust_amp,
                "prob_vis_mode": rust_prob_vis_mode,
            }],
        ),

        # ---------- Session hub (3 run triggers + 1 publish trigger, no timer) ----------
        Node(
            package="iros_cv_session_hub",
            executable="iros_cv_session_hub",
            name="iros_cv_session_hub",
            output="screen",
            parameters=[{
                "listen_duration_s": listen_duration_s,
                "output_prefix": output_prefix,

                "rust_service": rust_service,
                "rust_service_timeout_s": rust_service_timeout_s,
                "rust_prob_topic": rust_prob_topic,
                "rust_detected_topic": rust_detected_topic,

                "pcb_infer_service": pcb_infer_service,
                "pcb_service_timeout_s": pcb_service_timeout_s,
                "pcb_report_topic": pcb_report_topic,
                "pcb_annotated_topic": pcb_annotated_topic,

                "gear_infer_service": gear_infer_service,
                "gear_service_timeout_s": gear_service_timeout_s,
                "gear_report_topic": gear_report_topic,
                "gear_annotated_topic": gear_annotated_topic,
            }],
        ),
    ])
