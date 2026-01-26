# cv_session_hub/launch/cv_stack.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    # ---------- launch args ----------
    weights = LaunchConfiguration("weights")
    image_topic = LaunchConfiguration("image_topic")

    rust_service = LaunchConfiguration("rust_service")
    pcb_infer_service = LaunchConfiguration("pcb_infer_service")

    listen_duration_s = LaunchConfiguration("listen_duration_s")
    republish_rate_hz = LaunchConfiguration("republish_rate_hz")
    output_prefix = LaunchConfiguration("output_prefix")

    # optional debug image publisher
    use_debug_pub = LaunchConfiguration("use_debug_publisher")
    image_dir = LaunchConfiguration("image_dir")
    debug_rate_hz = LaunchConfiguration("debug_rate_hz")
    debug_loop = LaunchConfiguration("debug_loop")

    # pcb params (optional override)
    pcb_image_topic = LaunchConfiguration("pcb_image_topic")
    pcb_model_path = LaunchConfiguration("pcb_model_path")
    pcb_conf_thr = LaunchConfiguration("pcb_conf_thr")
    pcb_iou_thr = LaunchConfiguration("pcb_iou_thr")
    pcb_device = LaunchConfiguration("pcb_device")

    # rust params (optional override)
    rust_thr = LaunchConfiguration("rust_thr")
    rust_min_area = LaunchConfiguration("rust_min_area")
    rust_tile = LaunchConfiguration("rust_tile")
    rust_stride = LaunchConfiguration("rust_stride")
    rust_device = LaunchConfiguration("rust_device")
    rust_amp = LaunchConfiguration("rust_amp")
    rust_prob_vis_mode = LaunchConfiguration("rust_prob_vis_mode")

    return LaunchDescription([
        # ---------- arguments ----------
        DeclareLaunchArgument(
            "weights",
            default_value="",
            description="Path to rust segmentation checkpoint (.pth/.pt). Required."
        ),
        DeclareLaunchArgument(
            "image_topic",
            default_value="/camera/image_raw",
            description="Camera topic for rust node (and debug publisher if used)."
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

        DeclareLaunchArgument("listen_duration_s", default_value="2.0"),
        DeclareLaunchArgument("republish_rate_hz", default_value="5.0"),
        DeclareLaunchArgument("output_prefix", default_value="/cv_hub"),

        # debug publisher
        DeclareLaunchArgument("use_debug_publisher", default_value="false"),
        DeclareLaunchArgument("image_dir", default_value=""),
        DeclareLaunchArgument("debug_rate_hz", default_value="2.0"),
        DeclareLaunchArgument("debug_loop", default_value="true"),

        # pcb overrides
        DeclareLaunchArgument("pcb_image_topic", default_value="/camera/image_raw"),
        # ВАЖНО: НЕ ставь default="" иначе ты затрёшь дефолтный путь в ноде и она упадёт FileNotFoundError
        DeclareLaunchArgument(
            "pcb_model_path",
            default_value="/home/mobile/ros2_ws/src/iros_cv_algorithms/iros_cv_algorithms/algos/models/yolo12s-pcb.pt"
        ),
        DeclareLaunchArgument("pcb_conf_thr", default_value="0.25"),
        DeclareLaunchArgument("pcb_iou_thr", default_value="0.50"),
        DeclareLaunchArgument("pcb_device", default_value="0"),  # хотим строку "0"

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
                "image_topic": pcb_image_topic,
                "model_path": pcb_model_path,
                "conf_thr": pcb_conf_thr,
                "iou_thr": pcb_iou_thr,
                # КЛЮЧЕВОЕ: принудительно строковый тип, иначе YAML сделает int
                "device": ParameterValue(pcb_device, value_type=str),
            }],
        ),

        # ---------- Rust detector (service-driven) ----------
        Node(
            package="rust_detect_ros",
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

        # ---------- Session hub ----------
        Node(
            package="cv_session_hub",
            executable="cv_session_hub",
            name="cv_session_hub",
            output="screen",
            parameters=[{
                "listen_duration_s": listen_duration_s,
                "republish_rate_hz": republish_rate_hz,
                "output_prefix": output_prefix,

                "call_rust_service": True,
                "call_pcb_infer_service": True,

                "rust_service": rust_service,
                "pcb_infer_service": pcb_infer_service,

                "rust_prob_topic": "/rust/prob",
                "rust_detected_topic": "/rust/detected",

                "pcb_report_topic": "/pcb_inspector/report",
                "pcb_annotated_topic": "/pcb_inspector/annotated",
            }],
        ),
    ])
