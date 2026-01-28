# iros_cv_session_hub/launch/cv_stack.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    # ---------- launch args ----------
    image_topic = LaunchConfiguration("image_topic")

    rust_service = LaunchConfiguration("rust_service")
    pcb_infer_service = LaunchConfiguration("pcb_infer_service")
    gear_infer_service = LaunchConfiguration("gear_infer_service")

    listen_duration_s = LaunchConfiguration("listen_duration_s")
    output_prefix = LaunchConfiguration("output_prefix")

    rust_service_timeout_s = LaunchConfiguration("rust_service_timeout_s")
    pcb_service_timeout_s = LaunchConfiguration("pcb_service_timeout_s")
    gear_service_timeout_s = LaunchConfiguration("gear_service_timeout_s")

    # hub input topics
    rust_prob_topic = LaunchConfiguration("rust_prob_topic")
    rust_detected_topic = LaunchConfiguration("rust_detected_topic")
    pcb_report_topic = LaunchConfiguration("pcb_report_topic")
    pcb_annotated_topic = LaunchConfiguration("pcb_annotated_topic")
    gear_report_topic = LaunchConfiguration("gear_report_topic")
    gear_annotated_topic = LaunchConfiguration("gear_annotated_topic")

    # optional debug image publisher
    use_debug_pub = LaunchConfiguration("use_debug_publisher")
    image_dir = LaunchConfiguration("image_dir")
    debug_rate_hz = LaunchConfiguration("debug_rate_hz")
    debug_loop = LaunchConfiguration("debug_loop")

    # rust ORT params
    rust_device = LaunchConfiguration("rust_device")
    rust_use_camera_info = LaunchConfiguration("rust_use_camera_info")
    rust_resnet_onnx = LaunchConfiguration("rust_resnet_onnx")
    rust_unet_onnx = LaunchConfiguration("rust_unet_onnx")
    rust_warmup = LaunchConfiguration("rust_warmup")
    rust_resnet_thr = LaunchConfiguration("rust_resnet_thr")
    rust_thr = LaunchConfiguration("rust_thr")
    rust_conf_thr = LaunchConfiguration("rust_conf_thr")
    rust_mask_thr = LaunchConfiguration("rust_mask_thr")
    rust_prob_vis_mode = LaunchConfiguration("rust_prob_vis_mode")
    rust_log_timing = LaunchConfiguration("rust_log_timing")

    # defaults for ONNX from installed share
    default_resnet = PathJoinSubstitution([
        FindPackageShare("iros_rust_detect_ros"), "onnx_models", "resnet_fp16.onnx"
    ])
    default_unet = PathJoinSubstitution([
        FindPackageShare("iros_rust_detect_ros"), "onnx_models", "unet_fp16.onnx"
    ])

    return LaunchDescription([
        # ---------- arguments ----------
        DeclareLaunchArgument(
            "image_topic",
            default_value="/image_raw",
            description="Camera topic."
        ),

        DeclareLaunchArgument("rust_service", default_value="/rust_detect/run"),
        DeclareLaunchArgument("pcb_infer_service", default_value="/pcb_inspector/inference"),
        DeclareLaunchArgument("gear_infer_service", default_value="/gears_check/inference"),

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
        DeclareLaunchArgument("gear_report_topic", default_value="/gears_check/report"),
        DeclareLaunchArgument("gear_annotated_topic", default_value="/gears_check/annotated"),

        # debug publisher (по умолчанию выключен)
        DeclareLaunchArgument("use_debug_publisher", default_value="false"),
        DeclareLaunchArgument("image_dir", default_value=""),
        DeclareLaunchArgument("debug_rate_hz", default_value="1.0"),
        DeclareLaunchArgument("debug_loop", default_value="true"),

        # rust ORT overrides (твои желаемые дефолты)
        DeclareLaunchArgument("rust_device", default_value="cpu"),
        DeclareLaunchArgument("rust_use_camera_info", default_value="false"),
        DeclareLaunchArgument("rust_resnet_onnx", default_value=default_resnet),
        DeclareLaunchArgument("rust_unet_onnx", default_value=default_unet),
        DeclareLaunchArgument("rust_warmup", default_value="0"),
        DeclareLaunchArgument("rust_resnet_thr", default_value="0.5"),
        DeclareLaunchArgument("rust_thr", default_value="0.35"),
        DeclareLaunchArgument("rust_conf_thr", default_value="0.85"),
        DeclareLaunchArgument("rust_mask_thr", default_value="0.5"),
        DeclareLaunchArgument("rust_prob_vis_mode", default_value="invert_glow"),
        DeclareLaunchArgument("rust_log_timing", default_value="true"),

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

        # ---------- Rust detector (ORT, service-driven) ----------
        Node(
            package="iros_rust_detect_ros",
            executable="rust_detect_node",
            name="rust_detect_node",
            output="screen",
            parameters=[{
                "device": rust_device,
                "image_topic": image_topic,
                "use_camera_info": rust_use_camera_info,
                "service_name": rust_service,

                "resnet_onnx": rust_resnet_onnx,
                "unet_onnx": rust_unet_onnx,
                "warmup": rust_warmup,
                "resnet_thr": rust_resnet_thr,

                "thr": rust_thr,
                "conf_thr": rust_conf_thr,
                "mask_thr": rust_mask_thr,
                "prob_vis_mode": rust_prob_vis_mode,
                "log_timing": rust_log_timing,
            }],
        ),

        # ---------- Session hub ----------
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