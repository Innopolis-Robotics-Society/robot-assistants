from __future__ import annotations

from pathlib import Path
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_srvs.srv import Trigger
from geometry_msgs.msg import TransformStamped

from .pipeline import run_alignment_pipeline


def _R_to_quat_xyzw(R: np.ndarray) -> tuple[float, float, float, float]:
    # Robust rotation-matrix -> quaternion (x,y,z,w)
    # Assumes R is a proper rotation.
    m = R
    t = np.trace(m)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    else:
        if (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s

    # normalize
    q = np.array([x, y, z, w], dtype=float)
    q /= max(np.linalg.norm(q), 1e-12)
    return float(q[0]), float(q[1]), float(q[2]), float(q[3])


class IcpMeshAlignServer(Node):
    def __init__(self) -> None:
        super().__init__("icp_mesh_align_server")

        # Parameters (paths + pipeline tuning)
        self.declare_parameter("ref_path", "")
        self.declare_parameter("scan_path", "")
        self.declare_parameter("out_dir", "/tmp/icp_out")

        self.declare_parameter("auto_voxel", True)
        self.declare_parameter("voxel", 1.0)
        self.declare_parameter("voxel_ratio", 0.005)
        self.declare_parameter("voxel_min", 1e-3)

        self.declare_parameter("n_coarse", 30000)
        self.declare_parameter("n_fine", 200000)
        self.declare_parameter("sample_method", "poisson")
        self.declare_parameter("icp_max_iter", 60)
        self.declare_parameter("remove_outliers", False)

        self.declare_parameter("dist_thresh", 2.0)
        self.declare_parameter("heat_vmin", 0.0)
        self.declare_parameter("heat_vmax", 5.0)

        self.declare_parameter("size_ratio_fail_gt", 3.0)
        self.declare_parameter("size_ratio_fail_lt", 0.33)
        self.declare_parameter("size_ratio_warn_gt", 1.25)
        self.declare_parameter("size_ratio_warn_lt", 0.80)

        self.declare_parameter("write_hist", True)
        self.declare_parameter("write_point_heatmaps", True)
        self.declare_parameter("write_mesh_heatmaps", True)

        # Transform publication
        self.declare_parameter("ref_frame", "cad")
        self.declare_parameter("scan_frame", "scan")  # child
        qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.pub_tf = self.create_publisher(TransformStamped, "scan_to_ref", qos)

        # Service
        self.srv = self.create_service(Trigger, "align", self.handle_align)

        self.get_logger().info("icp_mesh_align_server is ready. Call service: /icp_mesh_align_server/align")

    def handle_align(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        ref_path = self.get_parameter("ref_path").get_parameter_value().string_value
        scan_path = self.get_parameter("scan_path").get_parameter_value().string_value
        out_dir = self.get_parameter("out_dir").get_parameter_value().string_value

        if not ref_path or not scan_path:
            response.success = False
            response.message = "ref_path and scan_path parameters must be set."
            self.get_logger().error(response.message)
            return response

        # Resolve paths
        ref_path_p = str(Path(ref_path).expanduser())
        scan_path_p = str(Path(scan_path).expanduser())
        out_dir_p = str(Path(out_dir).expanduser())

        self.get_logger().info("Starting alignment pipeline...")
        self.get_logger().info(f"ref_path:  {ref_path_p}")
        self.get_logger().info(f"scan_path: {scan_path_p}")
        self.get_logger().info(f"out_dir:   {out_dir_p}")

        try:
            out = run_alignment_pipeline(
                ref_path=ref_path_p,
                scan_path=scan_path_p,
                out_dir=out_dir_p,

                size_ratio_fail_gt=float(self.get_parameter("size_ratio_fail_gt").value),
                size_ratio_fail_lt=float(self.get_parameter("size_ratio_fail_lt").value),
                size_ratio_warn_gt=float(self.get_parameter("size_ratio_warn_gt").value),
                size_ratio_warn_lt=float(self.get_parameter("size_ratio_warn_lt").value),

                n_coarse=int(self.get_parameter("n_coarse").value),
                n_fine=int(self.get_parameter("n_fine").value),
                sample_method=str(self.get_parameter("sample_method").value),
                icp_max_iter=int(self.get_parameter("icp_max_iter").value),
                remove_outliers=bool(self.get_parameter("remove_outliers").value),

                auto_voxel=bool(self.get_parameter("auto_voxel").value),
                voxel=float(self.get_parameter("voxel").value),
                voxel_ratio=float(self.get_parameter("voxel_ratio").value),
                voxel_min=float(self.get_parameter("voxel_min").value),

                dist_thresh=float(self.get_parameter("dist_thresh").value),
                heat_vmin=float(self.get_parameter("heat_vmin").value),
                heat_vmax=float(self.get_parameter("heat_vmax").value),

                write_hist=bool(self.get_parameter("write_hist").value),
                write_point_heatmaps=bool(self.get_parameter("write_point_heatmaps").value),
                write_mesh_heatmaps=bool(self.get_parameter("write_mesh_heatmaps").value),

                log=self.get_logger().info,
                warn=self.get_logger().warn,
            )

            # Publish transform
            T = out.T_scan_to_ref
            R = T[:3, :3]
            t = T[:3, 3]
            qx, qy, qz, qw = _R_to_quat_xyzw(R)

            msg = TransformStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = str(self.get_parameter("ref_frame").value)
            msg.child_frame_id = str(self.get_parameter("scan_frame").value)
            msg.transform.translation.x = float(t[0])
            msg.transform.translation.y = float(t[1])
            msg.transform.translation.z = float(t[2])
            msg.transform.rotation.x = qx
            msg.transform.rotation.y = qy
            msg.transform.rotation.z = qz
            msg.transform.rotation.w = qw
            self.pub_tf.publish(msg)

            # Compose response
            icp = out.report.get("icp", {})
            r2s = out.report.get("dist_ref_to_scan_pcd", {})
            response.success = True
            response.message = (
                f"OK. out_dir={out.out_dir} | "
                f"icp_fitness={icp.get('fitness')} rmse={icp.get('inlier_rmse')} | "
                f"ref->scan p95={r2s.get('p95')} max={r2s.get('max')}"
            )
            self.get_logger().info(response.message)
            return response

        except Exception as e:
            response.success = False
            response.message = f"FAILED: {e}"
            self.get_logger().error(response.message)
            return response


def main() -> None:
    rclpy.init()
    node = IcpMeshAlignServer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
