#!/usr/bin/env python3
from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import open3d as o3d

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from std_msgs.msg import Header
from std_srvs.srv import Trigger
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2

from iros_custom_msgs.srv import SetReferenceMesh
from iros_custom_msgs.msg import ICPAlignmentStatistic


@dataclass
class CachedScan:
    frame_id: str
    stamp_sec: int
    stamp_nsec: int
    raw_xyz: np.ndarray
    down_pcd: o3d.geometry.PointCloud
    voxel_used: float
    diag: float
    n_raw: int
    n_down: int


def pc2_to_numpy_xyz(msg: PointCloud2) -> np.ndarray:
    pts = np.fromiter(
        (p for p in point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)),
        dtype=[("x", np.float32), ("y", np.float32), ("z", np.float32)],
    )
    if pts.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    xyz = np.empty((pts.size, 3), dtype=np.float32)
    xyz[:, 0] = pts["x"]
    xyz[:, 1] = pts["y"]
    xyz[:, 2] = pts["z"]
    return xyz


def numpy_xyz_to_o3d(xyz: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    if xyz.size == 0:
        return pcd
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64, copy=False))
    return pcd


def bbox_diag_pcd(pcd: o3d.geometry.PointCloud) -> float:
    if pcd.is_empty():
        return 0.0
    aabb = pcd.get_axis_aligned_bounding_box()
    ext = np.asarray(aabb.get_extent(), dtype=float)
    return float(np.linalg.norm(ext))


def estimate_normals(pcd: o3d.geometry.PointCloud, radius: float) -> None:
    if pcd.is_empty():
        return
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=float(radius), max_nn=30))
    pcd.normalize_normals()


def pack_rgb_to_float32(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    rgb_u32 = (r.astype(np.uint32) << 16) | (g.astype(np.uint32) << 8) | (b.astype(np.uint32))
    return rgb_u32.view(np.float32)


def o3d_to_pointcloud2_xyzrgb(pcd: o3d.geometry.PointCloud, header: Header) -> PointCloud2:
    pts = np.asarray(pcd.points, dtype=np.float32)

    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.is_bigendian = False

    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    msg.point_step = 16

    if pts.size == 0:
        msg.width = 0
        msg.row_step = 0
        msg.data = b""
        msg.is_dense = True
        return msg

    if pcd.has_colors():
        c = np.clip(np.asarray(pcd.colors, dtype=np.float32), 0.0, 1.0)
    else:
        c = np.zeros((pts.shape[0], 3), dtype=np.float32)

    r = (c[:, 0] * 255.0).astype(np.uint8)
    g = (c[:, 1] * 255.0).astype(np.uint8)
    b = (c[:, 2] * 255.0).astype(np.uint8)
    rgb = pack_rgb_to_float32(r, g, b)

    data = np.empty(
        (pts.shape[0],),
        dtype=[("x", np.float32), ("y", np.float32), ("z", np.float32), ("rgb", np.float32)],
    )
    data["x"] = pts[:, 0]
    data["y"] = pts[:, 1]
    data["z"] = pts[:, 2]
    data["rgb"] = rgb

    msg.width = int(pts.shape[0])
    msg.row_step = msg.point_step * msg.width
    msg.data = data.tobytes()
    msg.is_dense = True
    return msg


def make_heat_colors(dist: np.ndarray, vmax: float) -> np.ndarray:
    if dist.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    vmax = float(max(vmax, 1e-9))
    x = np.clip(dist / vmax, 0.0, 1.0)
    colors = np.zeros((dist.size, 3), dtype=np.float64)
    colors[:, 0] = x
    colors[:, 1] = 1.0 - x
    return colors


def stats(dist: np.ndarray) -> Tuple[float, float, float]:
    d = np.asarray(dist, dtype=float)
    return float(np.mean(d)), float(np.percentile(d, 95)), float(np.max(d))


def over_thresh_ratio(dist: np.ndarray, thresh: float) -> float:
    d = np.asarray(dist, dtype=float)
    if d.size == 0:
        return 0.0
    return float(np.mean(d > float(thresh)))


class IcpServiceNode(Node):
    def __init__(self) -> None:
        super().__init__("icp_service_node")

        # Topics
        self.declare_parameter("scan_topic", "/icp/scan_cloud")
        self.declare_parameter("heatmap_scan_to_ref_topic", "/icp/heatmap_scan_to_ref")
        self.declare_parameter("heatmap_ref_to_scan_topic", "/icp/heatmap_ref_to_scan")
        self.declare_parameter("stats_topic", "/icp/alignment_stats")

        # Reference
        self.declare_parameter("ref_mesh_path", "/home/mobile/ros2_ws/src/iros_icp_mesh_align/models/case_well.stl")
        self.declare_parameter("ref_mesh_scale", 0.01)
        self.declare_parameter("ref_sample_points", 200000)

        # Processing
        self.declare_parameter("voxel_size", 0.002)
        self.declare_parameter("max_corr_mult", 5.0)
        self.declare_parameter("icp_max_iter", 50)
        self.declare_parameter("prealign_centroids", True)

        # Size mismatch policy
        self.declare_parameter("size_ratio_fail_gt", 3.0)
        self.declare_parameter("size_ratio_fail_lt", 0.33)
        self.declare_parameter("size_ratio_warn_gt", 1.25)
        self.declare_parameter("size_ratio_warn_lt", 0.80)

        # QoS
        self.declare_parameter("use_transient_local_qos", True)

        # Output coloring + stats threshold
        self.declare_parameter("heat_vmax", 0.01)
        self.declare_parameter("dist_thresh", 0.003)  # 3mm in meters by default
        self.declare_parameter("debug_save_dir", "/tmp/icp_out")
        self.declare_parameter("debug_save_ply", True)

        self._lock = threading.Lock()
        self._latest_scan: Optional[CachedScan] = None

        self._ref_mesh_path: str = ""
        self._ref_pcd_down: Optional[o3d.geometry.PointCloud] = None
        self._ref_diag: float = 0.0
        self._ref_voxel_used: float = 0.0
        self._ref_scale_used: float = 1.0

        use_tl = bool(self.get_parameter("use_transient_local_qos").value)
        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL if use_tl else DurabilityPolicy.VOLATILE

        scan_topic = str(self.get_parameter("scan_topic").value)
        out_s2r = str(self.get_parameter("heatmap_scan_to_ref_topic").value)
        out_r2s = str(self.get_parameter("heatmap_ref_to_scan_topic").value)
        stats_topic = str(self.get_parameter("stats_topic").value)

        self._sub = self.create_subscription(PointCloud2, scan_topic, self._on_scan, qos)
        self._pub_scan_to_ref = self.create_publisher(PointCloud2, out_s2r, qos)
        self._pub_ref_to_scan = self.create_publisher(PointCloud2, out_r2s, qos)
        self._pub_stats = self.create_publisher(ICPAlignmentStatistic, stats_topic, qos)

        self._srv_set_ref = self.create_service(SetReferenceMesh, "/icp/set_reference_mesh", self._set_reference_cb)
        self._srv_run = self.create_service(Trigger, "/icp/run_icp", self._run_icp_cb)

        ref0 = str(self.get_parameter("ref_mesh_path").value).strip()
        if ref0:
            try:
                self._load_reference(ref0)
            except Exception as e:
                self.get_logger().error(f"Failed to load initial reference mesh: {e}")

        self.get_logger().info(f"Ready. Sub: {scan_topic}")
        self.get_logger().info(f"Pub scan->ref: {out_s2r}")
        self.get_logger().info(f"Pub ref->scan: {out_r2s}")
        self.get_logger().info(f"Pub stats:      {stats_topic}")

    def _publish_stats(
        self,
        *,
        header: Header,
        success: bool,
        message: str,
        size_ratio: float,
        size_warn: bool,
        icp_fitness: float = 0.0,
        icp_rmse: float = 0.0,
        scan_to_ref: Optional[np.ndarray] = None,
        ref_to_scan: Optional[np.ndarray] = None,
        scan_points: int = 0,
        ref_points: int = 0,
    ) -> None:
        msg = ICPAlignmentStatistic()
        msg.header = header
        msg.ref_mesh_path = self._ref_mesh_path
        msg.ref_mesh_scale = float(self._ref_scale_used)

        msg.voxel_size = float(self.get_parameter("voxel_size").value)
        msg.size_ratio_scan_over_ref = float(size_ratio)
        msg.size_mismatch_warning = bool(size_warn)

        msg.icp_fitness = float(icp_fitness)
        msg.icp_rmse = float(icp_rmse)

        thresh = float(self.get_parameter("dist_thresh").value)
        msg.dist_thresh = float(thresh)

        msg.scan_points = int(scan_points)
        msg.ref_points = int(ref_points)

        if scan_to_ref is not None and scan_to_ref.size > 0:
            m, p95, mx = stats(scan_to_ref)
            msg.scan_to_ref_mean = m
            msg.scan_to_ref_p95 = p95
            msg.scan_to_ref_max = mx
            msg.scan_to_ref_over_thresh_ratio = over_thresh_ratio(scan_to_ref, thresh)
        else:
            msg.scan_to_ref_mean = 0.0
            msg.scan_to_ref_p95 = 0.0
            msg.scan_to_ref_max = 0.0
            msg.scan_to_ref_over_thresh_ratio = 0.0

        if ref_to_scan is not None and ref_to_scan.size > 0:
            m, p95, mx = stats(ref_to_scan)
            msg.ref_to_scan_mean = m
            msg.ref_to_scan_p95 = p95
            msg.ref_to_scan_max = mx
            msg.ref_to_scan_over_thresh_ratio = over_thresh_ratio(ref_to_scan, thresh)
        else:
            msg.ref_to_scan_mean = 0.0
            msg.ref_to_scan_p95 = 0.0
            msg.ref_to_scan_max = 0.0
            msg.ref_to_scan_over_thresh_ratio = 0.0

        msg.success = bool(success)
        msg.message = str(message)

        self._pub_stats.publish(msg)

    def _on_scan(self, msg: PointCloud2) -> None:
        try:
            xyz = pc2_to_numpy_xyz(msg)
            n_raw = int(xyz.shape[0])

            voxel = float(self.get_parameter("voxel_size").value)
            pcd = numpy_xyz_to_o3d(xyz)
            if voxel > 0.0 and not pcd.is_empty():
                pcd = pcd.voxel_down_sample(voxel)
            estimate_normals(pcd, radius=max(voxel * 2.0, 1e-6))
            diag = bbox_diag_pcd(pcd)

            cached = CachedScan(
                frame_id=str(msg.header.frame_id),
                stamp_sec=int(msg.header.stamp.sec),
                stamp_nsec=int(msg.header.stamp.nanosec),
                raw_xyz=xyz,
                down_pcd=pcd,
                voxel_used=voxel,
                diag=diag,
                n_raw=n_raw,
                n_down=len(pcd.points),
            )
            with self._lock:
                self._latest_scan = cached

            self.get_logger().info(
                f"Cached scan: raw={n_raw} down={cached.n_down} voxel={voxel:g} diag={diag:.6g} frame='{cached.frame_id}'"
            )
        except Exception as e:
            self.get_logger().error(f"Failed to cache scan cloud: {e}")

    def _load_reference(self, mesh_path: str) -> None:
        mesh_path = str(Path(mesh_path).expanduser())
        if not os.path.isfile(mesh_path):
            raise FileNotFoundError(mesh_path)

        scale = float(self.get_parameter("ref_mesh_scale").value)
        if not np.isfinite(scale) or scale == 0.0:
            raise ValueError("ref_mesh_scale must be a finite non-zero number.")

        voxel = float(self.get_parameter("voxel_size").value)
        n_ref = max(int(self.get_parameter("ref_sample_points").value), 1000)

        self.get_logger().info(f"Loading reference mesh: {mesh_path} | scale={scale:g} voxel={voxel:g} sample={n_ref}")

        mesh = o3d.io.read_triangle_mesh(mesh_path)
        if mesh.is_empty():
            raise RuntimeError("Reference mesh is empty")

        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()

        verts = np.asarray(mesh.vertices, dtype=np.float64)
        verts *= scale
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.compute_vertex_normals()

        ref_pcd = mesh.sample_points_uniformly(number_of_points=n_ref)
        ref_down = ref_pcd.voxel_down_sample(voxel) if voxel > 0.0 else ref_pcd
        estimate_normals(ref_down, radius=max(voxel * 2.0, 1e-6))
        ref_diag = bbox_diag_pcd(ref_down)

        with self._lock:
            self._ref_mesh_path = mesh_path
            self._ref_pcd_down = ref_down
            self._ref_diag = ref_diag
            self._ref_voxel_used = voxel
            self._ref_scale_used = scale

        self.get_logger().info(f"Reference ready: down={len(ref_down.points)} diag={ref_diag:.6g}")

    def _set_reference_cb(self, req: SetReferenceMesh.Request, res: SetReferenceMesh.Response) -> SetReferenceMesh.Response:
        try:
            self._load_reference(req.mesh_path)
            res.success = True
            res.message = "Reference mesh loaded"
            self.get_logger().info(res.message)
        except Exception as e:
            res.success = False
            res.message = str(e)
            self.get_logger().error(res.message)
        return res

    def _check_size_mismatch(self, scan_diag: float, ref_diag: float) -> Tuple[bool, bool, float]:
        if ref_diag <= 1e-12:
            return True, False, float("inf")
        ratio = scan_diag / ref_diag
        fail_gt = float(self.get_parameter("size_ratio_fail_gt").value)
        fail_lt = float(self.get_parameter("size_ratio_fail_lt").value)
        warn_gt = float(self.get_parameter("size_ratio_warn_gt").value)
        warn_lt = float(self.get_parameter("size_ratio_warn_lt").value)
        fail = (ratio > fail_gt) or (ratio < fail_lt)
        warn = (ratio > warn_gt) or (ratio < warn_lt)
        return fail, warn, ratio

    def _ensure_scan_down_for_current_voxel(self, scan: CachedScan, voxel_now: float) -> CachedScan:
        if abs(scan.voxel_used - voxel_now) < 1e-12:
            return scan
        pcd = numpy_xyz_to_o3d(scan.raw_xyz)
        if voxel_now > 0.0 and not pcd.is_empty():
            pcd = pcd.voxel_down_sample(voxel_now)
        estimate_normals(pcd, radius=max(voxel_now * 2.0, 1e-6))
        diag = bbox_diag_pcd(pcd)
        return CachedScan(
            frame_id=scan.frame_id,
            stamp_sec=scan.stamp_sec,
            stamp_nsec=scan.stamp_nsec,
            raw_xyz=scan.raw_xyz,
            down_pcd=pcd,
            voxel_used=voxel_now,
            diag=diag,
            n_raw=scan.n_raw,
            n_down=len(pcd.points),
        )

    def _run_icp_cb(self, req: Trigger.Request, res: Trigger.Response) -> Trigger.Response:
        with self._lock:
            scan = self._latest_scan
            ref_down = self._ref_pcd_down
            ref_diag = self._ref_diag
            ref_path = self._ref_mesh_path
            ref_voxel_used = self._ref_voxel_used

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = scan.frame_id if scan else "scan"

        if ref_down is None:
            msg = "Reference mesh is not set. Call /icp/set_reference_mesh first."
            self.get_logger().error(msg)
            self._publish_stats(header=header, success=False, message=msg, size_ratio=0.0, size_warn=False)
            res.success = False
            res.message = msg
            return res

        if scan is None or scan.down_pcd.is_empty():
            msg = "No scan cloud cached yet."
            self.get_logger().error(msg)
            self._publish_stats(header=header, success=False, message=msg, size_ratio=0.0, size_warn=False)
            res.success = False
            res.message = msg
            return res

        try:
            voxel_now = float(self.get_parameter("voxel_size").value)
            scan = self._ensure_scan_down_for_current_voxel(scan, voxel_now)

            if abs(ref_voxel_used - voxel_now) > 1e-12:
                self.get_logger().warn("voxel_size changed since reference load; reloading reference for consistency.")
                self._load_reference(ref_path)
                with self._lock:
                    ref_down = self._ref_pcd_down
                    ref_diag = self._ref_diag
                if ref_down is None:
                    raise RuntimeError("Reference reload failed unexpectedly.")

            fail, warn, ratio = self._check_size_mismatch(scan.diag, ref_diag)

            if fail:
                msg = "Size mismatch too large (units/model mismatch). No scaling applied; aborting."
                self.get_logger().error(msg)
                self._publish_stats(header=header, success=False, message=msg, size_ratio=ratio, size_warn=warn)
                res.success = False
                res.message = msg
                return res

            max_corr = float(self.get_parameter("max_corr_mult").value) * voxel_now
            max_iter = int(self.get_parameter("icp_max_iter").value)
            prealign = bool(self.get_parameter("prealign_centroids").value)

            scan_work = o3d.geometry.PointCloud(scan.down_pcd)
            ref_work = o3d.geometry.PointCloud(ref_down)

            init = np.eye(4, dtype=float)
            if prealign:
                c_scan = np.asarray(scan_work.get_center(), dtype=float)
                c_ref = np.asarray(ref_work.get_center(), dtype=float)
                init[:3, 3] = (c_ref - c_scan)

            if not scan_work.has_normals():
                estimate_normals(scan_work, radius=max(voxel_now * 2.0, 1e-6))
            if not ref_work.has_normals():
                estimate_normals(ref_work, radius=max(voxel_now * 2.0, 1e-6))

            result = o3d.pipelines.registration.registration_icp(
                scan_work,
                ref_work,
                max_correspondence_distance=max_corr,
                init=init,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter),
            )

            T = np.asarray(result.transformation, dtype=float)

            scan_aligned = o3d.geometry.PointCloud(scan_work)
            scan_aligned.transform(T)

            d_scan_to_ref = np.asarray(scan_aligned.compute_point_cloud_distance(ref_work), dtype=float)
            d_ref_to_scan = np.asarray(ref_work.compute_point_cloud_distance(scan_aligned), dtype=float)

            heat_vmax = float(self.get_parameter("heat_vmax").value)
            scan_col = o3d.geometry.PointCloud(scan_aligned)
            scan_col.colors = o3d.utility.Vector3dVector(make_heat_colors(d_scan_to_ref, vmax=heat_vmax))
            ref_col = o3d.geometry.PointCloud(ref_work)
            ref_col.colors = o3d.utility.Vector3dVector(make_heat_colors(d_ref_to_scan, vmax=heat_vmax))

            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = scan.frame_id

            self._pub_scan_to_ref.publish(o3d_to_pointcloud2_xyzrgb(scan_col, header))
            self._pub_ref_to_scan.publish(o3d_to_pointcloud2_xyzrgb(ref_col, header))

            # Publish stats
            self._publish_stats(
                header=header,
                success=True,
                message="OK",
                size_ratio=ratio,
                size_warn=warn,
                icp_fitness=float(result.fitness),
                icp_rmse=float(result.inlier_rmse),
                scan_to_ref=d_scan_to_ref,
                ref_to_scan=d_ref_to_scan,
                scan_points=int(len(scan_col.points)),
                ref_points=int(len(ref_col.points)),
            )

            # Optional debug save
            dbg_dir = str(self.get_parameter("debug_save_dir").value)
            if bool(self.get_parameter("debug_save_ply").value) and dbg_dir:
                Path(dbg_dir).mkdir(parents=True, exist_ok=True)
                o3d.io.write_point_cloud(str(Path(dbg_dir) / "heat_scan_to_ref_latest.ply"), scan_col)
                o3d.io.write_point_cloud(str(Path(dbg_dir) / "heat_ref_to_scan_latest.ply"), ref_col)
                np.savetxt(str(Path(dbg_dir) / "T_scan_to_ref_latest.txt"), T, fmt="%.10g")

            res.success = True
            res.message = f"OK | fitness={result.fitness:.4f} rmse={result.inlier_rmse:.6g}"
            return res

        except Exception as e:
            msg = f"FAILED: {e}"
            self.get_logger().error(msg)
            self._publish_stats(header=header, success=False, message=msg, size_ratio=0.0, size_warn=False)
            res.success = False
            res.message = msg
            return res


def main() -> None:
    rclpy.init()
    node = IcpServiceNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
