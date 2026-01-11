#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Static PointCloud2 publisher from a mesh or point-cloud file.

- Loads model from disk (STL/OBJ/PLY/PCD...)
- Converts to points (samples if mesh)
- Applies explicit unit scaling via parameter model_scale (default 1.0)
- Publishes "latched" cloud (TRANSIENT_LOCAL + RELIABLE) to a topic (default: /icp/scan_cloud)
- Re-publishes on parameter updates (mesh_path, n_points, sample_method, voxel_downsample, model_scale, frame_id)
- Optional periodic republish

This is intended for testing your service-triggered ICP node.

Parameters:
- mesh_path (string): path to model file
- topic (string): output topic
- frame_id (string): frame for PointCloud2
- n_points (int): if mesh, number of points to sample
- sample_method (string): "poisson" or "uniform"
- voxel_downsample (float): 0 disables, else voxel size (after scaling)
- model_scale (float): multiply xyz by this factor (e.g. 0.001 for mm->m)
- publish_period_s (float): 0 => publish once; >0 => republish periodically
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import open3d as o3d

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult

from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField


def _numpy_xyz_to_pointcloud2(xyz: np.ndarray, header: Header) -> PointCloud2:
    msg = PointCloud2()
    msg.header = header
    msg.height = 1

    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.fields = fields
    msg.is_bigendian = False
    msg.point_step = 12

    if xyz.size == 0:
        msg.width = 0
        msg.row_step = 0
        msg.data = b""
        msg.is_dense = True
        return msg

    xyz32 = np.ascontiguousarray(xyz.astype(np.float32, copy=False))
    n = int(xyz32.shape[0])
    msg.width = n
    msg.row_step = msg.point_step * n
    msg.data = xyz32.tobytes()
    msg.is_dense = bool(np.isfinite(xyz32).all())
    return msg


def _read_model(path: str) -> Tuple[Optional[o3d.geometry.TriangleMesh], Optional[o3d.geometry.PointCloud]]:
    p = str(Path(path))
    mesh = o3d.io.read_triangle_mesh(p)
    if mesh is not None and not mesh.is_empty() and len(mesh.triangles) > 0:
        return mesh, None

    pcd = o3d.io.read_point_cloud(p)
    if pcd is not None and not pcd.is_empty():
        return None, pcd

    return None, None


def _mesh_to_points(mesh: o3d.geometry.TriangleMesh, n_points: int, method: str) -> np.ndarray:
    mesh = o3d.geometry.TriangleMesh(mesh)  # copy
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()

    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    method = method.strip().lower()
    n_points = max(int(n_points), 100)

    if method == "poisson":
        pcd = mesh.sample_points_poisson_disk(number_of_points=n_points, init_factor=5)
    elif method == "uniform":
        pcd = mesh.sample_points_uniformly(number_of_points=n_points)
    else:
        raise ValueError("sample_method must be 'poisson' or 'uniform'")

    return np.asarray(pcd.points, dtype=np.float32)


def _pcd_to_points(pcd: o3d.geometry.PointCloud) -> np.ndarray:
    return np.asarray(pcd.points, dtype=np.float32)


def _aabb_stats(xyz: np.ndarray) -> str:
    if xyz.size == 0:
        return "empty"
    mn = xyz.min(axis=0)
    mx = xyz.max(axis=0)
    ext = mx - mn
    diag = float(np.linalg.norm(ext))
    return f"min={mn.tolist()} max={mx.tolist()} extent={ext.tolist()} diag={diag:.6g}"


class StaticCloudPublisher(Node):
    def __init__(self) -> None:
        super().__init__("static_cloud_publisher")

        self.declare_parameter("mesh_path", "/home/mobile/ros2_ws/src/iros_icp_mesh_align/models/case_crack_1.stl")
        self.declare_parameter("topic", "/icp/scan_cloud")
        self.declare_parameter("frame_id", "scan")

        self.declare_parameter("n_points", 100000)
        self.declare_parameter("sample_method", "poisson")  # poisson|uniform
        self.declare_parameter("voxel_downsample", 0.0)      # applied after scaling
        self.declare_parameter("model_scale", 0.01)           # explicit units conversion (e.g. 0.001 mm->m)

        self.declare_parameter("publish_period_s", 0.1)      # 0 => publish once on load/update

        self._last_msg: Optional[PointCloud2] = None
        self._timer = None

        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        topic = str(self.get_parameter("topic").value)
        self._pub = self.create_publisher(PointCloud2, topic, qos)

        self.add_on_set_parameters_callback(self._on_params)

        self._configure_timer(float(self.get_parameter("publish_period_s").value))

        self.get_logger().info(f"Publishing static cloud to: {topic} (TRANSIENT_LOCAL, RELIABLE)")
        self.get_logger().info("Set parameter 'mesh_path' to load/publish. Use model_scale to match ROS units.")
        self.get_logger().info("Example for STL in mm and ROS in m: model_scale:=0.001")

        mesh_path = str(self.get_parameter("mesh_path").value).strip()
        if mesh_path:
            ok, msg = self._load_and_publish(mesh_path)
            if ok:
                self.get_logger().info(msg)
            else:
                self.get_logger().error(msg)

    def _configure_timer(self, period_s: float) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

        if period_s and period_s > 0.0:
            self._timer = self.create_timer(period_s, self._republish_timer_cb)
            self.get_logger().info(f"Periodic republish enabled: every {period_s:g}s")
        else:
            self.get_logger().info("Periodic republish disabled (publish-on-load/update only).")

    def _republish_timer_cb(self) -> None:
        if self._last_msg is None:
            return
        self._last_msg.header.stamp = self.get_clock().now().to_msg()
        self._pub.publish(self._last_msg)

    def _load_and_publish(self, mesh_path: str) -> Tuple[bool, str]:
        mesh_path = str(Path(mesh_path).expanduser())
        if not os.path.exists(mesh_path):
            return False, f"mesh_path does not exist: {mesh_path}"

        frame_id = str(self.get_parameter("frame_id").value)
        n_points = int(self.get_parameter("n_points").value)
        method = str(self.get_parameter("sample_method").value)
        voxel = float(self.get_parameter("voxel_downsample").value)
        scale = float(self.get_parameter("model_scale").value)

        mesh, pcd = _read_model(mesh_path)
        if mesh is None and pcd is None:
            return False, f"Failed to read model as mesh or point cloud: {mesh_path}"

        if mesh is not None:
            xyz = _mesh_to_points(mesh, n_points=n_points, method=method)
            src_kind = "mesh"
        else:
            xyz = _pcd_to_points(pcd)
            src_kind = "pointcloud"

        if xyz.size == 0:
            return False, "Loaded model produced empty point set."

        # Explicit unit scale (no auto-detection; controlled by user)
        if not np.isfinite(scale) or scale == 0.0:
            return False, "model_scale must be a finite non-zero number."
        xyz = xyz * np.float32(scale)

        if voxel and voxel > 0.0:
            tmp = o3d.geometry.PointCloud()
            tmp.points = o3d.utility.Vector3dVector(xyz.astype(np.float64, copy=False))
            tmp = tmp.voxel_down_sample(voxel_size=float(voxel))
            xyz = np.asarray(tmp.points, dtype=np.float32)

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = frame_id

        msg = _numpy_xyz_to_pointcloud2(xyz, header)
        self._last_msg = msg
        self._pub.publish(msg)

        stats = _aabb_stats(xyz)
        return True, (
            f"Published {len(xyz)} pts from {src_kind}: {mesh_path} | "
            f"frame_id='{frame_id}' | model_scale={scale:g} | {stats}"
        )

    def _on_params(self, params: list[Parameter]) -> SetParametersResult:
        try:
            # handle timer update
            for p in params:
                if p.name == "publish_period_s":
                    self._configure_timer(float(p.value))

            # reload & republish on any relevant change
            reload_keys = {"mesh_path", "frame_id", "n_points", "sample_method", "voxel_downsample", "model_scale"}
            if any(p.name in reload_keys for p in params):
                mesh_path = str(self.get_parameter("mesh_path").value).strip()
                for p in params:
                    if p.name == "mesh_path":
                        mesh_path = str(p.value).strip()

                if mesh_path:
                    ok, msg = self._load_and_publish(mesh_path)
                    if not ok:
                        return SetParametersResult(successful=False, reason=msg)
                    self.get_logger().info(msg)

            return SetParametersResult(successful=True)

        except Exception as e:
            return SetParametersResult(successful=False, reason=str(e))


def main() -> None:
    rclpy.init()
    node = StaticCloudPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
