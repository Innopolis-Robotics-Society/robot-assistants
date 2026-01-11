#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.parameter import Parameter as RclpyParameter
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from std_srvs.srv import Trigger
from sensor_msgs.msg import PointCloud2, PointField

import tf2_ros
from rcl_interfaces.msg import SetParametersResult


# -------------------------
# PointCloud2 -> numpy(xyz)
# -------------------------

_PF_DATATYPE_TO_DTYPE = {
    PointField.INT8: np.int8,
    PointField.UINT8: np.uint8,
    PointField.INT16: np.int16,
    PointField.UINT16: np.uint16,
    PointField.INT32: np.int32,
    PointField.UINT32: np.uint32,
    PointField.FLOAT32: np.float32,
    PointField.FLOAT64: np.float64,
}


def _get_xyz_offsets_and_types(msg: PointCloud2):
    field_map = {f.name: f for f in msg.fields}
    for k in ("x", "y", "z"):
        if k not in field_map:
            raise RuntimeError(f"PointCloud2 has no '{k}' field. Fields: {[f.name for f in msg.fields]}")
    fx, fy, fz = field_map["x"], field_map["y"], field_map["z"]
    dx = _PF_DATATYPE_TO_DTYPE.get(fx.datatype)
    dy = _PF_DATATYPE_TO_DTYPE.get(fy.datatype)
    dz = _PF_DATATYPE_TO_DTYPE.get(fz.datatype)
    if dx is None or dy is None or dz is None:
        raise RuntimeError("Unsupported PointField datatype for x/y/z")
    return fx.offset, fy.offset, fz.offset, dx, dy, dz


def pointcloud2_to_xyz_numpy(msg: PointCloud2, scale: float = 1.0) -> np.ndarray:
    """
    Reads ONLY x,y,z from PointCloud2 to (N,3) float32.
    Robust to extra fields (e.g. rgb) and padding as long as point_step is correct.
    """
    n_points = int(msg.width) * int(msg.height)
    if n_points == 0:
        return np.empty((0, 3), dtype=np.float32)

    # Defensive check
    expected_len = int(msg.row_step) * int(msg.height)
    if expected_len != len(msg.data):
        n_points = min(n_points, len(msg.data) // int(msg.point_step))

    ox, oy, oz, dx, dy, dz = _get_xyz_offsets_and_types(msg)

    dtype = np.dtype(
        {
            "names": ["x", "y", "z"],
            "formats": [dx, dy, dz],
            "offsets": [ox, oy, oz],
            "itemsize": int(msg.point_step),
        }
    )

    arr = np.frombuffer(msg.data, dtype=dtype, count=n_points)

    pts = np.empty((n_points, 3), dtype=np.float32)
    pts[:, 0] = arr["x"].astype(np.float32, copy=False)
    pts[:, 1] = arr["y"].astype(np.float32, copy=False)
    pts[:, 2] = arr["z"].astype(np.float32, copy=False)

    mask = np.isfinite(pts).all(axis=1)
    pts = pts[mask]

    if scale != 1.0:
        pts *= float(scale)

    return pts


def xyz_numpy_to_pointcloud2(points_xyz: np.ndarray, frame_id: str, stamp) -> PointCloud2:
    points_xyz = np.asarray(points_xyz, dtype=np.float32)

    msg = PointCloud2()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id

    msg.height = 1
    msg.width = int(points_xyz.shape[0])

    msg.is_bigendian = False
    msg.is_dense = False

    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]

    msg.point_step = 12
    msg.row_step = msg.point_step * msg.width
    msg.data = points_xyz.tobytes()
    return msg


# -------------------------
# Geometry helpers
# -------------------------

def voxel_downsample(points: np.ndarray, voxel: float) -> np.ndarray:
    if voxel <= 0.0 or points.shape[0] == 0:
        return points
    coords = np.floor(points / float(voxel)).astype(np.int32)
    _, idx = np.unique(coords, axis=0, return_index=True)
    return points[idx]


def crop_aabb(points: np.ndarray, roi_min: np.ndarray, roi_max: np.ndarray) -> np.ndarray:
    if points.shape[0] == 0:
        return points
    m = (
        (points[:, 0] >= roi_min[0]) & (points[:, 0] <= roi_max[0]) &
        (points[:, 1] >= roi_min[1]) & (points[:, 1] <= roi_max[1]) &
        (points[:, 2] >= roi_min[2]) & (points[:, 2] <= roi_max[2])
    )
    return points[m]


def quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    R = np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)],
            [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )
    return R


def transform_xyz(points: np.ndarray, tf_msg) -> np.ndarray:
    t = tf_msg.transform.translation
    r = tf_msg.transform.rotation
    R = quat_to_rot(r.x, r.y, r.z, r.w)
    trans = np.array([t.x, t.y, t.z], dtype=np.float32)
    return (points @ R.T) + trans


def write_ply_xyz(path: str, points: np.ndarray) -> None:
    points = np.asarray(points, dtype=np.float32)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")


def bbox_stats(points: np.ndarray, sample_max: int = 200_000) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if points.shape[0] == 0:
        return None
    if points.shape[0] > sample_max:
        idx = np.random.choice(points.shape[0], sample_max, replace=False)
        p = points[idx]
    else:
        p = points
    mn = p.min(axis=0)
    mx = p.max(axis=0)
    return mn, mx


# -------------------------
# Node
# -------------------------

class PointCloudAccumulator(Node):
    def __init__(self) -> None:
        super().__init__("pointcloud_accumulator")

        # Parameters
        self.declare_parameter("cloud_topic", "/points2")
        self.declare_parameter("target_frame", "base_link")
        self.declare_parameter("use_latest_tf", True)
        self.declare_parameter("max_cloud_age_sec", 1.5)

        # Cropping
        self.declare_parameter("enable_crop", False)  # IMPORTANT: default off to avoid 0 points
        self.declare_parameter("roi_min", [-2.0, -2.0, -2.0])
        self.declare_parameter("roi_max", [ 2.0,  2.0,  2.0])

        # Optional "table cut"
        self.declare_parameter("override_z_min", False)
        self.declare_parameter("z_min_value", 0.01)

        # Downsample
        self.declare_parameter("voxel_size_capture", 0.002)
        self.declare_parameter("voxel_size_global", 0.002)

        # If xyz is in mm -> 0.001
        self.declare_parameter("input_scale", 1.0)

        # Debug
        self.declare_parameter("debug_stats", True)

        # Output
        self.declare_parameter("publish_topic", "/scan_cloud")
        self.declare_parameter("publish_latched", True)

        # Save
        self.declare_parameter("save_path", "/tmp/scan_cloud.ply")

        self.cloud_topic = str(self.get_parameter("cloud_topic").value)
        self.target_frame = str(self.get_parameter("target_frame").value)
        self.use_latest_tf = bool(self.get_parameter("use_latest_tf").value)
        self.max_cloud_age = float(self.get_parameter("max_cloud_age_sec").value)

        self.enable_crop = bool(self.get_parameter("enable_crop").value)
        self.roi_min = np.array(self.get_parameter("roi_min").value, dtype=np.float32)
        self.roi_max = np.array(self.get_parameter("roi_max").value, dtype=np.float32)

        self.override_z_min = bool(self.get_parameter("override_z_min").value)
        self.z_min_value = float(self.get_parameter("z_min_value").value)

        self.voxel_capture = float(self.get_parameter("voxel_size_capture").value)
        self.voxel_global = float(self.get_parameter("voxel_size_global").value)

        self.input_scale = float(self.get_parameter("input_scale").value)
        self.debug_stats = bool(self.get_parameter("debug_stats").value)

        publish_topic = str(self.get_parameter("publish_topic").value)
        publish_latched = bool(self.get_parameter("publish_latched").value)

        # TF
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # QoS
        sub_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        if publish_latched:
            pub_qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
        else:
            pub_qos = QoSProfile(depth=1)

        self.sub = self.create_subscription(PointCloud2, self.cloud_topic, self._cloud_cb, sub_qos)
        self.pub = self.create_publisher(PointCloud2, publish_topic, pub_qos)

        # Services (private namespace)
        self.srv_reset = self.create_service(Trigger, "~/reset", self._reset_cb)
        self.srv_capture = self.create_service(Trigger, "~/capture", self._capture_cb)
        self.srv_publish = self.create_service(Trigger, "~/publish", self._publish_cb)
        self.srv_save = self.create_service(Trigger, "~/save", self._save_cb)

        # Runtime params update
        self.add_on_set_parameters_callback(self._on_set_params)

        # State
        self.last_cloud: Optional[PointCloud2] = None
        self.last_cloud_rx_time = None
        self.accum_points = np.empty((0, 3), dtype=np.float32)
        self.capture_count = 0

        self.get_logger().info(
            f"Listening: {self.cloud_topic} | target_frame: {self.target_frame} | publish: {publish_topic} | "
            f"enable_crop={self.enable_crop}"
        )

    def _on_set_params(self, params):
        try:
            for p in params:
                if p.name == "save_path" and p.type_ != RclpyParameter.Type.STRING:
                    return SetParametersResult(successful=False, reason="save_path must be string")

            for p in params:
                if p.name == "save_path":
                    pass
                elif p.name == "enable_crop":
                    self.enable_crop = bool(p.value)
                elif p.name == "roi_min":
                    self.roi_min = np.array(p.value, dtype=np.float32)
                elif p.name == "roi_max":
                    self.roi_max = np.array(p.value, dtype=np.float32)
                elif p.name == "override_z_min":
                    self.override_z_min = bool(p.value)
                elif p.name == "z_min_value":
                    self.z_min_value = float(p.value)
                elif p.name == "voxel_size_capture":
                    self.voxel_capture = float(p.value)
                elif p.name == "voxel_size_global":
                    self.voxel_global = float(p.value)
                elif p.name == "input_scale":
                    self.input_scale = float(p.value)
                elif p.name == "debug_stats":
                    self.debug_stats = bool(p.value)

            return SetParametersResult(successful=True)
        except Exception as e:
            return SetParametersResult(successful=False, reason=repr(e))

    def _cloud_cb(self, msg: PointCloud2) -> None:
        self.last_cloud = msg
        self.last_cloud_rx_time = self.get_clock().now()

    def _reset_cb(self, req, resp: Trigger.Response) -> Trigger.Response:
        self.accum_points = np.empty((0, 3), dtype=np.float32)
        self.capture_count = 0
        resp.success = True
        resp.message = "Accumulator cleared."
        return resp

    def _lookup_tf(self, cloud_msg: PointCloud2):
        if cloud_msg.header.frame_id == self.target_frame:
            return None

        timeout = Duration(seconds=0.2)

        # stamped TF first (correct for moving robot)
        try:
            tf = self.tf_buffer.lookup_transform(
                self.target_frame,
                cloud_msg.header.frame_id,
                cloud_msg.header.stamp,
                timeout=timeout,
            )
            return tf
        except Exception:
            if not self.use_latest_tf:
                raise

        # fallback latest TF
        tf = self.tf_buffer.lookup_transform(
            self.target_frame,
            cloud_msg.header.frame_id,
            rclpy.time.Time().to_msg(),
            timeout=timeout,
        )
        return tf

    def _capture_cb(self, req, resp: Trigger.Response) -> Trigger.Response:
        if self.last_cloud is None or self.last_cloud_rx_time is None:
            resp.success = False
            resp.message = "No cloud received yet."
            return resp

        age = (self.get_clock().now() - self.last_cloud_rx_time).nanoseconds * 1e-9
        if age > self.max_cloud_age:
            resp.success = False
            resp.message = f"Last cloud too old: age={age:.3f}s > {self.max_cloud_age:.3f}s"
            return resp

        try:
            src_frame = self.last_cloud.header.frame_id

            # Read xyz in source frame
            pts = pointcloud2_to_xyz_numpy(self.last_cloud, scale=self.input_scale)

            if self.debug_stats:
                bb = bbox_stats(pts)
                if bb is None:
                    self.get_logger().warn(f"[capture] src={src_frame}: finite_pts=0 (all invalid?)")
                else:
                    mn, mx = bb
                    self.get_logger().info(
                        f"[capture] src={src_frame}: finite_pts={pts.shape[0]} "
                        f"bbox_min={mn.tolist()} bbox_max={mx.tolist()}"
                    )

            # Transform xyz only
            tf = self._lookup_tf(self.last_cloud)
            if tf is not None and pts.shape[0] > 0:
                pts = transform_xyz(pts, tf)

                if self.debug_stats:
                    bb2 = bbox_stats(pts)
                    if bb2 is not None:
                        mn2, mx2 = bb2
                        self.get_logger().info(
                            f"[capture] tgt={self.target_frame}: finite_pts={pts.shape[0]} "
                            f"bbox_min={mn2.tolist()} bbox_max={mx2.tolist()}"
                        )

            # Crop (optional)
            if self.enable_crop:
                roi_min = self.roi_min.copy()
                roi_max = self.roi_max.copy()
                if self.override_z_min:
                    roi_min[2] = max(roi_min[2], float(self.z_min_value))

                before = pts.shape[0]
                pts = crop_aabb(pts, roi_min, roi_max)

                if self.debug_stats:
                    self.get_logger().info(
                        f"[capture] crop enabled: roi_min={roi_min.tolist()} roi_max={roi_max.tolist()} "
                        f"kept={pts.shape[0]}/{before}"
                    )

            # Downsample per-capture
            pts = voxel_downsample(pts, self.voxel_capture)

            # Accumulate + global downsample
            if pts.shape[0] > 0:
                self.accum_points = np.vstack([self.accum_points, pts])
                self.accum_points = voxel_downsample(self.accum_points, self.voxel_global)

            self.capture_count += 1

            resp.success = True
            resp.message = (
                f"Captured #{self.capture_count}: added={pts.shape[0]} pts, total={self.accum_points.shape[0]} pts"
            )
            return resp

        except Exception as e:
            resp.success = False
            resp.message = f"Capture failed: {repr(e)}"
            return resp

    def _publish_cb(self, req, resp: Trigger.Response) -> Trigger.Response:
        if self.accum_points.shape[0] == 0:
            resp.success = False
            resp.message = "Accumulator is empty."
            return resp

        msg = xyz_numpy_to_pointcloud2(
            self.accum_points,
            frame_id=self.target_frame,
            stamp=self.get_clock().now().to_msg(),
        )
        self.pub.publish(msg)

        resp.success = True
        resp.message = f"Published {self.accum_points.shape[0]} points on {self.pub.topic_name}."
        return resp

    def _save_cb(self, req, resp: Trigger.Response) -> Trigger.Response:
        if self.accum_points.shape[0] == 0:
            resp.success = False
            resp.message = "Accumulator is empty."
            return resp

        path = str(self.get_parameter("save_path").value)
        try:
            write_ply_xyz(path, self.accum_points)
            resp.success = True
            resp.message = f"Saved {self.accum_points.shape[0]} points to {path}"
            return resp
        except Exception as e:
            resp.success = False
            resp.message = f"Save failed: {repr(e)}"
            return resp


def main() -> None:
    rclpy.init()
    node = PointCloudAccumulator()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
