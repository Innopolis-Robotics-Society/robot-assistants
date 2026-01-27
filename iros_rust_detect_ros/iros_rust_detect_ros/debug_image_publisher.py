#!/usr/bin/env python3
"""
ROS2 debug image publisher:
- Reads images from a folder
- Publishes them to sensor_msgs/Image at 1 Hz in sorted order
- Optionally loops

Usage:
  ros2 run iros_rust_detect_ros debug_image_publisher --ros-args \
    -p folder:="/abs/path/to/debug_io" \
    -p image_topic:="/camera/image_raw" \
    -p rate_hz:=1.0 \
    -p loop:=true \
    -p encoding:="bgr8"

Notes:
- OpenCV reads as BGR by default.
- Your rust node expects "bgr8" input and converts to RGB internally.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(folder: Path) -> List[Path]:
    files = []
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    return files


class DebugImagePublisher(Node):
    def __init__(self):
        super().__init__("debug_image_publisher")

        self.declare_parameter("folder", "")
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("rate_hz", 1.0)
        self.declare_parameter("loop", True)
        self.declare_parameter("encoding", "bgr8")  # bgr8|rgb8|mono8

        folder = self.get_parameter("folder").get_parameter_value().string_value.strip()
        if not folder:
            raise RuntimeError("Parameter 'folder' is empty. Pass -p folder:=/abs/path/to/debug_io")

        self.folder = Path(folder)
        if not self.folder.exists() or not self.folder.is_dir():
            raise RuntimeError(f"Folder not found or not a directory: {self.folder}")

        self.image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.loop = bool(self.get_parameter("loop").value)
        self.encoding = self.get_parameter("encoding").get_parameter_value().string_value

        self.bridge = CvBridge()
        self.pub = self.create_publisher(Image, self.image_topic, 10)

        self.files = list_images(self.folder)
        if not self.files:
            raise RuntimeError(f"No images found in: {self.folder}")

        self.idx = 0

        period = 1.0 / max(1e-6, self.rate_hz)
        self.timer = self.create_timer(period, self._tick)

        self.get_logger().info(f"Publishing {len(self.files)} images from: {self.folder}")
        self.get_logger().info(f"Topic: {self.image_topic} | rate_hz={self.rate_hz} | loop={self.loop} | encoding={self.encoding}")
        self.get_logger().info(f"First: {self.files[0].name}")

    def _tick(self):
        if self.idx >= len(self.files):
            if self.loop:
                self.idx = 0
            else:
                self.get_logger().info("Done (no loop). Shutting down.")
                rclpy.shutdown()
                return

        fp = self.files[self.idx]

        img = cv2.imread(str(fp), cv2.IMREAD_UNCHANGED)
        if img is None:
            self.get_logger().warn(f"Failed to read: {fp}")
            self.idx += 1
            return

        # Ensure encoding compatibility:
        # - bgr8 expects 3 channels uint8
        # - mono8 expects 1 channel uint8
        if self.encoding == "mono8":
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if img.dtype != "uint8":
                img = img.astype("uint8")
        else:
            # bgr8 / rgb8: force 3-channel uint8
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:
                # drop alpha
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            if img.dtype != "uint8":
                img = img.astype("uint8")
            if self.encoding == "rgb8":
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        msg = self.bridge.cv2_to_imgmsg(img, encoding=self.encoding)
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "debug_camera"

        self.pub.publish(msg)
        self.get_logger().info(f"[{self.idx+1}/{len(self.files)}] published: {fp.name}")

        self.idx += 1


def main():
    rclpy.init()
    node = DebugImagePublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

