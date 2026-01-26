#!/usr/bin/env python3
from __future__ import annotations

import glob
from pathlib import Path
from typing import List

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def list_images(image_dir: str) -> List[str]:
    p = Path(image_dir).expanduser()
    if not p.exists():
        return []
    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(str(p / f"*{ext}")))
        files.extend(glob.glob(str(p / f"*{ext.upper()}")))
    return sorted(set(files))


class DebugImagePublisher(Node):
    def __init__(self):
        super().__init__("debug_image_publisher")

        self.declare_parameter("image_dir", "")
        self.declare_parameter("image_topic", "/image")
        self.declare_parameter("rate_hz", 1.0)
        self.declare_parameter("loop", True)

        self.image_dir = str(self.get_parameter("image_dir").value)
        self.image_topic = str(self.get_parameter("image_topic").value)
        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.loop = bool(self.get_parameter("loop").value)

        if self.rate_hz <= 0:
            self.rate_hz = 1.0

        self.bridge = CvBridge()

        self.files = list_images(self.image_dir)
        if not self.files:
            self.get_logger().warn(f"No images found in: {self.image_dir}")
        else:
            self.get_logger().info(
                f"Publishing {len(self.files)} images to {self.image_topic} @ {self.rate_hz} Hz, loop={self.loop}"
            )

        self.pub = self.create_publisher(Image, self.image_topic, qos_profile_sensor_data)

        self.idx = 0
        self.timer = self.create_timer(1.0 / self.rate_hz, self.on_timer)

    def on_timer(self):
        if not self.files:
            return

        if self.idx >= len(self.files):
            if self.loop:
                self.idx = 0
            else:
                self.get_logger().info("Done publishing all images (loop=false).")
                rclpy.shutdown()
                return

        fp = self.files[self.idx]
        self.idx += 1

        img = cv2.imread(fp, cv2.IMREAD_COLOR)  # BGR
        if img is None:
            self.get_logger().warn(f"Failed to read image: {fp}")
            return

        msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "debug_camera"

        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = DebugImagePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
