#!/usr/bin/env python3
from __future__ import annotations

import json
import threading
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from std_srvs.srv import Trigger
from std_msgs.msg import String
from sensor_msgs.msg import Image


def _now_mono() -> float:
    return time.monotonic()


class GearInspectorStub(Node):
    """
    Stub "gear inspector" with RELIABLE pub/sub on annotated topic.
    """

    def __init__(self) -> None:
        super().__init__("gear_inspector_stub")

        self._cbg = ReentrantCallbackGroup()

        # ---- params ----
        self.declare_parameter("image_topic", "/image_raw")
        self.declare_parameter("service_name", "/gear_inspector/inference")
        self.declare_parameter("annotated_topic", "/gear_inspector/annotated")
        self.declare_parameter("report_topic", "/gear_inspector/report")

        self.declare_parameter("wait_image_timeout_s", 2.0)
        self.declare_parameter("require_fresh_image", True)

        image_topic = str(self.get_parameter("image_topic").value)
        service_name = str(self.get_parameter("service_name").value)
        annotated_topic = str(self.get_parameter("annotated_topic").value)
        report_topic = str(self.get_parameter("report_topic").value)

        # ---- QoS ----
        # Use RELIABLE to match subscribers that require RELIABLE.
        # KeepLast is enough for one-shot.
        qos_reliable_img = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        qos_reliable_text = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        # ---- state ----
        self._lock = threading.Lock()
        self._last_img: Optional[Image] = None
        self._last_img_t_mono: float = 0.0
        self._img_event = threading.Event()

        # ---- I/O ----
        # Subscribe to image topic with RELIABLE as well (usually fine; but if camera is BEST_EFFORT only,
        # this may also become incompatible. If that happens, keep subscription BEST_EFFORT and only make
        # annotated publisher RELIABLE.)
        self._sub = self.create_subscription(
            Image, image_topic, self._on_image, qos_reliable_img, callback_group=self._cbg
        )

        self._pub_annot = self.create_publisher(Image, annotated_topic, qos_reliable_img)
        self._pub_report = self.create_publisher(String, report_topic, qos_reliable_text)

        self._srv = self.create_service(Trigger, service_name, self._on_inference, callback_group=self._cbg)

        self.get_logger().info(
            f"gear_inspector_stub ready: service={service_name} image_topic={image_topic} "
            f"annotated_topic={annotated_topic} report_topic={report_topic} (RELIABLE)"
        )

    def _on_image(self, msg: Image) -> None:
        with self._lock:
            self._last_img = msg
            self._last_img_t_mono = _now_mono()
            self._img_event.set()

    def _get_image(self, start_mono: float, timeout_s: float, require_fresh: bool) -> Optional[Image]:
        deadline = _now_mono() + max(0.0, timeout_s)
        while _now_mono() < deadline:
            with self._lock:
                img = self._last_img
                t_mono = self._last_img_t_mono

            if img is not None and (not require_fresh or t_mono >= start_mono):
                return img

            remaining = deadline - _now_mono()
            self._img_event.wait(timeout=min(0.05, max(0.0, remaining)))
            self._img_event.clear()

        return None

    def _on_inference(self, _req: Trigger.Request, res: Trigger.Response) -> Trigger.Response:
        start_mono = _now_mono()
        timeout_s = float(self.get_parameter("wait_image_timeout_s").value)
        require_fresh = bool(self.get_parameter("require_fresh_image").value)

        img = self._get_image(start_mono=start_mono, timeout_s=timeout_s, require_fresh=require_fresh)
        if img is None:
            res.success = False
            res.message = json.dumps(
                {"overall_ok": False, "ok": False, "reason": "no_image", "ts_unix": time.time()},
                ensure_ascii=False,
            )
            return res

        self._pub_annot.publish(img)

        report = {
            "command": "inference",
            "overall_ok": True,
            "ok": True,
            "reason": "stub_ok",
            "ts_unix": time.time(),
        }
        s = String()
        s.data = json.dumps(report, ensure_ascii=False)
        self._pub_report.publish(s)

        res.success = True
        res.message = "ok"
        return res


def main() -> None:
    rclpy.init()
    node = GearInspectorStub()

    ex = MultiThreadedExecutor(num_threads=2)
    ex.add_node(node)
    try:
        ex.spin()
    finally:
        ex.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
