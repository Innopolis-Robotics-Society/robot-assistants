# iros_cv_algorithms/cv_algorithms_node.py

from __future__ import annotations

import json
import threading
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger

from cv_bridge import CvBridge


class CvAlgorithmsNode(Node):
    def __init__(self):
        super().__init__("cv_algorithms_node")

        # params
        self.declare_parameter("image_topic", "/image")
        self.declare_parameter("image_timeout_s", 0.5)

        self.declare_parameter("trigger_topic", "/cv_algorithms/run")     # std_msgs/Bool
        self.declare_parameter("trigger_service", "/cv_algorithms/run")   # std_srvs/Trigger

        self.declare_parameter("result_prefix", "/cv_algorithms/result")

        self.declare_parameter("mode", "trigger")         # "trigger" | "timer"
        self.declare_parameter("process_period_s", 1.0)   # только для timer-режима

        self._mode = self.get_parameter("mode").value
        self._process_period_s = float(self.get_parameter("process_period_s").value)

        # триггеры создаём только если trigger-режим
        if self._mode == "trigger":
            self.create_subscription(Bool, self._trigger_topic, self._on_trigger_topic, 10)
            self.create_service(Trigger, self._trigger_service, self._on_trigger_service)

        # таймер создаём только если timer-режим
        if self._mode == "timer":
            self._timer = self.create_timer(self._process_period_s, self._on_timer)

        self.get_logger().info(f"Mode: {self._mode}")


        self._image_topic = self.get_parameter("image_topic").value
        self._image_timeout_s = float(self.get_parameter("image_timeout_s").value)
        self._trigger_topic = self.get_parameter("trigger_topic").value
        self._trigger_service = self.get_parameter("trigger_service").value
        self._result_prefix = self.get_parameter("result_prefix").value

        # algorithms + pubs
        from iros_cv_algorithms.algos.corner_detection import CornerDetectionAlgorithm
        from iros_cv_algorithms.algos.rust_detection import RustDetectionAlgorithm

        # NOTE: добавляй сюда свои алгоритмы
        self._algorithms = [
            CornerDetectionAlgorithm(),
            RustDetectionAlgorithm(),
        ]

        self._pubs = {
            algo.key: self.create_publisher(String, f"{self._result_prefix}/{algo.key}", 10)
            for algo in self._algorithms
        }

        self._bridge = CvBridge()

        # busy guard
        self._busy_lock = threading.Lock()
        self._busy = False

        self.get_logger().info(
            f"Ready. image_topic={self._image_topic}, timeout={self._image_timeout_s}s, "
            f"trigger_topic={self._trigger_topic}, trigger_service={self._trigger_service}, "
            f"result_prefix={self._result_prefix}"
        )
        for algo in self._algorithms:
            self.get_logger().info(f"Algo '{algo.key}' -> topic '{self._result_prefix}/{algo.key}'")

    def _on_trigger_topic(self, msg: Bool):
        if msg.data:
            self._start_processing()

    def _on_timer(self):
        self._start_processing()

    def _on_trigger_service(self, request: Trigger.Request, response: Trigger.Response):
        started = self._start_processing()
        response.success = started
        response.message = "started" if started else "busy"
        return response

    def _start_processing(self) -> bool:
        with self._busy_lock:
            if self._busy:
                return False
            self._busy = True

        threading.Thread(target=self._process_once, daemon=True).start()
        return True

    def _grab_one_image(self) -> Optional[Image]:
        """
        Разово подписываемся, ждём 1 кадр, отписываемся.
        """
        event = threading.Event()
        holder = {"msg": None}

        def cb(msg: Image):
            if holder["msg"] is None:
                holder["msg"] = msg
                event.set()

        sub = self.create_subscription(Image, self._image_topic, cb, qos_profile_sensor_data)
        ok = event.wait(timeout=self._image_timeout_s)

        try:
            self.destroy_subscription(sub)
        except Exception:
            pass

        return holder["msg"] if ok else None

    def _process_once(self):
        try:
            t0 = time.perf_counter()
            img_msg = self._grab_one_image()
            grab_ms = (time.perf_counter() - t0) * 1000.0

            if img_msg is None:
                self.get_logger().warn(
                    f"No image received within {self._image_timeout_s}s (grab_ms={grab_ms:.1f})"
                )
                return

            t1 = time.perf_counter()
            cv_bgr = self._bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
            convert_ms = (time.perf_counter() - t1) * 1000.0

            for algo in self._algorithms:
                try:
                    t = time.perf_counter()
                    res = algo.run(cv_bgr)
                    algo_ms = (time.perf_counter() - t) * 1000.0
                    payload = {
                        "algo": algo.key,
                        "ok": True,
                        "grab_ms": grab_ms,
                        "convert_ms": convert_ms,
                        "algo_ms": algo_ms,
                        "result": res,
                    }
                except Exception as e:
                    payload = {
                        "algo": algo.key,
                        "ok": False,
                        "grab_ms": grab_ms,
                        "convert_ms": convert_ms,
                        "error": str(e),
                    }

                self._pubs[algo.key].publish(
                    String(data=json.dumps(payload, ensure_ascii=False, default=str))
                )

        finally:
            with self._busy_lock:
                self._busy = False


def main(args=None):
    rclpy.init(args=args)
    node = CvAlgorithmsNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
