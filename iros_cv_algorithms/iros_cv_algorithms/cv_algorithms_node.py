# iros_cv_algorithms/cv_algorithms_node.py

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
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
        self.declare_parameter("process_period_s", 1.0)   # only for timer-mode

        # pcb baseline sampling
        self.declare_parameter("pcb_baseline_samples", 5)
        self.declare_parameter("pcb_baseline_interval_s", 1.0)

        # debug saving for pcb baseline
        self.declare_parameter("save_pcb_baseline_debug", True)
        self.declare_parameter("debug_images_dir", "images")  # relative to CWD by default

        # read params FIRST
        self._image_topic = self.get_parameter("image_topic").value
        self._image_timeout_s = float(self.get_parameter("image_timeout_s").value)
        self._trigger_topic = self.get_parameter("trigger_topic").value
        self._trigger_service = self.get_parameter("trigger_service").value
        self._result_prefix = self.get_parameter("result_prefix").value

        self._mode = self.get_parameter("mode").value
        self._process_period_s = float(self.get_parameter("process_period_s").value)

        self._pcb_baseline_samples = int(self.get_parameter("pcb_baseline_samples").value)
        self._pcb_baseline_interval_s = float(self.get_parameter("pcb_baseline_interval_s").value)

        self._save_pcb_baseline_debug = bool(self.get_parameter("save_pcb_baseline_debug").value)
        self._debug_images_dir = Path(str(self.get_parameter("debug_images_dir").value)).expanduser()
        if not self._debug_images_dir.is_absolute():
            self._debug_images_dir = Path.cwd() / self._debug_images_dir
        if self._save_pcb_baseline_debug:
            self._debug_images_dir.mkdir(parents=True, exist_ok=True)

        # triggers/timer AFTER reading topics
        if self._mode == "trigger":
            self.create_subscription(Bool, self._trigger_topic, self._on_trigger_topic, 10)
            self.create_service(Trigger, self._trigger_service, self._on_trigger_service)
        elif self._mode == "timer":
            self._timer = self.create_timer(self._process_period_s, self._on_timer)
        else:
            self.get_logger().warn(f"Unknown mode '{self._mode}', fallback to trigger")
            self._mode = "trigger"
            self.create_subscription(Bool, self._trigger_topic, self._on_trigger_topic, 10)
            self.create_service(Trigger, self._trigger_service, self._on_trigger_service)

        # algorithms + pubs
        from iros_cv_algorithms.algos.corner_detection import CornerDetectionAlgorithm
        from iros_cv_algorithms.algos.rust_detection import RustDetectionAlgorithm
        from iros_cv_algorithms.algos.pcb_detection import PCBDetectionAlgorithm

        self._algorithms = [
            CornerDetectionAlgorithm(),
            RustDetectionAlgorithm(),
            PCBDetectionAlgorithm(calib_samples=self._pcb_baseline_samples),
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
            f"Ready. mode={self._mode}, image_topic={self._image_topic}, timeout={self._image_timeout_s}s, "
            f"trigger_topic={self._trigger_topic}, trigger_service={self._trigger_service}, "
            f"result_prefix={self._result_prefix}"
        )
        if self._save_pcb_baseline_debug:
            self.get_logger().info(f"PCB baseline debug dir: {self._debug_images_dir}")

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
        Subscribe once, wait for 1 frame, unsubscribe.
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

    def _publish(self, algo_key: str, payload: dict):
        self._pubs[algo_key].publish(String(data=json.dumps(payload, ensure_ascii=False, default=str)))

    def _draw_and_save(self, image_bgr, dets: list, out_path: Path):
        if not dets:
            cv2.imwrite(str(out_path), image_bgr)
            return

        img = image_bgr.copy()
        h, w = img.shape[:2]

        for d in dets:
            try:
                cls = str(d.get("cls", ""))
                x = float(d["x"])
                y = float(d["y"])
                bw = float(d["w"])
                bh = float(d["h"])
            except Exception:
                continue

            x1 = int((x - bw / 2.0) * w)
            y1 = int((y - bh / 2.0) * h)
            x2 = int((x + bw / 2.0) * w)
            y2 = int((y + bh / 2.0) * h)

            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if cls:
                cv2.putText(img, cls, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imwrite(str(out_path), img)

    def _maybe_save_pcb_baseline_debug(self, image_bgr, res: dict):
        """
        Save only baseline calibration images and baseline summary.
        """
        if not self._save_pcb_baseline_debug:
            return
        if not isinstance(res, dict):
            return

        # save annotated images during calibration
        if res.get("calibrating"):
            dets = res.get("detections", [])
            step = res.get("progress", {}).get("have", None)
            need = res.get("progress", {}).get("need", None)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            suffix = f"{step:02d}_of_{need:02d}" if isinstance(step, int) and isinstance(need, int) else "step"
            out_img = self._debug_images_dir / f"pcb_baseline_{stamp}_{suffix}.jpg"
            self._draw_and_save(image_bgr, dets, out_img)

        # save baseline summary when created
        if res.get("reason") == "baseline_created" and res.get("baseline_set"):
            stamp = time.strftime("%Y%m%d_%H%M%S")
            out_json = self._debug_images_dir / f"pcb_baseline_{stamp}.json"
            try:
                out_json.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            except Exception:
                pass

    def _process_once(self):
        try:
            # first frame
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
                # run on current frame
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
                    self._publish(algo.key, payload)
                    continue

                self._publish(algo.key, payload)

                # baseline debug saving (first frame)
                if algo.key == "pcb_detection":
                    self._maybe_save_pcb_baseline_debug(cv_bgr, res)

                # If pcb_detection is calibrating: capture more frames and feed them
                if algo.key == "pcb_detection" and isinstance(res, dict) and res.get("calibrating"):
                    remaining = max(0, self._pcb_baseline_samples - 1)
                    for _ in range(remaining):
                        time.sleep(self._pcb_baseline_interval_s)

                        t0b = time.perf_counter()
                        img_msg_b = self._grab_one_image()
                        grab_ms_b = (time.perf_counter() - t0b) * 1000.0
                        if img_msg_b is None:
                            self.get_logger().warn("PCB baseline: no image during calibration step")
                            break

                        t1b = time.perf_counter()
                        cv_bgr_b = self._bridge.imgmsg_to_cv2(img_msg_b, desired_encoding="bgr8")
                        convert_ms_b = (time.perf_counter() - t1b) * 1000.0

                        try:
                            tb = time.perf_counter()
                            res_b = algo.run(cv_bgr_b)
                            algo_ms_b = (time.perf_counter() - tb) * 1000.0
                            payload_b = {
                                "algo": algo.key,
                                "ok": True,
                                "grab_ms": grab_ms_b,
                                "convert_ms": convert_ms_b,
                                "algo_ms": algo_ms_b,
                                "result": res_b,
                            }
                        except Exception as e:
                            res_b = {}
                            payload_b = {
                                "algo": algo.key,
                                "ok": False,
                                "grab_ms": grab_ms_b,
                                "convert_ms": convert_ms_b,
                                "error": str(e),
                            }

                        self._publish(algo.key, payload_b)

                        # baseline debug saving (each calibration frame)
                        if isinstance(res_b, dict):
                            self._maybe_save_pcb_baseline_debug(cv_bgr_b, res_b)

                        if isinstance(res_b, dict) and not res_b.get("calibrating"):
                            break

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
