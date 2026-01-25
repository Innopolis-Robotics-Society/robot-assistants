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
from std_msgs.msg import String
from std_srvs.srv import Trigger

from cv_bridge import CvBridge

from iros_custom_msgs.msg import Detection2D, PcbCheck, Statistics


class CvAlgorithmsNode(Node):
    def __init__(self):
        super().__init__("cv_algorithms_node")

        # params
        self.declare_parameter("image_topic", "/image")
        self.declare_parameter("image_timeout_s", 0.5)

        self.declare_parameter("trigger_service", "/cv_algorithms/run")

        self.declare_parameter("result_prefix", "/cv_algorithms/result")

        self.declare_parameter("mode", "trigger")  # "trigger" | "timer"
        self.declare_parameter("process_period_s", 1.0)

        # pcb baseline sampling
        self.declare_parameter("pcb_baseline_samples", 5)
        self.declare_parameter("pcb_baseline_interval_s", 1.0)

        # debug saving for pcb baseline
        self.declare_parameter("save_pcb_baseline_debug", True)
        self.declare_parameter("debug_images_dir", "/tmp//iros_cv_algorithms/debug_images")

        # overlay publishing
        self.declare_parameter("publish_overlay", True)
        self.declare_parameter("overlay_prefix", "/cv_algorithms/overlay")

        # read params
        self._image_topic = self.get_parameter("image_topic").value
        self._image_timeout_s = float(self.get_parameter("image_timeout_s").value)
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

        self._publish_overlay = bool(self.get_parameter("publish_overlay").value)
        self._overlay_prefix = str(self.get_parameter("overlay_prefix").value)

        # triggers / timer
        if self._mode == "trigger":
            self.create_service(Trigger, self._trigger_service, self._on_trigger_service)
        elif self._mode == "timer":
            self._timer = self.create_timer(self._process_period_s, self._on_timer)
        else:
            self.get_logger().warn(f"Unknown mode '{self._mode}', fallback to trigger")
            self._mode = "trigger"
            self.create_service(Trigger, self._trigger_service, self._on_trigger_service)

        # algorithms
        from iros_cv_algorithms.algos.corner_detection import CornerDetectionAlgorithm
        from iros_cv_algorithms.algos.rust_detection import RustDetectionAlgorithm
        from iros_cv_algorithms.algos.pcb_detection import PCBDetectionAlgorithm

        self._algorithms = [
            CornerDetectionAlgorithm(),
            RustDetectionAlgorithm(),
            PCBDetectionAlgorithm(calib_samples=self._pcb_baseline_samples),
        ]

        # publishers:
        # - pcb_detection -> custom msg
        self._pub_pcb = self.create_publisher(PcbCheck, f"{self._result_prefix}/pcb_detection", 10)

        # - others -> json string
        self._pubs_str = {
            algo.key: self.create_publisher(String, f"{self._result_prefix}/{algo.key}", 10)
            for algo in self._algorithms
            if algo.key != "pcb_detection"
        }

        # - overlay images (bgr8), per-algorithm:
        #   /cv_algorithms/overlay/<algo_key>/image
        self._pubs_overlay = {}
        if self._publish_overlay:
            for algo in self._algorithms:
                topic = f"{self._overlay_prefix}/{algo.key}/image"
                self._pubs_overlay[algo.key] = self.create_publisher(Image, topic, qos_profile_sensor_data)

        self._bridge = CvBridge()

        self._busy_lock = threading.Lock()
        self._busy = False

        self.get_logger().info(
            f"Ready. mode={self._mode}, image_topic={self._image_topic}, timeout={self._image_timeout_s}s, "
            f"trigger_service={self._trigger_service}, result_prefix={self._result_prefix}"
        )
        if self._publish_overlay:
            self.get_logger().info(f"Overlay enabled. overlay_prefix={self._overlay_prefix}")
        if self._save_pcb_baseline_debug:
            self.get_logger().info(f"PCB baseline debug dir: {self._debug_images_dir}")

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

    def _publish_str(self, algo_key: str, payload: dict):
        msg_str = json.dumps(payload, ensure_ascii=False, default=str)
        self.get_logger().info(f"[{algo_key}] {msg_str}")
        self._pubs_str[algo_key].publish(String(data=msg_str))

    def _to_det_msg(self, d: dict) -> Detection2D:
        m = Detection2D()
        m.class_name = str(d.get("class", d.get("class_name", "")))
        m.x = float(d.get("x", 0.0))
        m.y = float(d.get("y", 0.0))
        m.w = float(d.get("w", 0.0))
        m.h = float(d.get("h", 0.0))
        m.conf = float(d.get("conf", 0.0))
        return m

    def _publish_pcb(self, img_msg: Image, grab_ms: float, convert_ms: float, algo_ms: float, algo_ok: bool, res: dict):
        msg = PcbCheck()
        msg.header = img_msg.header

        msg.stats = Statistics()
        msg.stats.grab_ms = float(grab_ms)
        msg.stats.convert_ms = float(convert_ms)
        msg.stats.algo_ms = float(algo_ms)

        msg.algo_ok = bool(algo_ok)

        msg.ok = bool(res.get("ok", False))
        msg.match = bool(res.get("match", False))
        msg.baseline_set = bool(res.get("baseline_set", False))
        msg.calibrating = bool(res.get("calibrating", False))
        msg.reason = str(res.get("reason", ""))

        msg.anchor_class = str(res.get("anchor_class", ""))

        anchor = res.get("anchor")
        if isinstance(anchor, dict):
            msg.has_anchor = True
            msg.anchor = self._to_det_msg(anchor)
        else:
            msg.has_anchor = False
            msg.anchor = Detection2D()

        preds = res.get("pred", [])
        if isinstance(preds, list):
            msg.pred = [self._to_det_msg(p) for p in preds if isinstance(p, dict)]
        else:
            msg.pred = []

        # логгер "то же самое сообщение"
        self.get_logger().info(f"[pcb_detection] {msg}")

        self._pub_pcb.publish(msg)

    # ---------------- overlay ----------------

    def _draw_overlay_from_res(self, image_bgr, res: dict):
        """
        Универсально для детекций в формате:
          res["pred"] = list[{class,x,y,w,h,conf}] (xywhn)
        Опционально:
          res["anchor"] = {class,x,y,w,h,conf} (xywhn) -> рисуем красным
        """
        if not isinstance(res, dict):
            return None

        preds = res.get("pred")
        if not isinstance(preds, list) or len(preds) == 0:
            return None

        img = image_bgr.copy()
        h, w = img.shape[:2]

        # anchor (optional)
        anchor = res.get("anchor") if isinstance(res.get("anchor"), dict) else None
        if anchor is not None:
            try:
                ax = float(anchor.get("x", 0.0))
                ay = float(anchor.get("y", 0.0))
                aw = float(anchor.get("w", 0.0))
                ah = float(anchor.get("h", 0.0))
                acls = str(anchor.get("class", anchor.get("class_name", "")))
                aconf = float(anchor.get("conf", 0.0))

                x1 = int((ax - aw / 2.0) * w)
                y1 = int((ay - ah / 2.0) * h)
                x2 = int((ax + aw / 2.0) * w)
                y2 = int((ay + ah / 2.0) * h)
                x1 = max(0, min(w - 1, x1))
                y1 = max(0, min(h - 1, y1))
                x2 = max(0, min(w - 1, x2))
                y2 = max(0, min(h - 1, y2))

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                label = f"ANCHOR {acls} {aconf:.2f}".strip()
                cv2.putText(img, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            except Exception:
                pass

        # detections
        for p in preds:
            if not isinstance(p, dict):
                continue
            try:
                cls = str(p.get("class", p.get("class_name", "")))
                conf = float(p.get("conf", 0.0))
                x = float(p["x"])
                y = float(p["y"])
                bw = float(p["w"])
                bh = float(p["h"])
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
            label = f"{cls} {conf:.2f}".strip() if cls else f"{conf:.2f}"
            cv2.putText(img, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # small status text (optional)
        reason = res.get("reason")
        if isinstance(reason, str) and reason:
            try:
                cv2.putText(img, reason, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            except Exception:
                pass

        return img

    def _publish_overlay_image(self, algo_key: str, img_msg: Image, overlay_bgr):
        if not self._publish_overlay:
            return
        pub = self._pubs_overlay.get(algo_key)
        if pub is None:
            return
        out_msg = self._bridge.cv2_to_imgmsg(overlay_bgr, encoding="bgr8")
        out_msg.header = img_msg.header
        pub.publish(out_msg)

    # ---------------- pcb debug saving ----------------

    def _draw_and_save(self, image_bgr, preds: list, out_path: Path):
        img = image_bgr.copy()
        h, w = img.shape[:2]
        for p in preds or []:
            try:
                cls = str(p.get("class", p.get("class_name", "")))
                conf = float(p.get("conf", 0.0))
                x = float(p["x"])
                y = float(p["y"])
                bw = float(p["w"])
                bh = float(p["h"])
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
            label = f"{cls} {conf:.2f}".strip() if cls else f"{conf:.2f}"
            cv2.putText(img, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imwrite(str(out_path), img)

    def _maybe_save_pcb_baseline_debug(self, image_bgr, res: dict):
        if not self._save_pcb_baseline_debug or not isinstance(res, dict):
            return

        # сохраняем только эталон (match=True)
        if res.get("match") is not True:
            return

        preds = res.get("pred", [])
        stamp = time.strftime("%Y%m%d_%H%M%S")

        if res.get("calibrating"):
            have = res.get("progress", {}).get("have")
            need = res.get("progress", {}).get("need")
            suffix = f"{int(have):02d}_of_{int(need):02d}" if have is not None and need is not None else "step"
            out_img = self._debug_images_dir / f"pcb_baseline_{stamp}_{suffix}.jpg"
            self._draw_and_save(image_bgr, preds, out_img)

        if res.get("reason") == "baseline_created" and res.get("baseline_set"):
            out_img = self._debug_images_dir / f"pcb_baseline_{stamp}_final.jpg"
            self._draw_and_save(image_bgr, preds, out_img)

    # ---------------- main loop ----------------

    def _process_once(self):
        try:
            t0 = time.perf_counter()
            img_msg = self._grab_one_image()
            grab_ms = (time.perf_counter() - t0) * 1000.0

            if img_msg is None:
                self.get_logger().warn(f"No image received within {self._image_timeout_s}s")
                return

            t1 = time.perf_counter()
            cv_bgr = self._bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
            convert_ms = (time.perf_counter() - t1) * 1000.0

            for algo in self._algorithms:
                # run
                try:
                    t = time.perf_counter()
                    res = algo.run(cv_bgr)
                    algo_ms = (time.perf_counter() - t) * 1000.0
                    algo_ok = True
                except Exception as e:
                    res = {"ok": False, "reason": f"exception: {e}"}
                    algo_ms = 0.0
                    algo_ok = False

                # publish result
                if algo.key == "pcb_detection":
                    self._publish_pcb(img_msg, grab_ms, convert_ms, algo_ms, algo_ok, res)

                    if isinstance(res, dict):
                        # overlay
                        overlay = self._draw_overlay_from_res(cv_bgr, res)
                        if overlay is not None:
                            self._publish_overlay_image(algo.key, img_msg, overlay)

                        # baseline debug images
                        self._maybe_save_pcb_baseline_debug(cv_bgr, res)
                else:
                    payload = {
                        "algo": algo.key,
                        "ok": algo_ok,
                        "grab_ms": grab_ms,
                        "convert_ms": convert_ms,
                        "algo_ms": algo_ms,
                        "result": res,
                    }
                    self._publish_str(algo.key, payload)

                    if isinstance(res, dict):
                        overlay = self._draw_overlay_from_res(cv_bgr, res)
                        if overlay is not None:
                            self._publish_overlay_image(algo.key, img_msg, overlay)

                # pcb calibration extra frames
                if algo.key == "pcb_detection" and isinstance(res, dict) and res.get("calibrating"):
                    remaining = max(0, self._pcb_baseline_samples - 1)
                    for _ in range(remaining):
                        time.sleep(self._pcb_baseline_interval_s)

                        img_msg_b = self._grab_one_image()
                        if img_msg_b is None:
                            break

                        t1b = time.perf_counter()
                        cv_bgr_b = self._bridge.imgmsg_to_cv2(img_msg_b, desired_encoding="bgr8")
                        convert_ms_b = (time.perf_counter() - t1b) * 1000.0

                        try:
                            tb = time.perf_counter()
                            res_b = algo.run(cv_bgr_b)
                            algo_ms_b = (time.perf_counter() - tb) * 1000.0
                            algo_ok_b = True
                        except Exception as e:
                            res_b = {"ok": False, "reason": f"exception: {e}"}
                            algo_ms_b = 0.0
                            algo_ok_b = False

                        # grab_ms_b не пересчитываю (это второй кадр); можно оставить 0
                        self._publish_pcb(img_msg_b, 0.0, convert_ms_b, algo_ms_b, algo_ok_b, res_b)

                        if isinstance(res_b, dict):
                            overlay_b = self._draw_overlay_from_res(cv_bgr_b, res_b)
                            if overlay_b is not None:
                                self._publish_overlay_image(algo.key, img_msg_b, overlay_b)

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
