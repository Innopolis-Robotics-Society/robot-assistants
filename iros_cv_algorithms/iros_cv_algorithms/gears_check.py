# gears_check.py
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_srvs.srv import Trigger

from .gear_counter import GearCounterConfig, count_teeth


@dataclass
class Baseline:
    expected_gears: int
    mean_teeth: List[float]
    tol: List[float]
    n_samples: int


class CalibrationStore:
    def __init__(self, tol_floor: float, tol_margin: float) -> None:
        self._tol_floor = float(tol_floor)
        self._tol_margin = float(tol_margin)
        self._samples: List[List[float]] = []
        self._baseline: Optional[Baseline] = None

    def reset(self) -> None:
        self._samples.clear()
        self._baseline = None

    def is_set(self) -> bool:
        return self._baseline is not None

    def baseline(self) -> Optional[Baseline]:
        return self._baseline

    def append_samples(self, samples: List[List[float]]) -> None:
        for s in samples:
            self._samples.append([float(x) for x in s])
        self._recompute()

    def _recompute(self) -> None:
        if not self._samples:
            self._baseline = None
            return

        m = len(self._samples[0])
        if m == 0:
            self._baseline = None
            return

        good = [s for s in self._samples if len(s) == m]
        if len(good) != len(self._samples):
            self._samples = good
            if not self._samples:
                self._baseline = None
                return

        arr = np.array(self._samples, dtype=np.float64)  # (n,m)
        mean = arr.mean(axis=0)
        dev = np.max(np.abs(arr - mean[None, :]), axis=0)
        tol = np.maximum(self._tol_floor, dev + self._tol_margin)

        self._baseline = Baseline(
            expected_gears=int(m),
            mean_teeth=[float(x) for x in mean.tolist()],
            tol=[float(x) for x in tol.tolist()],
            n_samples=int(arr.shape[0]),
        )

    def check(self, teeth: List[float]) -> Dict[str, Any]:
        b = self._baseline
        if b is None:
            return {"overall_ok": False, "reason": "baseline_not_set"}

        cur = [float(x) for x in teeth]
        if len(cur) != b.expected_gears:
            return {
                "overall_ok": False,
                "reason": "gear_count_mismatch",
                "expected_gears": b.expected_gears,
                "gears_cur": len(cur),
                "teeth_ref": b.mean_teeth,
                "teeth_cur": cur,
                "tolerance": b.tol,
            }

        per_gear = []
        ok_all = True
        for i in range(b.expected_gears):
            ref = float(b.mean_teeth[i])
            tol = float(b.tol[i])
            c = float(cur[i])
            diff = c - ref
            ok = abs(diff) <= tol
            ok_all = ok_all and ok
            per_gear.append({"index": i, "ref": ref, "cur": c, "diff": diff, "tol": tol, "ok": ok})

        return {
            "overall_ok": bool(ok_all),
            "reason": "ok" if ok_all else "teeth_mismatch",
            "expected_gears": b.expected_gears,
            "teeth_ref": b.mean_teeth,
            "teeth_cur": cur,
            "tolerance": b.tol,
            "per_gear": per_gear,
        }


class GearInspectorNode(Node):
    def __init__(self) -> None:
        super().__init__("gears_check")
        self._cbg = ReentrantCallbackGroup()
        self._bridge = CvBridge()

        # ROS params
        self.declare_parameter("image_topic", "/image_raw")
        self.declare_parameter("calib_samples", 5)
        self.declare_parameter("infer_samples", 3)
        self.declare_parameter("frame_wait_timeout_sec", 2.0)

        # tolerance policy
        self.declare_parameter("teeth_tol_floor", 1.0)
        self.declare_parameter("teeth_tol_margin", 0.5)
        self.declare_parameter("calib_min_accepted", 2)

        # Algorithm params (mapped to GearCounterConfig)
        self.declare_parameter("brighten_region", [0, 0, 550, 550])
        self.declare_parameter("brighten_factor", 1.2)
        self.declare_parameter("brighten_blend", True)
        self.declare_parameter("bilateral_d", 9)
        self.declare_parameter("bilateral_sigma_color", 75.0)
        self.declare_parameter("bilateral_sigma_space", 75.0)

        self.declare_parameter("hough_dp", 1.5)
        self.declare_parameter("hough_min_dist", 100.0)
        self.declare_parameter("hough_param1", 60.0)
        self.declare_parameter("hough_param2", 60.0)
        self.declare_parameter("hough_min_radius", 5)
        self.declare_parameter("hough_max_radius", 25)

        self.declare_parameter("use_clahe_for_hough", True)
        self.declare_parameter("clahe_clip_limit", 2.0)
        self.declare_parameter("clahe_tile_grid", 8)

        self.declare_parameter("max_neighbor_dist", 200.0)
        self.declare_parameter("sort_y_tol", 25)

        self.declare_parameter("big_r_min", 35)
        self.declare_parameter("big_r_max", 300)
        self.declare_parameter("small_r_min", 10)
        self.declare_parameter("small_r_max", 35)

        self.declare_parameter("big_r_max_from_neighbor_frac", 0.45)
        self.declare_parameter("small_r_max_from_neighbor_frac", 0.25)

        self.declare_parameter("radius_step", 1)
        self.declare_parameter("radius_thickness", 3)
        self.declare_parameter("radius_angles", 720)
        self.declare_parameter("canny1", 40)
        self.declare_parameter("canny2", 120)
        self.declare_parameter("blur_ksize", 5)
        self.declare_parameter("refine_subpixel", True)
        self.declare_parameter("radius_size_penalty", 0.0)

        self.declare_parameter("teeth_k1", 7.0)
        self.declare_parameter("teeth_k2_num", 1.15)
        self.declare_parameter("teeth_k2_den", 1.8)

        # Topics
        img_topic = str(self.get_parameter("image_topic").value)
        self._sub = self.create_subscription(Image, img_topic, self._on_image, 10, callback_group=self._cbg)
        self._pub_annot = self.create_publisher(Image, "~/annotated", 10)
        self._pub_report = self.create_publisher(String, "~/report", 10)

        # Services
        self._srv_calib = self.create_service(Trigger, "~/calibration", self._srv_calibration, callback_group=self._cbg)
        self._srv_infer = self.create_service(Trigger, "~/inference", self._srv_inference, callback_group=self._cbg)
        self._srv_reset = self.create_service(Trigger, "~/reset", self._srv_reset, callback_group=self._cbg)

        # Image buffer
        self._img_lock = threading.Lock()
        self._img_cv: Optional[np.ndarray] = None
        self._img_stamp_ns: int = 0
        self._img_cond = threading.Condition(self._img_lock)

        # Busy gate
        self._busy_lock = threading.Lock()
        self._busy = False

        # Calibration store
        tol_floor = float(self.get_parameter("teeth_tol_floor").value)
        tol_margin = float(self.get_parameter("teeth_tol_margin").value)
        self._calib = CalibrationStore(tol_floor=tol_floor, tol_margin=tol_margin)

        self.get_logger().info(f"gears_check started: image_topic={img_topic}")

    # ---------------- image buffering ----------------

    def _on_image(self, msg: Image) -> None:
        try:
            cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warning(f"cv_bridge convert failed: {e}")
            return
        stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        with self._img_cond:
            self._img_cv = cv_img
            self._img_stamp_ns = stamp_ns
            self._img_cond.notify_all()

    def _capture_burst(self, n: int) -> List[np.ndarray]:
        n = max(1, int(n))
        timeout = float(self.get_parameter("frame_wait_timeout_sec").value)
        out: List[np.ndarray] = []

        with self._img_cond:
            if self._img_cv is None:
                t0 = time.time()
                while self._img_cv is None and (time.time() - t0) < timeout:
                    self._img_cond.wait(timeout=0.05)
                if self._img_cv is None:
                    return []
            last = self._img_stamp_ns
            out.append(self._img_cv.copy())

        for _ in range(1, n):
            with self._img_cond:
                t0 = time.time()
                while self._img_stamp_ns == last and (time.time() - t0) < timeout:
                    self._img_cond.wait(timeout=0.02)
                last = self._img_stamp_ns
                if self._img_cv is not None:
                    out.append(self._img_cv.copy())
                else:
                    break
        return out

    # ---------------- busy gate ----------------

    def _try_enter_busy(self, response: Trigger.Response) -> bool:
        with self._busy_lock:
            if self._busy:
                response.success = False
                response.message = "busy"
                self._publish_report(
                    {"command": "unknown", "baseline_set": self._calib.is_set(), "overall_ok": False, "reason": "busy"}
                )
                return False
            self._busy = True
            return True

    def _leave_busy(self) -> None:
        with self._busy_lock:
            self._busy = False

    # ---------------- config mapping ----------------

    def _make_cfg(self) -> GearCounterConfig:
        reg = self.get_parameter("brighten_region").value
        if not isinstance(reg, (list, tuple)) or len(reg) != 4:
            reg = [0, 0, 550, 550]

        return GearCounterConfig(
            brighten_region=(int(reg[0]), int(reg[1]), int(reg[2]), int(reg[3])),
            brighten_factor=float(self.get_parameter("brighten_factor").value),
            brighten_blend=bool(self.get_parameter("brighten_blend").value),
            bilateral_d=int(self.get_parameter("bilateral_d").value),
            bilateral_sigma_color=float(self.get_parameter("bilateral_sigma_color").value),
            bilateral_sigma_space=float(self.get_parameter("bilateral_sigma_space").value),
            hough_dp=float(self.get_parameter("hough_dp").value),
            hough_min_dist=float(self.get_parameter("hough_min_dist").value),
            hough_param1=float(self.get_parameter("hough_param1").value),
            hough_param2=float(self.get_parameter("hough_param2").value),
            hough_min_radius=int(self.get_parameter("hough_min_radius").value),
            hough_max_radius=int(self.get_parameter("hough_max_radius").value),
            use_clahe_for_hough=bool(self.get_parameter("use_clahe_for_hough").value),
            clahe_clip_limit=float(self.get_parameter("clahe_clip_limit").value),
            clahe_tile_grid=int(self.get_parameter("clahe_tile_grid").value),
            max_neighbor_dist=float(self.get_parameter("max_neighbor_dist").value),
            sort_y_tol=int(self.get_parameter("sort_y_tol").value),
            big_r_min=int(self.get_parameter("big_r_min").value),
            big_r_max=int(self.get_parameter("big_r_max").value),
            small_r_min=int(self.get_parameter("small_r_min").value),
            small_r_max=int(self.get_parameter("small_r_max").value),
            big_r_max_from_neighbor_frac=float(self.get_parameter("big_r_max_from_neighbor_frac").value),
            small_r_max_from_neighbor_frac=float(self.get_parameter("small_r_max_from_neighbor_frac").value),
            radius_step=int(self.get_parameter("radius_step").value),
            radius_thickness=int(self.get_parameter("radius_thickness").value),
            radius_angles=int(self.get_parameter("radius_angles").value),
            canny1=int(self.get_parameter("canny1").value),
            canny2=int(self.get_parameter("canny2").value),
            blur_ksize=int(self.get_parameter("blur_ksize").value),
            refine_subpixel=bool(self.get_parameter("refine_subpixel").value),
            radius_size_penalty=float(self.get_parameter("radius_size_penalty").value),
            teeth_k1=float(self.get_parameter("teeth_k1").value),
            teeth_k2_num=float(self.get_parameter("teeth_k2_num").value),
            teeth_k2_den=float(self.get_parameter("teeth_k2_den").value),
        )

    # ---------------- services ----------------

    def _srv_reset(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        if not self._try_enter_busy(response):
            return response
        try:
            self._calib.reset()
            report = {"command": "reset", "baseline_set": False, "overall_ok": True, "reason": "reset_done"}
            self._publish_report(report)
            response.success = True
            response.message = "reset_done"
            return response
        finally:
            self._leave_busy()

    def _srv_calibration(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        if not self._try_enter_busy(response):
            return response
        try:
            n = int(self.get_parameter("calib_samples").value)
            frames = self._capture_burst(n)
            if not frames:
                report = {"command": "calibration", "baseline_set": self._calib.is_set(), "overall_ok": False, "reason": "no_image"}
                self._publish_report(report)
                response.success = False
                response.message = "no_image"
                return response

            cfg = self._make_cfg()

            accepted: List[List[float]] = []
            per_frame: List[Dict[str, Any]] = []
            last_annot: Optional[np.ndarray] = None

            for i, img in enumerate(frames, start=1):
                res = count_teeth(img, cfg=cfg)
                last_annot = res.annotated_bgr

                if not res.ok:
                    self.get_logger().info(f"calibration: frame {i}/{len(frames)} rejected ({res.reason})")
                    per_frame.append({"i": i, "ok": False, "reason": res.reason, "gears": len(res.teeth)})
                    continue

                accepted.append(res.teeth)
                per_frame.append({"i": i, "ok": True, "reason": "ok", "gears": len(res.teeth)})
                self.get_logger().info(
                    f"calibration: frame {i}/{len(frames)} accepted gears={len(res.teeth)} teeth={[round(t,2) for t in res.teeth]}"
                )

            if not accepted:
                report = {"command": "calibration", "baseline_set": self._calib.is_set(), "overall_ok": False, "reason": "no_accepted_frames", "per_frame": per_frame}
                self._publish_report(report)
                if last_annot is not None:
                    self._publish_annotated(self._overlay_text(last_annot, "CALIB FAIL"))
                response.success = False
                response.message = "no_accepted_frames"
                return response

            # Choose mode by gear count
            counts = [len(a) for a in accepted]
            mode_gears = max(set(counts), key=counts.count)
            accepted_mode = [a for a in accepted if len(a) == mode_gears]

            min_acc = int(self.get_parameter("calib_min_accepted").value)

            # If baseline doesn't exist yet, require enough consistent samples
            if not self._calib.is_set() and len(accepted_mode) < min_acc:
                report = {
                    "command": "calibration",
                    "baseline_set": False,
                    "overall_ok": False,
                    "reason": "not_enough_consistent_samples",
                    "mode_gears": mode_gears,
                    "accepted_mode": len(accepted_mode),
                    "need_at_least": min_acc,
                    "per_frame": per_frame,
                }
                self._publish_report(report)
                if last_annot is not None:
                    self._publish_annotated(self._overlay_text(last_annot, "CALIB FAIL"))
                response.success = False
                response.message = "not_enough_consistent_samples"
                return response

            # If baseline exists, require matching expected gear count
            if self._calib.is_set():
                b = self._calib.baseline()
                assert b is not None
                accepted_mode = [a for a in accepted_mode if len(a) == b.expected_gears]
                if not accepted_mode:
                    report = {
                        "command": "calibration",
                        "baseline_set": True,
                        "overall_ok": False,
                        "reason": "no_samples_matching_baseline_gear_count",
                        "expected_gears": b.expected_gears,
                        "per_frame": per_frame,
                    }
                    self._publish_report(report)
                    if last_annot is not None:
                        self._publish_annotated(self._overlay_text(last_annot, "CALIB FAIL"))
                    response.success = False
                    response.message = "no_samples_matching_baseline_gear_count"
                    return response

            before_n = self._calib.baseline().n_samples if self._calib.is_set() and self._calib.baseline() else 0
            self._calib.append_samples(accepted_mode)
            b2 = self._calib.baseline()
            assert b2 is not None

            report = {
                "command": "calibration",
                "baseline_set": True,
                "overall_ok": True,
                "reason": "baseline_updated" if before_n > 0 else "baseline_created",
                "expected_gears": b2.expected_gears,
                "teeth_ref": b2.mean_teeth,
                "tolerance": b2.tol,
                "n_samples_total": b2.n_samples,
                "accepted_frames": len(accepted_mode),
                "total_frames": len(frames),
                "per_frame": per_frame,
            }
            self._publish_report(report)
            if last_annot is not None:
                self._publish_annotated(self._overlay_text(last_annot, "CALIB OK"))
            response.success = True
            response.message = str(report["reason"])
            return response
        finally:
            self._leave_busy()

    def _srv_inference(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        if not self._try_enter_busy(response):
            return response
        try:
            if not self._calib.is_set():
                report = {"command": "inference", "baseline_set": False, "overall_ok": False, "reason": "baseline_not_set"}
                self._publish_report(report)
                response.success = False
                response.message = "baseline_not_set"
                return response

            n = int(self.get_parameter("infer_samples").value)
            frames = self._capture_burst(n)
            if not frames:
                report = {"command": "inference", "baseline_set": True, "overall_ok": False, "reason": "no_image"}
                self._publish_report(report)
                response.success = False
                response.message = "no_image"
                return response

            cfg = self._make_cfg()

            per_frame: List[Dict[str, Any]] = []
            overall_ok = True
            first_fail_reason = "ok"

            last_annot: Optional[np.ndarray] = None
            last_details: Dict[str, Any] = {}

            for i, img in enumerate(frames, start=1):
                res = count_teeth(img, cfg=cfg)
                last_annot = res.annotated_bgr

                if not res.ok:
                    details = {"overall_ok": False, "reason": res.reason, "gears_cur": len(res.teeth), "debug": res.debug}
                    frame_ok = False
                else:
                    details = self._calib.check(res.teeth)
                    details["debug"] = res.debug  # keep last debug for tuning
                    frame_ok = bool(details.get("overall_ok", False))

                per_frame.append({"i": i, "ok": frame_ok, "reason": details.get("reason", "unknown")})
                self.get_logger().info(f"inference: frame {i}/{len(frames)} ok={frame_ok} reason={details.get('reason')}")

                last_details = details
                if not frame_ok and overall_ok:
                    overall_ok = False
                    first_fail_reason = str(details.get("reason", "fail"))

            report = {
                "command": "inference",
                "baseline_set": True,
                "overall_ok": bool(overall_ok),
                "reason": "ok" if overall_ok else first_fail_reason,
                "per_frame": per_frame,
            }
            report.update(last_details)

            self._publish_report(report)
            if last_annot is not None:
                self._publish_annotated(self._overlay_text(last_annot, "OK" if overall_ok else "FAIL"))

            response.success = bool(overall_ok)
            response.message = str(report["reason"])
            return response
        finally:
            self._leave_busy()

    # ---------------- publish helpers ----------------

    def _publish_report(self, report: Dict[str, Any]) -> None:
        msg = String()
        msg.data = json.dumps(report, ensure_ascii=False)
        self._pub_report.publish(msg)

    def _publish_annotated(self, bgr: np.ndarray) -> None:
        try:
            msg = self._bridge.cv2_to_imgmsg(bgr, encoding="bgr8")
        except Exception as e:
            self.get_logger().warning(f"cv_bridge publish convert failed: {e}")
            return
        self._pub_annot.publish(msg)

    @staticmethod
    def _overlay_text(img: np.ndarray, text: str) -> np.ndarray:
        out = img.copy()
        cv2.putText(out, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(out, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 1, cv2.LINE_AA)
        return out


def main() -> None:
    rclpy.init()
    node = GearInspectorNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
