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
from rclpy.qos import qos_profile_sensor_data

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_srvs.srv import Trigger

from .gear_counter import GearCounterConfig, count_teeth


# -----------------------------
# Baseline store
# -----------------------------
@dataclass
class Baseline:
    expected_gears: int
    mean_teeth: List[float]       # baseline mean
    stat_tol: List[float]         # informational tolerance from baseline history
    n_samples: int                # number of calibration sessions appended


class CalibrationStore:
    """
    Stores baseline as mean over appended calibration sessions.
    Each calibration service call appends ONE averaged vector (mean over good frames).
    """
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

    def append_sample(self, sample: List[float]) -> None:
        self._samples.append([float(x) for x in sample])
        self._recompute()

    def _recompute(self) -> None:
        if not self._samples:
            self._baseline = None
            return

        m = len(self._samples[0])
        if m == 0:
            self._baseline = None
            return

        # keep only same-length samples
        good = [s for s in self._samples if len(s) == m]
        if len(good) != len(self._samples):
            self._samples = good
            if not self._samples:
                self._baseline = None
                return

        arr = np.array(self._samples, dtype=np.float64)  # (n,m)
        mean = arr.mean(axis=0)

        # informational tolerance from max deviation + margin, floored
        dev = np.max(np.abs(arr - mean[None, :]), axis=0)
        stat_tol = np.maximum(self._tol_floor, dev + self._tol_margin)

        self._baseline = Baseline(
            expected_gears=int(m),
            mean_teeth=[float(x) for x in mean.tolist()],
            stat_tol=[float(x) for x in stat_tol.tolist()],
            n_samples=int(arr.shape[0]),
        )


# -----------------------------
# Node
# -----------------------------
class GearInspectorNode(Node):
    def __init__(self) -> None:
        super().__init__("gears_check")
        self._cbg = ReentrantCallbackGroup()
        self._bridge = CvBridge()

        # ROS I/O
        self.declare_parameter("image_topic", "/image_raw")
        self.declare_parameter("frame_wait_timeout_sec", 2.0)

        # Samples per command
        self.declare_parameter("calib_samples", 5)
        self.declare_parameter("infer_samples", 5)

        # Inference: keep reading stream until first good frame is found (limits)
        self.declare_parameter("infer_find_timeout_sec", 5.0)  # seconds
        self.declare_parameter("infer_max_attempts", 50)       # frames

        # Accept only frames with exactly this gear count
        self.declare_parameter("required_gears", 4)

        # Kept for compatibility (NOT used)
        self.declare_parameter("calib_consensus_tol", 1.0)
        self.declare_parameter("calib_consensus_need", 3)
        self.declare_parameter("infer_consensus_tol", 1.0)
        self.declare_parameter("infer_consensus_need", 3)

        # Final compare tolerance vs baseline
        self.declare_parameter("baseline_abs_tol", 2.0)

        # Baseline statistical info tolerances
        self.declare_parameter("teeth_tol_floor", 1.0)
        self.declare_parameter("teeth_tol_margin", 0.5)

        # Output topics
        self.declare_parameter("out_image_topic", "~/image")
        self.declare_parameter("out_data_topic", "~/data")

        # ---------------- Algorithm params (mapped to GearCounterConfig) ----------------
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
        out_img_topic = str(self.get_parameter("out_image_topic").value)
        out_data_topic = str(self.get_parameter("out_data_topic").value)

        # IMPORTANT: sensor_data QoS to match camera streams
        self._sub = self.create_subscription(
            Image,
            img_topic,
            self._on_image,
            qos_profile_sensor_data,
            callback_group=self._cbg
        )
        self._pub_img = self.create_publisher(Image, out_img_topic, 10)
        self._pub_data = self.create_publisher(String, out_data_topic, 10)

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

        # Baseline store
        tol_floor = float(self.get_parameter("teeth_tol_floor").value)
        tol_margin = float(self.get_parameter("teeth_tol_margin").value)
        self._calib = CalibrationStore(tol_floor=tol_floor, tol_margin=tol_margin)

        self.get_logger().info(
            f"gears_check started: image_topic={img_topic}, out_image={out_img_topic}, out_data={out_data_topic}"
        )

    # ---------------- report formatting ----------------
    def _mk_report(self, command: str, *, ok: bool, reason: str, **extra) -> Dict[str, Any]:
        base = {
            "command": command,
            "baseline_set": bool(self._calib.is_set()),
            "ok": bool(ok),
            "reason": str(reason),
        }
        base.update(extra)
        return base

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
        """
        Capture N frames from the stream. Frames are considered "new" when stamp changes.
        """
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

    def _wait_next_frame(self, last_stamp_ns: int, timeout_sec: float) -> Optional[np.ndarray]:
        """
        Wait for the next frame (stamp changes). Returns BGR image or None.
        """
        with self._img_cond:
            t0 = time.time()
            while self._img_cv is None and (time.time() - t0) < timeout_sec:
                self._img_cond.wait(timeout=0.05)
            if self._img_cv is None:
                return None

            while self._img_stamp_ns == last_stamp_ns and (time.time() - t0) < timeout_sec:
                self._img_cond.wait(timeout=0.02)

            if self._img_cv is None:
                return None
            return self._img_cv.copy()

    # ---------------- busy gate ----------------
    def _try_enter_busy(self, response: Trigger.Response, command: str) -> bool:
        with self._busy_lock:
            if self._busy:
                response.success = False
                response.message = "busy"
                self._publish_report(self._mk_report(command, ok=False, reason="busy"))
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

    # ---------------- mean over good frames ----------------
    @staticmethod
    def _mean_vectors(vectors: List[List[float]]) -> List[float]:
        arr = np.array(vectors, dtype=np.float64)  # (n,m)
        mean = arr.mean(axis=0)
        return [float(x) for x in mean.tolist()]

    # ---------------- publish helpers ----------------
    def _publish_report(self, report: Dict[str, Any]) -> None:
        msg = String()
        msg.data = json.dumps(report, ensure_ascii=False)
        self._pub_data.publish(msg)

    def _publish_image(self, bgr: np.ndarray) -> None:
        try:
            msg = self._bridge.cv2_to_imgmsg(bgr, encoding="bgr8")
        except Exception as e:
            self.get_logger().warning(f"cv_bridge publish convert failed: {e}")
            return
        self._pub_img.publish(msg)

    @staticmethod
    def _overlay_text(img: np.ndarray, text: str) -> np.ndarray:
        out = img.copy()
        cv2.putText(out, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(out, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 1, cv2.LINE_AA)
        return out

    # ---------------- services ----------------
    def _srv_reset(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        if not self._try_enter_busy(response, "reset"):
            return response
        try:
            self._calib.reset()
            report = self._mk_report("reset", ok=True, reason="reset_done")
            self._publish_report(report)
            response.success = True
            response.message = "reset_done"
            return response
        finally:
            self._leave_busy()

    def _srv_calibration(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        if not self._try_enter_busy(response, "calibration"):
            return response
        try:
            required_gears = int(self.get_parameter("required_gears").value)

            # baseline gear count must match
            if self._calib.is_set():
                b0 = self._calib.baseline()
                if b0 is not None and b0.expected_gears != required_gears:
                    report = self._mk_report(
                        "calibration",
                        ok=False,
                        reason="baseline_expected_gears_mismatch_required_gears",
                        expected_gears=b0.expected_gears,
                        required_gears=required_gears,
                    )
                    self._publish_report(report)
                    response.success = False
                    response.message = "baseline_expected_gears_mismatch_required_gears"
                    return response

            n = int(self.get_parameter("calib_samples").value)
            frames = self._capture_burst(n)
            if not frames:
                report = self._mk_report("calibration", ok=False, reason="no_image")
                self._publish_report(report)
                response.success = False
                response.message = "no_image"
                return response

            cfg = self._make_cfg()

            per_frame: List[Dict[str, Any]] = []
            good_vectors: List[List[float]] = []
            last_annot: Optional[np.ndarray] = None

            for i, img in enumerate(frames, start=1):
                res = count_teeth(img, cfg=cfg)
                last_annot = res.annotated_bgr

                if not res.ok:
                    self.get_logger().info(f"calibration: frame {i}/{len(frames)} rejected ({res.reason})")
                    per_frame.append({"i": i, "ok": False, "reason": res.reason, "gears": len(res.teeth), "teeth": []})
                    continue

                gears = len(res.teeth)
                if gears != required_gears:
                    self.get_logger().info(f"calibration: frame {i}/{len(frames)} ignored (gears={gears} != {required_gears})")
                    per_frame.append({"i": i, "ok": False, "reason": "gears_not_required", "gears": gears, "teeth": []})
                    continue

                vec = [float(x) for x in res.teeth]
                good_vectors.append(vec)
                per_frame.append({"i": i, "ok": True, "reason": "ok", "gears": gears, "teeth": vec})
                self.get_logger().info(f"calibration: frame {i}/{len(frames)} accepted gears={gears}")

            # SUCCESS if >= 1 good frame
            if not good_vectors:
                report = self._mk_report(
                    "calibration",
                    ok=False,
                    reason="no_frames_with_required_gears",
                    required_gears=required_gears,
                    per_frame=per_frame,
                )
                self._publish_report(report)
                if last_annot is not None:
                    self._publish_image(self._overlay_text(last_annot, "CALIB FAIL"))
                response.success = False
                response.message = "no_frames_with_required_gears"
                return response

            avg_teeth = self._mean_vectors(good_vectors)
            before = self._calib.baseline().n_samples if self._calib.is_set() and self._calib.baseline() else 0
            self._calib.append_sample(avg_teeth)
            b = self._calib.baseline()
            assert b is not None

            reason = "baseline_updated" if before > 0 else "baseline_created"
            report = self._mk_report(
                "calibration",
                ok=True,
                reason=reason,
                required_gears=required_gears,
                expected_gears=b.expected_gears,
                averaging={"method": "mean_over_good_frames", "n_good": len(good_vectors), "n_total": len(frames)},
                calib_avg_teeth=[float(x) for x in avg_teeth],
                teeth_ref=b.mean_teeth,
                baseline_stat_tol=b.stat_tol,
                n_samples_total=b.n_samples,
                per_frame=per_frame,
                note="consensus_disabled",
            )
            self._publish_report(report)
            if last_annot is not None:
                self._publish_image(self._overlay_text(last_annot, "CALIB OK"))

            response.success = True
            response.message = reason
            return response
        finally:
            self._leave_busy()

    def _srv_inference(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        if not self._try_enter_busy(response, "inference"):
            return response
        try:
            if not self._calib.is_set():
                report = self._mk_report("inference", ok=False, reason="baseline_not_set")
                self._publish_report(report)
                response.success = False
                response.message = "baseline_not_set"
                return response

            required_gears = int(self.get_parameter("required_gears").value)
            b = self._calib.baseline()
            assert b is not None

            if b.expected_gears != required_gears:
                report = self._mk_report(
                    "inference",
                    ok=False,
                    reason="baseline_expected_gears_mismatch_required_gears",
                    expected_gears=b.expected_gears,
                    required_gears=required_gears,
                )
                self._publish_report(report)
                response.success = False
                response.message = "baseline_expected_gears_mismatch_required_gears"
                return response

            infer_samples = int(self.get_parameter("infer_samples").value)
            frame_timeout = float(self.get_parameter("frame_wait_timeout_sec").value)
            find_timeout = float(self.get_parameter("infer_find_timeout_sec").value)
            max_attempts = int(self.get_parameter("infer_max_attempts").value)

            cfg = self._make_cfg()

            per_frame: List[Dict[str, Any]] = []
            good_vectors: List[List[float]] = []
            last_annot: Optional[np.ndarray] = None
            last_debug: Dict[str, Any] = {}

            # start stamp
            with self._img_cond:
                last_stamp = self._img_stamp_ns

            # 1) Find first good frame (gears == required_gears)
            t_start = time.time()
            found = False
            attempt = 0

            while attempt < max_attempts and (time.time() - t_start) < find_timeout:
                attempt += 1
                img = self._wait_next_frame(last_stamp, timeout_sec=frame_timeout)
                if img is None:
                    per_frame.append({"i": attempt, "ok": False, "reason": "frame_timeout", "gears": 0, "teeth": []})
                    continue

                with self._img_cond:
                    last_stamp = self._img_stamp_ns

                res = count_teeth(img, cfg=cfg)
                last_annot = res.annotated_bgr
                last_debug = res.debug

                if not res.ok:
                    per_frame.append({"i": attempt, "ok": False, "reason": res.reason, "gears": len(res.teeth), "teeth": []})
                    continue

                gears = len(res.teeth)
                if gears != required_gears:
                    per_frame.append({"i": attempt, "ok": False, "reason": "gears_not_required", "gears": gears, "teeth": []})
                    continue

                vec = [float(x) for x in res.teeth]
                good_vectors.append(vec)
                per_frame.append({"i": attempt, "ok": True, "reason": "ok", "gears": gears, "teeth": vec})
                found = True
                break

            if not found:
                report = self._mk_report(
                    "inference",
                    ok=False,
                    reason="no_frames_with_required_gears",
                    required_gears=required_gears,
                    per_frame=per_frame,
                    debug=last_debug,
                    note="inference_waited_for_good_frame_but_not_found",
                )
                self._publish_report(report)
                if last_annot is not None:
                    self._publish_image(self._overlay_text(last_annot, "FAIL"))
                response.success = False
                response.message = "no_frames_with_required_gears"
                return response

            # 2) Collect more frames (infer_samples-1), keep only good ones
            remaining = max(0, infer_samples - 1)
            for k in range(remaining):
                img = self._wait_next_frame(last_stamp, timeout_sec=frame_timeout)
                if img is None:
                    per_frame.append({"i": attempt + k + 1, "ok": False, "reason": "frame_timeout", "gears": 0, "teeth": []})
                    continue

                with self._img_cond:
                    last_stamp = self._img_stamp_ns

                res = count_teeth(img, cfg=cfg)
                last_annot = res.annotated_bgr
                last_debug = res.debug

                if not res.ok:
                    per_frame.append({"i": attempt + k + 1, "ok": False, "reason": res.reason, "gears": len(res.teeth), "teeth": []})
                    continue

                gears = len(res.teeth)
                if gears != required_gears:
                    per_frame.append({"i": attempt + k + 1, "ok": False, "reason": "gears_not_required", "gears": gears, "teeth": []})
                    continue

                vec = [float(x) for x in res.teeth]
                good_vectors.append(vec)
                per_frame.append({"i": attempt + k + 1, "ok": True, "reason": "ok", "gears": gears, "teeth": vec})

            # good_vectors is guaranteed >= 1
            avg_teeth = self._mean_vectors(good_vectors)

            # 3) Compare with baseline
            base_tol = float(self.get_parameter("baseline_abs_tol").value)

            per_gear = []
            mismatches = []
            overall_ok = True

            for i in range(b.expected_gears):
                ref = float(b.mean_teeth[i])
                cur = float(avg_teeth[i])
                diff = cur - ref
                ok_i = abs(diff) <= base_tol
                overall_ok = overall_ok and ok_i
                item = {"index": i, "ref": ref, "cur": cur, "diff": diff, "tol": base_tol, "ok": ok_i}
                per_gear.append(item)
                if not ok_i:
                    mismatches.append(item)

            report = self._mk_report(
                "inference",
                ok=overall_ok,
                reason="ok" if overall_ok else "gears_different",
                required_gears=required_gears,
                expected_gears=b.expected_gears,
                teeth_ref=b.mean_teeth,
                teeth_cur=[float(x) for x in avg_teeth],
                averaging={"method": "mean_over_good_frames", "n_good": len(good_vectors)},
                tolerance={"abs_teeth": base_tol},
                mismatches=mismatches[:50],
                per_gear=per_gear,
                per_frame=per_frame,
                debug=last_debug,
                note="inference_waits_until_first_good_then_collects_more",
            )

            self._publish_report(report)
            if last_annot is not None:
                self._publish_image(self._overlay_text(last_annot, "OK" if overall_ok else "FAIL"))

            response.success = bool(overall_ok)
            response.message = str(report["reason"])
            return response
        finally:
            self._leave_busy()


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
