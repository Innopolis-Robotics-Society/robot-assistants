# gears_check.py
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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

from .gear_counter import GearCounterConfig, count_teeth, brighten_region


# -----------------------------
# Baseline store
# -----------------------------
@dataclass
class Baseline:
    expected_gears: int
    mean_teeth: List[float]       # baseline mean
    stat_tol: List[float]         # baseline statistical tolerance (info)
    n_samples: int                # number of calibration "sessions" added


class CalibrationStore:
    """
    Stores baseline as mean over appended calibration sessions.
    Each calibration service call adds ONE averaged vector (after per-call consensus).
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

        good = [s for s in self._samples if len(s) == m]
        if len(good) != len(self._samples):
            self._samples = good
            if not self._samples:
                self._baseline = None
                return

        arr = np.array(self._samples, dtype=np.float64)  # (n,m)
        mean = arr.mean(axis=0)
        dev = np.max(np.abs(arr - mean[None, :]), axis=0)
        stat_tol = np.maximum(self._tol_floor, dev + self._tol_margin)

        self._baseline = Baseline(
            expected_gears=int(m),
            mean_teeth=[float(x) for x in mean.tolist()],
            stat_tol=[float(x) for x in stat_tol.tolist()],
            n_samples=int(arr.shape[0]),
        )


# -----------------------------
# Consensus helpers
# -----------------------------
def _mode_int(vals: List[int]) -> int:
    return max(set(vals), key=vals.count) if vals else 0


def _consensus_mean(
    samples: List[List[float]],
    tol: float,
    need: int,
) -> Tuple[bool, List[float], List[int], Dict[str, Any]]:
    """
    Consensus selection:
      - median per gear
      - frame is inlier if max(|frame - median|) <= tol
      - if inliers >= need: consensus = mean(inliers)

    Returns:
      ok, consensus_vector, inlier_indices, debug
    """
    dbg: Dict[str, Any] = {}
    if not samples:
        return False, [], [], {"reason": "no_samples"}

    m = len(samples[0])
    same = [s for s in samples if len(s) == m]
    if len(same) != len(samples):
        samples = same
        if not samples:
            return False, [], [], {"reason": "no_consistent_length"}

    arr = np.array(samples, dtype=np.float64)  # (n,m)
    med = np.median(arr, axis=0)
    dev = np.max(np.abs(arr - med[None, :]), axis=1)  # per-frame max dev

    inliers = np.where(dev <= float(tol))[0].tolist()

    dbg["median"] = [float(x) for x in med.tolist()]
    dbg["frame_dev_max"] = [float(x) for x in dev.tolist()]
    dbg["inliers"] = inliers
    dbg["tol"] = float(tol)
    dbg["need"] = int(need)

    if len(inliers) < int(need):
        dbg["reason"] = "not_enough_inliers"
        return False, [], inliers, dbg

    cons = arr[inliers].mean(axis=0)
    return True, [float(x) for x in cons.tolist()], inliers, dbg


# -----------------------------
# Node
# -----------------------------
class GearInspectorNode(Node):
    def __init__(self) -> None:
        super().__init__("gears_check")
        self._cbg = ReentrantCallbackGroup()
        self._bridge = CvBridge()

        # ROS I/O
        self.declare_parameter("image_topic", "/rgb/image_raw")
        self.declare_parameter("frame_wait_timeout_sec", 2.0)

        # Samples per command
        self.declare_parameter("calib_samples", 20)
        self.declare_parameter("infer_samples", 20)

        # NEW: per-command consensus (noise filtering)
        self.declare_parameter("calib_consensus_tol", 1.0)
        self.declare_parameter("calib_consensus_need", 2)
        self.declare_parameter("infer_consensus_tol", 1.0)
        self.declare_parameter("infer_consensus_need", 2)

        # NEW: final compare tolerance vs baseline (your “2”)
        self.declare_parameter("baseline_abs_tol", 2.5)

        # Baseline statistical info tolerances
        self.declare_parameter("teeth_tol_floor", 1.0)
        self.declare_parameter("teeth_tol_margin", 0.5)

        # Output topics: keep PCB-like names if you want client compatibility
        # (change to ~/report, ~/annotated if you prefer)
        self.declare_parameter("out_image_topic", "~/image")
        self.declare_parameter("out_data_topic", "~/data")

        # Extra output: image after brighten only
        self.declare_parameter("out_bright_image_topic", "~/brightened")

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
        out_bright_topic = str(self.get_parameter("out_bright_image_topic").value)
        out_data_topic = str(self.get_parameter("out_data_topic").value)

        self._sub = self.create_subscription(Image, img_topic, self._on_image, 10, callback_group=self._cbg)
        self._pub_img = self.create_publisher(Image, out_img_topic, 10)
        self._pub_bright = self.create_publisher(Image, out_bright_topic, 10)
        self._pub_data = self.create_publisher(String, out_data_topic, 10)

        # Services (Trigger: success/message)
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
            f"gears_check started: image_topic={img_topic}, out_image={out_img_topic}, out_bright={out_bright_topic}, out_image={out_img_topic}, out_data={out_data_topic}"
        )

    # ---------------- report formatting (PCB-like) ----------------
    def _mk_report(self, command: str, *, ok: bool, reason: str, **extra) -> Dict[str, Any]:
        base = {
            "command": command,
            "baseline_set": bool(self._calib.is_set()),
            "calibrating": False,
            "match": (command == "calibration"),
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
                self._publish_report(self._mk_report("unknown", ok=False, reason="busy"))
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
            report = self._mk_report("reset", ok=True, reason="reset_done")
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
            self.get_logger().info(f"Калибровка: начало (frames={n}, expected_gears=4)")

            frames = self._capture_burst(n)
            if not frames:
                report = self._mk_report("calibration", ok=False, reason="no_image")
                self._publish_report(report)
                response.success = False
                response.message = "no_image"
                self.get_logger().info("Калибровка: конец (FAIL: no_image)")
                return response

            cfg = self._make_cfg()
            expected_gears = 4  # fixed expected number of gears during calibration

            per_frame: List[Dict[str, Any]] = []
            detections: List[List[float]] = []
            last_annot: Optional[np.ndarray] = None
            last_bright: Optional[np.ndarray] = None

            for i, img in enumerate(frames, start=1):
                try:
                    last_bright = brighten_region(img, cfg.brighten_region, factor=cfg.brighten_factor, blend=cfg.brighten_blend)
                except Exception:
                    last_bright = img
                res = count_teeth(img, cfg=cfg)
                last_annot = res.annotated_bgr

                if not res.ok:
                    self.get_logger().info(f"Калибровка: фото {i}/{len(frames)}: FAIL ({res.reason})")
                    per_frame.append({"i": i, "ok": False, "reason": res.reason, "gears": 0, "teeth": []})
                    continue

                # Discard frames that do not match expected number of gears during calibration
                if len(res.teeth) != expected_gears:
                    self.get_logger().info(
                        f"Калибровка: фото {i}/{len(frames)}: DISCARD (gears={len(res.teeth)} != expected {expected_gears})"
                    )
                    per_frame.append(
                        {"i": i, "ok": False, "reason": "gear_count_mismatch", "gears": len(res.teeth), "teeth": [float(x) for x in res.teeth]}
                    )
                    continue

                detections.append(res.teeth)
                per_frame.append({"i": i, "ok": True, "reason": "ok", "gears": len(res.teeth), "teeth": [float(x) for x in res.teeth]})
                self.get_logger().info(
                    f"Калибровка: фото {i}/{len(frames)}: OK, gears={len(res.teeth)}, teeth={[float(x) for x in res.teeth]}"
                )


            # Summary for all captured frames (all N photos)
            try:
                summary_lines = []
                for pf in per_frame:
                    if pf.get("ok"):
                        summary_lines.append(f"#{pf['i']}: OK gears={len(pf.get('teeth', []))} teeth={pf.get('teeth')}")
                    else:
                        summary_lines.append(f"#{pf['i']}: FAIL ({pf.get('reason')})")
                self.get_logger().info(f"Калибровка: сводка по кадрам (N={len(frames)}): " + " | ".join(summary_lines))
            except Exception:
                pass
            if not detections:
                # If we saw detections but with wrong gear count, report it explicitly.
                saw_mismatch = any(pf.get("reason") == "gear_count_mismatch" for pf in per_frame)
                fail_reason = "gear_count_mismatch" if saw_mismatch else "no_valid_detections"
                report = self._mk_report(
                    "calibration",
                    ok=False,
                    reason=fail_reason,
                    expected_gears=expected_gears,
                    per_frame=per_frame,
                )
                self._publish_report(report)
                if last_bright is not None:
                    self._publish_bright_image(self._overlay_text(last_bright, "BRIGHT"))
                if last_annot is not None:
                    self._publish_image(self._overlay_text(last_annot, "CALIB FAIL"))
                response.success = False
                response.message = fail_reason
                self.get_logger().info(f"Калибровка: конец (FAIL: {fail_reason})")
                return response

            # All detections already filtered to expected_gears during the loop
            mode_gears = expected_gears
            det_mode = detections

            need = int(self.get_parameter("calib_consensus_need").value)
            tol = float(self.get_parameter("calib_consensus_tol").value)

            ok, avg_teeth, inliers, dbg = _consensus_mean(det_mode, tol=tol, need=need)
            if not ok:
                report = self._mk_report(
                    "calibration",
                    ok=False,
                    reason="calib_no_consensus",
                    mode_gears=mode_gears,
                    detections_mode=len(det_mode),
                    consensus={"ok": False, "debug": dbg},
                    per_frame=per_frame,
                )
                self._publish_report(report)
                if last_bright is not None:
                    self._publish_bright_image(self._overlay_text(last_bright, "BRIGHT"))
                if last_annot is not None:
                    self._publish_image(self._overlay_text(last_annot, "CALIB FAIL"))
                response.success = False
                response.message = "calib_no_consensus"
                self.get_logger().info("Калибровка: конец (FAIL: calib_no_consensus)")
                return response

            before = self._calib.baseline().n_samples if self._calib.is_set() and self._calib.baseline() else 0
            self._calib.append_sample(avg_teeth)
            b = self._calib.baseline()
            assert b is not None

            reason = "baseline_updated" if before > 0 else "baseline_created"
            report = self._mk_report(
                "calibration",
                ok=True,
                reason=reason,
                progress={"have": len(inliers), "need": need},
                expected_gears=b.expected_gears,
                calib_avg_teeth=[float(x) for x in avg_teeth],
                calib_inliers=inliers,
                calib_consensus_tol=tol,
                calib_consensus_need=need,
                teeth_ref=b.mean_teeth,
                baseline_stat_tol=b.stat_tol,
                n_samples_total=b.n_samples,
                per_frame=per_frame,
            )
            self._publish_report(report)
            if last_bright is not None:
                self._publish_bright_image(self._overlay_text(last_bright, "BRIGHT"))
            if last_annot is not None:
                self._publish_image(self._overlay_text(last_annot, "CALIB OK"))

            response.success = True
            response.message = reason
            self.get_logger().info(f"Калибровка: конец (OK: {reason})")
            return response
        finally:
            self._leave_busy()

    def _srv_inference(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        if not self._try_enter_busy(response):
            return response
        try:
            log = self.get_logger()

            if not self._calib.is_set():
                log.info("Проверка: начало -> FAIL (baseline_not_set)")
                report = self._mk_report("inference", ok=False, reason="baseline_not_set")
                self._publish_report(report)
                response.success = False
                response.message = "baseline_not_set"
                return response

            b = self._calib.baseline()
            assert b is not None

            n = int(self.get_parameter("infer_samples").value)
            log.info(f"Проверка: начало (frames={n}, expected_gears={b.expected_gears}, baseline_mean={list(map(float, b.mean_teeth))})")

            frames = self._capture_burst(n)
            if not frames:
                log.info("Проверка: конец (FAIL: no_image)")
                report = self._mk_report("inference", ok=False, reason="no_image")
                self._publish_report(report)
                response.success = False
                response.message = "no_image"
                return response

            cfg = self._make_cfg()
            expected_gears = 4  # fixed expected number of gears during calibration

            per_frame: List[Dict[str, Any]] = []
            detections: List[List[float]] = []
            last_annot: Optional[np.ndarray] = None
            last_bright: Optional[np.ndarray] = None
            last_debug: Dict[str, Any] = {}

            for i, img in enumerate(frames, start=1):
                try:
                    last_bright = brighten_region(img, cfg.brighten_region, factor=cfg.brighten_factor, blend=cfg.brighten_blend)
                except Exception:
                    last_bright = img
                res = count_teeth(img, cfg=cfg)
                last_annot = res.annotated_bgr
                last_debug = res.debug

                if not res.ok:
                    per_frame.append({"i": i, "ok": False, "reason": res.reason, "gears": 0, "teeth": []})
                    log.info(f"Проверка: фото {i}/{n}: FAIL ({res.reason})")
                    continue

                teeth = [float(x) for x in res.teeth]
                per_frame.append({"i": i, "ok": True, "reason": "ok", "gears": len(teeth), "teeth": teeth})
                detections.append(teeth)
                log.info(f"Проверка: фото {i}/{n}: OK, gears={len(teeth)}, teeth={teeth}")


            # Summary for all captured frames (all N photos)
            try:
                summary_lines = []
                for pf in per_frame:
                    if pf.get("ok"):
                        g = pf.get("gears", len(pf.get("teeth", [])))
                        summary_lines.append(f"#{pf['i']}: OK gears={g} teeth={pf.get('teeth')}")
                    else:
                        summary_lines.append(f"#{pf['i']}: FAIL ({pf.get('reason')})")
                self.get_logger().info(f"Проверка: сводка по кадрам (N={n}): " + " | ".join(summary_lines))
            except Exception:
                pass
            if not detections:
                log.info("Проверка: конец (FAIL: no_valid_detections)")
                report = self._mk_report(
                    "inference",
                    ok=False,
                    reason="no_valid_detections",
                    per_frame=per_frame,
                    debug=last_debug,
                )
                self._publish_report(report)
                if last_bright is not None:
                    self._publish_bright_image(self._overlay_text(last_bright, "BRIGHT"))
                if last_annot is not None:
                    self._publish_image(self._overlay_text(last_annot, "FAIL"))
                response.success = False
                response.message = "no_valid_detections"
                return response

            counts = [len(x) for x in detections]
            gears_mode = _mode_int(counts)
            log.info(f"Проверка: валидных кадров={len(detections)}/{n}; gears_count={counts}; mode={gears_mode}; expected={b.expected_gears}")

            det_match = [x for x in detections if len(x) == b.expected_gears]
            if not det_match:
                log.info("Проверка: конец (FAIL: gear_count_mismatch)")
                report = self._mk_report(
                    "inference",
                    ok=False,
                    reason="gear_count_mismatch",
                    expected_gears=b.expected_gears,
                    gears_mode=gears_mode,
                    per_frame=per_frame,
                    debug=last_debug,
                )
                self._publish_report(report)
                if last_bright is not None:
                    self._publish_bright_image(self._overlay_text(last_bright, "BRIGHT"))
                if last_annot is not None:
                    self._publish_image(self._overlay_text(last_annot, "FAIL"))
                response.success = False
                response.message = "gear_count_mismatch"
                return response

            need = int(self.get_parameter("infer_consensus_need").value)
            tol = float(self.get_parameter("infer_consensus_tol").value)
            log.info(f"Проверка: консенсус -> start (match_expected={len(det_match)}, need={need}, tol={tol})")

            ok, avg_teeth, inliers, dbg = _consensus_mean(det_match, tol=tol, need=need)
            if not ok:
                log.info(f"Проверка: консенсус -> FAIL (infer_no_consensus), debug={dbg}")
                report = self._mk_report(
                    "inference",
                    ok=False,
                    reason="infer_no_consensus",
                    consensus={"ok": False, "debug": dbg},
                    expected_gears=b.expected_gears,
                    per_frame=per_frame,
                    debug=last_debug,
                )
                self._publish_report(report)
                if last_bright is not None:
                    self._publish_bright_image(self._overlay_text(last_bright, "BRIGHT"))
                if last_annot is not None:
                    self._publish_image(self._overlay_text(last_annot, "FAIL"))
                response.success = False
                response.message = "infer_no_consensus"
                return response

            avg_list = [float(x) for x in avg_teeth]
            log.info(f"Проверка: консенсус -> OK (inliers={len(inliers)}, avg_teeth={avg_list})")

            base_tol = float(self.get_parameter("baseline_abs_tol").value)

            per_gear = []
            mismatches = []
            overall_ok = True

            log.info(f"Проверка: сравнение с baseline (abs_tol={base_tol})")
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
                log.info(f"Проверка: gear {i}: ref={ref:.3f}, cur={cur:.3f}, diff={diff:+.3f}, tol={base_tol:.3f} -> {'OK' if ok_i else 'FAIL'}")

            reason = "ok" if overall_ok else "gears_different"
            log.info(f"Проверка: конец ({'OK' if overall_ok else 'FAIL'}: {reason})")

            report = self._mk_report(
                "inference",
                ok=overall_ok,
                reason=reason,
                expected_gears=b.expected_gears,
                teeth_ref=b.mean_teeth,
                teeth_cur=avg_list,
                tolerance={"abs_teeth": base_tol},
                mismatches=mismatches[:50],
                per_gear=per_gear,
                infer_inliers=inliers,
                infer_consensus_tol=tol,
                infer_consensus_need=need,
                baseline_stat_tol=b.stat_tol,
                per_frame=per_frame,
                debug=last_debug,
            )

            self._publish_report(report)
            if last_bright is not None:
                self._publish_bright_image(self._overlay_text(last_bright, "BRIGHT"))
            if last_annot is not None:
                self._publish_image(self._overlay_text(last_annot, "OK" if overall_ok else "FAIL"))

            response.success = bool(overall_ok)
            response.message = str(report["reason"])
            return response
        finally:
            self._leave_busy()

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


    def _publish_bright_image(self, bgr: np.ndarray) -> None:
        """Publish image after brighten (before further processing)."""
        try:
            msg = self._bridge.cv2_to_imgmsg(bgr, encoding="bgr8")
        except Exception as e:
            self.get_logger().warning(f"cv_bridge publish convert failed: {e}")
            return
        self._pub_bright.publish(msg)

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