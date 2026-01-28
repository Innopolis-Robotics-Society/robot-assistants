#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS2 PCB Inspector Node (standalone, NO ROI)

Implements:
- Services: ~/calibration, ~/inference, ~/reset  (std_srvs/Trigger)
- Subscribes to latest frame from image_topic (sensor_msgs/Image)
- Runs YOLO (Ultralytics) on full frame
- Calibration:
    * burst N frames
    * requires identical class-count signature across accepted frames
    * builds baseline means + per-object tolerances (dx,dy + dw/dh rel + IoU min)
- Repeated calibration updates baseline (incremental mean + update maxima/minima)
- Inference:
    * burst M frames
    * checks each frame vs baseline
    * publishes annotated image and JSON report

Notes:
- Matching is by (class, sorted by x then y) index inside class.
- No timing constraints; service calls are rejected if node is busy.
"""

from __future__ import annotations

import json
import os
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


# ----------------------------- data structures -----------------------------


@dataclass(frozen=True)
class Det:
    cls: str
    x: float  # center x in [0..1]
    y: float  # center y in [0..1]
    w: float  # width  in [0..1]
    h: float  # height in [0..1]
    conf: float


@dataclass
class BoxMean:
    x: float
    y: float
    w: float
    h: float


@dataclass
class BaselineEntry:
    mean: BoxMean
    n: int = 0
    max_dx: float = 0.0
    max_dy: float = 0.0
    max_dw_rel: float = 0.0
    max_dh_rel: float = 0.0
    min_iou: float = 1.0
    dx_tol: float = 0.0
    dy_tol: float = 0.0
    dw_rel_tol: float = 0.0
    dh_rel_tol: float = 0.0
    iou_min: float = 0.0


# ----------------------------- math utils -----------------------------


def _rel_diff(a: float, b: float, eps: float = 1e-9) -> float:
    return abs(a - b) / max(abs(a), eps)


def _xywh_to_xyxy(x: float, y: float, w: float, h: float) -> Tuple[float, float, float, float]:
    return x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0


def _iou_xywh(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax1, ay1, ax2, ay2 = _xywh_to_xyxy(ax, ay, aw, ah)
    bx1, by1, bx2, by2 = _xywh_to_xyxy(bx, by, bw, bh)

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 1e-12 else 0.0


def _median(vals: List[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return float(s[mid])
    return float((s[mid - 1] + s[mid]) / 2.0)


# ----------------------------- node -----------------------------


class PCBInspectorNode(Node):
    def __init__(self) -> None:
        super().__init__("pcb_inspector")
        self._cbg = ReentrantCallbackGroup()
        self._bridge = CvBridge()

        # Topics / IO (as requested)
        self.declare_parameter("image_topic", "/image_raw")

        # YOLO (as requested)
        self.declare_parameter(
            "model_path",
            "/home/mobile/ros2_ws/src/iros_cv_algorithms/iros_cv_algorithms/algos/models/yolo11s_best.pt",
        )
        self.declare_parameter("conf_thr", 0.25)
        self.declare_parameter("iou_thr", 0.50)
        self.declare_parameter("device", "0")

        # Burst
        self.declare_parameter("calib_samples", 5)
        self.declare_parameter("infer_samples", 3)

        # Tolerances (normalized coords)
        self.declare_parameter("min_pos_tol", 0.01)
        self.declare_parameter("min_size_tol", 0.25)
        self.declare_parameter("min_iou", 0.75)
        self.declare_parameter("pos_margin", 0.005)
        self.declare_parameter("size_margin", 0.05)
        self.declare_parameter("iou_margin", 0.05)

        # Frame wait
        self.declare_parameter("frame_wait_timeout_sec", 2.0)

        # Publishing toggles
        self.declare_parameter("draw_expected_boxes", True)

        # ROS entities
        img_topic = str(self.get_parameter("image_topic").value)
        self._sub = self.create_subscription(Image, img_topic, self._on_image, 10, callback_group=self._cbg)
        self._pub_annot = self.create_publisher(Image, "~/annotated", 10)
        self._pub_report = self.create_publisher(String, "~/report", 10)

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

        # Baseline
        self._base_lock = threading.Lock()
        self._baseline: Optional[Dict[str, List[BaselineEntry]]] = None
        self._baseline_counts: Optional[Dict[str, int]] = None

        # Load YOLO
        model_path = str(self.get_parameter("model_path").value)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO model not found: {model_path}")

        from ultralytics import YOLO  # lazy import

        self._conf_thr = float(self.get_parameter("conf_thr").value)
        self._nms_iou_thr = float(self.get_parameter("iou_thr").value)
        self._device = str(self.get_parameter("device").value)
        self._model = YOLO(model_path)

        self.get_logger().info(f"PCBInspector started: image_topic={img_topic}, model={model_path}, device={self._device}")

    # ------------------------- ROS image buffer -------------------------

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
        out: List[np.ndarray] = []
        timeout = float(self.get_parameter("frame_wait_timeout_sec").value)

        with self._img_cond:
            if self._img_cv is None:
                t0 = time.time()
                while self._img_cv is None and (time.time() - t0) < timeout:
                    self._img_cond.wait(timeout=0.05)
                if self._img_cv is None:
                    return []
            last_stamp = self._img_stamp_ns
            out.append(self._img_cv.copy())

        for _ in range(1, n):
            with self._img_cond:
                t0 = time.time()
                while self._img_stamp_ns == last_stamp and (time.time() - t0) < timeout:
                    self._img_cond.wait(timeout=0.02)
                last_stamp = self._img_stamp_ns
                if self._img_cv is not None:
                    out.append(self._img_cv.copy())
                else:
                    break
        return out

    # ------------------------- Busy gate -------------------------

    def _try_enter_busy(self, response: Trigger.Response) -> bool:
        with self._busy_lock:
            if self._busy:
                response.success = False
                response.message = "busy"
                self._publish_report({"command": "unknown", "baseline_set": self._baseline_is_set(), "overall_ok": False, "reason": "busy"})
                return False
            self._busy = True
            return True

    def _leave_busy(self) -> None:
        with self._busy_lock:
            self._busy = False

    # ------------------------- Services -------------------------

    def _srv_reset(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        if not self._try_enter_busy(response):
            return response
        try:
            with self._base_lock:
                self._baseline = None
                self._baseline_counts = None
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
            frames = self._capture_burst(int(self.get_parameter("calib_samples").value))
            if not frames:
                report = {"command": "calibration", "baseline_set": self._baseline_is_set(), "overall_ok": False, "reason": "no_image"}
                self._publish_report(report)
                response.success = False
                response.message = "no_image"
                return response

            with self._base_lock:
                baseline_exists = self._baseline is not None

            if not baseline_exists:
                ok, report, annot = self._calibration_build(frames)
            else:
                ok, report, annot = self._calibration_update(frames)

            self._publish_report(report)
            if annot is not None:
                self._publish_annotated(annot)

            response.success = bool(ok)
            response.message = str(report.get("reason", ""))
            return response
        finally:
            self._leave_busy()

    def _srv_inference(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        if not self._try_enter_busy(response):
            return response
        try:
            if not self._baseline_is_set():
                report = {"command": "inference", "baseline_set": False, "overall_ok": False, "reason": "baseline_not_set"}
                self._publish_report(report)
                response.success = False
                response.message = "baseline_not_set"
                return response

            frames = self._capture_burst(int(self.get_parameter("infer_samples").value))
            if not frames:
                report = {"command": "inference", "baseline_set": True, "overall_ok": False, "reason": "no_image"}
                self._publish_report(report)
                response.success = False
                response.message = "no_image"
                return response

            ok, report, annot = self._inference(frames)
            self._publish_report(report)
            if annot is not None:
                self._publish_annotated(annot)

            response.success = bool(ok)
            response.message = str(report.get("reason", ""))
            return response
        finally:
            self._leave_busy()

    # ------------------------- YOLO inference (full frame) -------------------------

    def _infer(self, image_bgr: np.ndarray) -> List[Det]:
        results = self._model.predict(
            source=image_bgr,
            conf=self._conf_thr,
            iou=self._nms_iou_thr,
            device=self._device,
            verbose=False,
        )
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []

        names = self._model.names
        xywhn = r.boxes.xywhn.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        conf = r.boxes.conf.cpu().numpy()

        dets: List[Det] = []
        for (x, y, w, h), c, p in zip(xywhn, cls, conf):
            pp = float(p)
            if pp < self._conf_thr:
                continue
            cls_name = str(names.get(int(c), str(int(c))))
            dets.append(Det(cls=cls_name, x=float(x), y=float(y), w=float(w), h=float(h), conf=pp))
        return dets

    def _signature(self, dets: List[Det]) -> Dict[str, List[Det]]:
        sig: Dict[str, List[Det]] = {}
        for d in dets:
            sig.setdefault(d.cls, []).append(d)
        for k in sig:
            sig[k].sort(key=lambda z: (z.x, z.y))
        return sig

    def _counts(self, sig: Dict[str, List[Det]]) -> Dict[str, int]:
        return {k: len(v) for k, v in sig.items()}

    # ------------------------- baseline helpers -------------------------

    def _baseline_is_set(self) -> bool:
        with self._base_lock:
            return self._baseline is not None and self._baseline_counts is not None

    def _compute_tolerances(self, be: BaselineEntry) -> None:
        min_pos_tol = float(self.get_parameter("min_pos_tol").value)
        min_size_tol = float(self.get_parameter("min_size_tol").value)
        min_iou = float(self.get_parameter("min_iou").value)
        pos_margin = float(self.get_parameter("pos_margin").value)
        size_margin = float(self.get_parameter("size_margin").value)
        iou_margin = float(self.get_parameter("iou_margin").value)

        be.dx_tol = max(min_pos_tol, be.max_dx + pos_margin)
        be.dy_tol = max(min_pos_tol, be.max_dy + pos_margin)
        be.dw_rel_tol = max(min_size_tol, be.max_dw_rel + size_margin)
        be.dh_rel_tol = max(min_size_tol, be.max_dh_rel + size_margin)
        be.iou_min = max(min_iou, be.min_iou - iou_margin)

    def _estimate_translation_from_baseline(self, sig: Dict[str, List[Det]]) -> Tuple[float, float]:
        with self._base_lock:
            baseline = self._baseline
            base_counts = self._baseline_counts
        if baseline is None or base_counts is None:
            return 0.0, 0.0
        if self._counts(sig) != base_counts:
            return 0.0, 0.0

        dxs: List[float] = []
        dys: List[float] = []
        for cls, entries in baseline.items():
            cur_list = sig.get(cls, [])
            for i, be in enumerate(entries):
                if i >= len(cur_list):
                    continue
                dxs.append(cur_list[i].x - be.mean.x)
                dys.append(cur_list[i].y - be.mean.y)
        return _median(dxs), _median(dys)

    @staticmethod
    def _apply_translation(sig: Dict[str, List[Det]], tx: float, ty: float) -> Dict[str, List[Det]]:
        if abs(tx) < 1e-12 and abs(ty) < 1e-12:
            return sig
        out: Dict[str, List[Det]] = {}
        for cls, lst in sig.items():
            out[cls] = [Det(cls=d.cls, x=d.x - tx, y=d.y - ty, w=d.w, h=d.h, conf=d.conf) for d in lst]
        return out

    @staticmethod
    def _estimate_translation_between(ref_sig: Dict[str, List[Det]], cur_sig: Dict[str, List[Det]], counts: Dict[str, int]) -> Tuple[float, float]:
        dxs: List[float] = []
        dys: List[float] = []
        for cls, cnt in counts.items():
            r = ref_sig.get(cls, [])
            c = cur_sig.get(cls, [])
            if len(r) < cnt or len(c) < cnt:
                continue
            for i in range(cnt):
                dxs.append(c[i].x - r[i].x)
                dys.append(c[i].y - r[i].y)
        return _median(dxs), _median(dys)

    # ------------------------- calibration -------------------------

    def _calibration_build(self, frames: List[np.ndarray]) -> Tuple[bool, Dict[str, Any], Optional[np.ndarray]]:
        self.get_logger().info(f"calibration(build): frames={len(frames)}")

        sigs: List[Dict[str, List[Det]]] = []
        counts_list: List[Dict[str, int]] = []
        last_annot: Optional[np.ndarray] = None

        for i, img in enumerate(frames, start=1):
            sig = self._signature(self._infer(img))
            counts = self._counts(sig)
            sigs.append(sig)
            counts_list.append(counts)
            self.get_logger().info(f"calibration(build): frame {i}/{len(frames)} counts={counts}")
            last_annot = self._draw_overlay(img, sig, None, None, [], [], [], 0.0, 0.0)

        def key_of(c: Dict[str, int]) -> Tuple[Tuple[str, int], ...]:
            return tuple(sorted(c.items()))

        freq: Dict[Tuple[Tuple[str, int], ...], int] = {}
        for c in counts_list:
            k = key_of(c)
            freq[k] = freq.get(k, 0) + 1

        mode_key, mode_n = max(freq.items(), key=lambda kv: kv[1])
        base_counts = dict(mode_key)
        good_idx = [i for i, c in enumerate(counts_list) if key_of(c) == mode_key]

        if len(good_idx) < 2:
            report = {
                "command": "calibration",
                "baseline_set": False,
                "overall_ok": False,
                "reason": "not_enough_consistent_samples",
                "mode_counts": base_counts,
                "mode_have": len(good_idx),
                "need_at_least": 2,
            }
            return False, report, last_annot

        ref_sig = sigs[good_idx[0]]
        aligned_sigs: List[Dict[str, List[Det]]] = []
        for j, idx in enumerate(good_idx, start=1):
            cur_sig = sigs[idx]
            tx, ty = self._estimate_translation_between(ref_sig, cur_sig, base_counts)
            aligned_sigs.append(self._apply_translation(cur_sig, tx, ty))
            self.get_logger().info(f"calibration(build): align {j}/{len(good_idx)} tx={tx:.6f} ty={ty:.6f}")

        baseline: Dict[str, List[BaselineEntry]] = {}
        for cls, cnt in base_counts.items():
            if cnt <= 0:
                continue
            xs = np.zeros((len(aligned_sigs), cnt), dtype=np.float64)
            ys = np.zeros((len(aligned_sigs), cnt), dtype=np.float64)
            ws = np.zeros((len(aligned_sigs), cnt), dtype=np.float64)
            hs = np.zeros((len(aligned_sigs), cnt), dtype=np.float64)

            for si, sig in enumerate(aligned_sigs):
                lst = sig.get(cls, [])
                for k in range(cnt):
                    d = lst[k]
                    xs[si, k] = d.x
                    ys[si, k] = d.y
                    ws[si, k] = d.w
                    hs[si, k] = d.h

            mean_x = xs.mean(axis=0)
            mean_y = ys.mean(axis=0)
            mean_w = ws.mean(axis=0)
            mean_h = hs.mean(axis=0)

            entries: List[BaselineEntry] = []
            for k in range(cnt):
                be = BaselineEntry(mean=BoxMean(float(mean_x[k]), float(mean_y[k]), float(mean_w[k]), float(mean_h[k])))
                be.n = len(aligned_sigs)
                entries.append(be)
            baseline[cls] = entries

        for sig in aligned_sigs:
            for cls, entries in baseline.items():
                cur_list = sig.get(cls, [])
                for i, be in enumerate(entries):
                    d = cur_list[i]
                    dx = abs(d.x - be.mean.x)
                    dy = abs(d.y - be.mean.y)
                    dw = _rel_diff(be.mean.w, d.w)
                    dh = _rel_diff(be.mean.h, d.h)
                    iou = _iou_xywh((be.mean.x, be.mean.y, be.mean.w, be.mean.h), (d.x, d.y, d.w, d.h))
                    be.max_dx = max(be.max_dx, dx)
                    be.max_dy = max(be.max_dy, dy)
                    be.max_dw_rel = max(be.max_dw_rel, dw)
                    be.max_dh_rel = max(be.max_dh_rel, dh)
                    be.min_iou = min(be.min_iou, iou)

        for entries in baseline.values():
            for be in entries:
                self._compute_tolerances(be)

        with self._base_lock:
            self._baseline = baseline
            self._baseline_counts = base_counts

        report = {
            "command": "calibration",
            "baseline_set": True,
            "overall_ok": True,
            "reason": "baseline_created",
            "counts_ref": base_counts,
            "accepted_frames": len(good_idx),
            "total_frames": len(frames),
            "mode_frequency": mode_n,
        }
        return True, report, last_annot

    def _calibration_update(self, frames: List[np.ndarray]) -> Tuple[bool, Dict[str, Any], Optional[np.ndarray]]:
        self.get_logger().info(f"calibration(update): frames={len(frames)}")
        with self._base_lock:
            base_counts = dict(self._baseline_counts or {})

        accepted = 0
        rejected = 0
        last_annot: Optional[np.ndarray] = None

        for i, img in enumerate(frames, start=1):
            sig = self._signature(self._infer(img))
            cur_counts = self._counts(sig)

            if cur_counts != base_counts:
                rejected += 1
                self.get_logger().info(f"calibration(update): frame {i}/{len(frames)} rejected counts={cur_counts} ref={base_counts}")
                last_annot = self._draw_overlay(img, sig, None, base_counts, [], [], [], 0.0, 0.0)
                continue

            tx, ty = self._estimate_translation_from_baseline(sig)
            sig_al = self._apply_translation(sig, tx, ty)
            self._update_baseline_with_sample(sig_al)
            accepted += 1
            self.get_logger().info(f"calibration(update): frame {i}/{len(frames)} accepted tx={tx:.6f} ty={ty:.6f}")

            ok, details, annot = self._check_and_annotate(img, sig)
            last_annot = annot

        ok_final = accepted > 0
        report = {
            "command": "calibration",
            "baseline_set": True,
            "overall_ok": bool(ok_final),
            "reason": "baseline_updated" if ok_final else "no_accepted_samples",
            "counts_ref": base_counts,
            "accepted_frames": accepted,
            "rejected_frames": rejected,
            "total_frames": len(frames),
        }
        return ok_final, report, last_annot

    def _update_baseline_with_sample(self, sig_aligned: Dict[str, List[Det]]) -> None:
        with self._base_lock:
            if self._baseline is None or self._baseline_counts is None:
                return
            baseline = self._baseline
            base_counts = self._baseline_counts
            if self._counts(sig_aligned) != base_counts:
                return

            for cls, entries in baseline.items():
                cur_list = sig_aligned.get(cls, [])
                for i, be in enumerate(entries):
                    d = cur_list[i]
                    dx = abs(d.x - be.mean.x)
                    dy = abs(d.y - be.mean.y)
                    dw = _rel_diff(be.mean.w, d.w)
                    dh = _rel_diff(be.mean.h, d.h)
                    iou = _iou_xywh((be.mean.x, be.mean.y, be.mean.w, be.mean.h), (d.x, d.y, d.w, d.h))

                    be.max_dx = max(be.max_dx, dx)
                    be.max_dy = max(be.max_dy, dy)
                    be.max_dw_rel = max(be.max_dw_rel, dw)
                    be.max_dh_rel = max(be.max_dh_rel, dh)
                    be.min_iou = min(be.min_iou, iou)

                    n1 = be.n + 1
                    be.mean.x = be.mean.x + (d.x - be.mean.x) / n1
                    be.mean.y = be.mean.y + (d.y - be.mean.y) / n1
                    be.mean.w = be.mean.w + (d.w - be.mean.w) / n1
                    be.mean.h = be.mean.h + (d.h - be.mean.h) / n1
                    be.n = n1

                    self._compute_tolerances(be)

    # ------------------------- inference/check -------------------------

    def _inference(self, frames: List[np.ndarray]) -> Tuple[bool, Dict[str, Any], Optional[np.ndarray]]:
        self.get_logger().info(f"inference: frames={len(frames)}")

        per_frame: List[Dict[str, Any]] = []
        overall_ok = True
        first_fail_reason = "ok"

        last_annot: Optional[np.ndarray] = None
        last_details: Dict[str, Any] = {}

        for i, img in enumerate(frames, start=1):
            sig = self._signature(self._infer(img))
            ok, details, annot = self._check_and_annotate(img, sig)
            per_frame.append({"i": i, "ok": bool(ok), "reason": details.get("reason", "unknown")})
            self.get_logger().info(f"inference: frame {i}/{len(frames)} ok={ok} reason={details.get('reason')}")
            last_annot = annot
            last_details = details

            if not ok and overall_ok:
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
        return overall_ok, report, last_annot

    def _check_and_annotate(self, image_bgr: np.ndarray, sig: Dict[str, List[Det]]) -> Tuple[bool, Dict[str, Any], np.ndarray]:
        with self._base_lock:
            baseline = self._baseline
            base_counts = self._baseline_counts

        if baseline is None or base_counts is None:
            annot = self._draw_overlay(image_bgr, sig, None, None, [], [], [], 0.0, 0.0)
            return False, {"reason": "baseline_not_set"}, annot

        cur_counts = self._counts(sig)

        tx, ty = (0.0, 0.0)
        if cur_counts == base_counts:
            tx, ty = self._estimate_translation_from_baseline(sig)
        sig_al = self._apply_translation(sig, tx, ty)

        missing: List[Dict[str, Any]] = []
        extra: List[Dict[str, Any]] = []

        all_classes = sorted(set(list(base_counts.keys()) + list(cur_counts.keys())))
        for cls in all_classes:
            ref_n = int(base_counts.get(cls, 0))
            cur_n = int(cur_counts.get(cls, 0))
            if cur_n < ref_n:
                for k in range(cur_n, ref_n):
                    missing.append({"class": cls, "index": k})
            elif cur_n > ref_n:
                cur_list = sig.get(cls, [])
                for k in range(ref_n, cur_n):
                    d = cur_list[k]
                    extra.append({"class": cls, "index": k, "conf": d.conf, "x": d.x, "y": d.y, "w": d.w, "h": d.h})

        mismatches: List[Dict[str, Any]] = []
        ok = True
        reason = "ok"

        if cur_counts != base_counts:
            ok = False
            reason = "counts_mismatch"

        for cls, entries in baseline.items():
            cur_list = sig_al.get(cls, [])
            m = min(len(entries), len(cur_list))
            for i in range(m):
                be = entries[i]
                d = cur_list[i]
                dx = abs(d.x - be.mean.x)
                dy = abs(d.y - be.mean.y)
                dw_rel = _rel_diff(be.mean.w, d.w)
                dh_rel = _rel_diff(be.mean.h, d.h)
                iou = _iou_xywh((be.mean.x, be.mean.y, be.mean.w, be.mean.h), (d.x, d.y, d.w, d.h))

                bad = (dx > be.dx_tol) or (dy > be.dy_tol) or (dw_rel > be.dw_rel_tol) or (dh_rel > be.dh_rel_tol) or (iou < be.iou_min)
                if bad:
                    ok = False
                    if reason == "ok":
                        reason = "geometry_mismatch"
                    mismatches.append({
                        "class": cls,
                        "index": i,
                        "dx": dx, "dy": dy,
                        "dw_rel": dw_rel, "dh_rel": dh_rel,
                        "iou": iou,
                        "tol": {
                            "dx": be.dx_tol, "dy": be.dy_tol,
                            "dw_rel": be.dw_rel_tol, "dh_rel": be.dh_rel_tol,
                            "iou_min": be.iou_min,
                        },
                    })

        details: Dict[str, Any] = {
            "baseline_set": True,
            "reason": reason,
            "counts_ref": base_counts,
            "counts_cur": cur_counts,
            "tx_ty": {"tx": tx, "ty": ty},
        }
        if missing:
            details["missing"] = missing
        if extra:
            details["extra"] = extra
        if mismatches:
            details["mismatches"] = mismatches[:50]

        annot = self._draw_overlay(image_bgr, sig, baseline, base_counts, mismatches, missing, extra, tx, ty)
        return ok, details, annot

    # ------------------------- visualization -------------------------

    def _draw_overlay(
        self,
        img: np.ndarray,
        sig: Dict[str, List[Det]],
        baseline: Optional[Dict[str, List[BaselineEntry]]],
        base_counts: Optional[Dict[str, int]],
        mismatches: List[Dict[str, Any]],
        missing: List[Dict[str, Any]],
        extra: List[Dict[str, Any]],
        tx: float,
        ty: float,
    ) -> np.ndarray:
        out = img.copy()
        H, W = out.shape[:2]

        # expected baseline boxes (yellow)
        if baseline is not None and bool(self.get_parameter("draw_expected_boxes").value):
            for cls, entries in baseline.items():
                for i, be in enumerate(entries):
                    x1, y1, x2, y2 = _xywh_to_xyxy(be.mean.x, be.mean.y, be.mean.w, be.mean.h)
                    x1p = int(round(x1 * W))
                    y1p = int(round(y1 * H))
                    x2p = int(round(x2 * W))
                    y2p = int(round(y2 * H))
                    cv2.rectangle(out, (x1p, y1p), (x2p, y2p), (0, 255, 255), 2)
                    cv2.putText(out, f"EXP {cls}[{i}]", (x1p, y2p + 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        bad_set = {(m["class"], int(m["index"])) for m in mismatches}

        # current detections (green/red)
        for cls, lst in sig.items():
            for i, d in enumerate(lst):
                x1, y1, x2, y2 = _xywh_to_xyxy(d.x, d.y, d.w, d.h)
                x1p = int(round(x1 * W))
                y1p = int(round(y1 * H))
                x2p = int(round(x2 * W))
                y2p = int(round(y2 * H))

                if (cls, i) in bad_set:
                    color = (0, 0, 255)
                    tag = "BAD"
                else:
                    color = (0, 255, 0)
                    tag = "OK"

                cv2.rectangle(out, (x1p, y1p), (x2p, y2p), color, 2)
                cv2.putText(out, f"{tag} {cls}[{i}] {d.conf:.2f}", (x1p, max(0, y1p - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # missing (expected box in red)
        if baseline is not None:
            for m in missing:
                cls = str(m["class"])
                idx = int(m["index"])
                be_list = baseline.get(cls, [])
                if 0 <= idx < len(be_list):
                    be = be_list[idx]
                    x1, y1, x2, y2 = _xywh_to_xyxy(be.mean.x, be.mean.y, be.mean.w, be.mean.h)
                    x1p = int(round(x1 * W))
                    y1p = int(round(y1 * H))
                    x2p = int(round(x2 * W))
                    y2p = int(round(y2 * H))
                    cv2.rectangle(out, (x1p, y1p), (x2p, y2p), (0, 0, 255), 3)
                    cv2.putText(out, f"MISSING {cls}[{idx}]", (x1p, max(0, y1p - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        # extra (orange)
        for e in extra:
            x1, y1, x2, y2 = _xywh_to_xyxy(float(e["x"]), float(e["y"]), float(e["w"]), float(e["h"]))
            x1p = int(round(x1 * W))
            y1p = int(round(y1 * H))
            x2p = int(round(x2 * W))
            y2p = int(round(y2 * H))
            cv2.rectangle(out, (x1p, y1p), (x2p, y2p), (0, 165, 255), 3)
            cv2.putText(out, f"EXTRA {e['class']} {float(e.get('conf', 0.0)):.2f}", (x1p, y2p + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2, cv2.LINE_AA)

        cv2.putText(out, f"align tx={tx:.6f} ty={ty:.6f}", (10, H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        return out

    # ------------------------- publishing -------------------------

    def _publish_report(self, report: Dict[str, Any]) -> None:
        msg = String()
        msg.data = json.dumps(report, ensure_ascii=False)
        self._pub_report.publish(msg)

    def _publish_annotated(self, annot_bgr: np.ndarray) -> None:
        try:
            msg = self._bridge.cv2_to_imgmsg(annot_bgr, encoding="bgr8")
        except Exception as e:
            self.get_logger().warning(f"cv_bridge publish convert failed: {e}")
            return
        self._pub_annot.publish(msg)


def main() -> None:
    rclpy.init()
    node = PCBInspectorNode()
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
