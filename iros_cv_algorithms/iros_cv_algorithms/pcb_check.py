#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS2 PCB Inspector Node (standalone) with robust red-anchor ROI

What changed vs previous version:
- ROI detection made more robust:
  1) Red mask = HSV red OR "redness score" (R - max(G,B)) threshold.
  2) Candidate filtering by area + circularity + aspect ratio.
  3) Corner selection via minAreaRect on candidate centers, then snapping each corner
     to the nearest unused candidate (prevents picking the red LED inside the rectangle).
  4) Stronger validation: min pairwise separation, max corner snap distance, ROI size sanity.

If ROI isn't found, check the published ~/annotated image (draw_roi_debug=True) and tune:
- anchor_s_min, anchor_v_min
- anchor_redness_thr, anchor_min_area_px
- anchor_near_corner_max_px, anchor_min_sep_px
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

import math
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
    x: float  # center x in ROI crop norm [0..1]
    y: float  # center y in ROI crop norm [0..1]
    w: float  # width  in ROI crop norm [0..1]
    h: float  # height in ROI crop norm [0..1]
    conf: float


@dataclass(frozen=True)
class ROI:
    x0: int
    y0: int
    w: int
    h: int
    quad: np.ndarray  # shape (4,2) float32 in full-image pixels, ordered TL,TR,BR,BL


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


def _rel_diff(a: float, b: float, eps: float = 1e-9) -> float:
    return abs(a - b) / max(abs(a), eps)


def _xywh_to_xyxy(x: float, y: float, w: float, h: float) -> Tuple[float, float, float, float]:
    x1 = x - w / 2.0
    y1 = y - h / 2.0
    x2 = x + w / 2.0
    y2 = y + h / 2.0
    return x1, y1, x2, y2


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


def _order_quad_points(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 points into TL, TR, BR, BL for image coords (x right, y down).
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts[:, 0] + pts[:, 1]
    d = pts[:, 0] - pts[:, 1]
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmax(d)]
    bl = pts[np.argmin(d)]
    return np.stack([tl, tr, br, bl], axis=0).astype(np.float32)


# ----------------------------- node -----------------------------


class PCBInspectorNode(Node):
    def __init__(self) -> None:
        super().__init__("pcb_inspector")

        self._cbg = ReentrantCallbackGroup()
        self._bridge = CvBridge()

        # Parameters
        self.declare_parameter("image_topic", "/image_raw")

        # YOLO
        self.declare_parameter("model_path", "/home/mobile/ros2_ws/src/iros_cv_algorithms/iros_cv_algorithms/algos/models/yolo11s_best.pt")
        self.declare_parameter("conf_thr", 0.25)
        self.declare_parameter("iou_thr", 0.50)
        self.declare_parameter("device", "0")

        # Burst sizes
        self.declare_parameter("calib_samples", 5)
        self.declare_parameter("infer_samples", 3)

        # Floors (minimum tolerances) in ROI-normalized coordinates
        self.declare_parameter("min_pos_tol", 0.01)   # dx/dy
        self.declare_parameter("min_size_tol", 0.25)  # relative w/h error
        self.declare_parameter("min_iou", 0.75)

        # Margins
        self.declare_parameter("pos_margin", 0.005)
        self.declare_parameter("size_margin", 0.05)
        self.declare_parameter("iou_margin", 0.05)

        # Frame wait
        self.declare_parameter("frame_wait_timeout_sec", 2.0)

        # Anchor/ROI detection params (tune these first)
        self.declare_parameter("anchor_min_area_px", 150)      # if too high -> not_enough_red_blobs
        self.declare_parameter("anchor_max_area_px", 200000)
        self.declare_parameter("anchor_circularity_min", 0.25) # lower if anchors are not circular
        self.declare_parameter("anchor_aspect_max", 2.0)
        self.declare_parameter("anchor_topk", 20)

        # HSV thresholds
        self.declare_parameter("anchor_h_lo1", 0)
        self.declare_parameter("anchor_h_hi1", 25)
        self.declare_parameter("anchor_h_lo2", 160)
        self.declare_parameter("anchor_h_hi2", 179)
        self.declare_parameter("anchor_s_min", 60)
        self.declare_parameter("anchor_v_min", 40)

        # Redness score thresholds (BGR)
        self.declare_parameter("anchor_redness_thr", 50)  # threshold on (R - max(G,B))
        self.declare_parameter("anchor_r_min", 60)        # minimal R channel

        # Geometry validation
        self.declare_parameter("anchor_min_sep_px", 25)         # min distance between chosen corners
        self.declare_parameter("anchor_near_corner_max_px", 80) # max distance to snap rect corner to candidate
        self.declare_parameter("roi_margin_px", 10)
        self.declare_parameter("roi_min_area_frac", 0.05)

        self.declare_parameter("draw_roi_debug", True)

        # Topics
        img_topic = str(self.get_parameter("image_raw").value)
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
        self._busy: bool = False

        # Baseline state (ROI-normalized)
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

    # ------------------------- ROS callbacks -------------------------

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
            n = int(self.get_parameter("calib_samples").value)
            frames = self._capture_burst(n)
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

            n = int(self.get_parameter("infer_samples").value)
            frames = self._capture_burst(n)
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

    # ------------------------- busy gate -------------------------

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

    # ------------------------- frame capture -------------------------

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

    # ------------------------- anchor ROI detection (robust) -------------------------

    def _detect_roi(self, image_bgr: np.ndarray) -> Tuple[Optional[ROI], Dict[str, Any]]:
        H, W = image_bgr.shape[:2]
        dbg: Dict[str, Any] = {}

        # ---- Build red mask: HSV OR redness score ----
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

        h_lo1 = int(self.get_parameter("anchor_h_lo1").value)
        h_hi1 = int(self.get_parameter("anchor_h_hi1").value)
        h_lo2 = int(self.get_parameter("anchor_h_lo2").value)
        h_hi2 = int(self.get_parameter("anchor_h_hi2").value)
        s_min = int(self.get_parameter("anchor_s_min").value)
        v_min = int(self.get_parameter("anchor_v_min").value)

        lower1 = np.array([h_lo1, s_min, v_min], dtype=np.uint8)
        upper1 = np.array([h_hi1, 255, 255], dtype=np.uint8)
        lower2 = np.array([h_lo2, s_min, v_min], dtype=np.uint8)
        upper2 = np.array([h_hi2, 255, 255], dtype=np.uint8)

        mask_hsv = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2))

        b, g, r = cv2.split(image_bgr)
        redness_thr = int(self.get_parameter("anchor_redness_thr").value)
        r_min = int(self.get_parameter("anchor_r_min").value)
        redness = cv2.subtract(r, cv2.max(b, g))  # uint8, saturates at 0
        mask_redness = cv2.inRange(redness, redness_thr, 255)
        mask_rmin = cv2.inRange(r, r_min, 255)
        mask_bgr = cv2.bitwise_and(mask_redness, mask_rmin)

        mask = cv2.bitwise_or(mask_hsv, mask_bgr)

        # ---- Morphology ----
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

        # ---- Contours -> candidates ----
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_a = int(self.get_parameter("anchor_min_area_px").value)
        max_a = int(self.get_parameter("anchor_max_area_px").value)
        circ_min = float(self.get_parameter("anchor_circularity_min").value)
        asp_max = float(self.get_parameter("anchor_aspect_max").value)
        topk = int(self.get_parameter("anchor_topk").value)

        candidates: List[Tuple[float, float, float]] = []  # (cx, cy, area)
        for c in contours:
            area = float(cv2.contourArea(c))
            if area < min_a or area > max_a:
                continue

            per = float(cv2.arcLength(c, True))
            if per <= 1e-6:
                continue
            circularity = float(4.0 * np.pi * area / (per * per))

            x, y, ww, hh = cv2.boundingRect(c)
            aspect = float(max(ww, hh) / max(min(ww, hh), 1))

            if circularity < circ_min:
                continue
            if aspect > asp_max:
                continue

            M = cv2.moments(c)
            if abs(M.get("m00", 0.0)) < 1e-6:
                continue
            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])
            candidates.append((cx, cy, area))

        candidates.sort(key=lambda t: t[2], reverse=True)
        candidates = candidates[:max(4, topk)]

        dbg["anchors_candidates"] = len(candidates)
        dbg["cand_pts"] = [[float(c[0]), float(c[1]), float(c[2])] for c in candidates[:30]]

        if len(candidates) < 4:
            dbg["reason"] = "not_enough_red_blobs"
            return None, dbg

        pts = np.array([[c[0], c[1]] for c in candidates], dtype=np.float32)

        # ---- Corner selection via minAreaRect + snap to nearest candidates ----
        rect = cv2.minAreaRect(pts.reshape(-1, 1, 2))
        box = cv2.boxPoints(rect).astype(np.float32)  # 4 corners of fitted rectangle
        box = _order_quad_points(box)

        snap_max = float(self.get_parameter("anchor_near_corner_max_px").value)
        chosen: List[np.ndarray] = []
        used = np.zeros((len(pts),), dtype=bool)

        for corner in box:
            d2 = np.sum((pts - corner[None, :]) ** 2, axis=1)
            order = np.argsort(d2)
            sel = None
            for idx in order:
                if used[idx]:
                    continue
                dist = float(math.sqrt(float(d2[idx])))
                if dist <= snap_max:
                    sel = idx
                    break
            if sel is None:
                dbg["reason"] = "no_candidate_near_rect_corner"
                dbg["rect_box"] = box.tolist()
                return None, dbg
            used[sel] = True
            chosen.append(pts[sel])

        quad = _order_quad_points(np.stack(chosen, axis=0))

        # ---- Validate distinct corners ----
        min_sep = float(self.get_parameter("anchor_min_sep_px").value)
        for i in range(4):
            for j in range(i + 1, 4):
                if float(np.linalg.norm(quad[i] - quad[j])) < min_sep:
                    dbg["reason"] = "corner_points_not_distinct"
                    dbg["roi_quad"] = quad.tolist()
                    return None, dbg

        # ---- BBox ROI around quad ----
        x_min = int(np.floor(np.min(quad[:, 0])))
        y_min = int(np.floor(np.min(quad[:, 1])))
        x_max = int(np.ceil(np.max(quad[:, 0])))
        y_max = int(np.ceil(np.max(quad[:, 1])))

        margin = int(self.get_parameter("roi_margin_px").value)
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(W - 1, x_max + margin)
        y_max = min(H - 1, y_max + margin)

        roi_w = max(1, x_max - x_min + 1)
        roi_h = max(1, y_max - y_min + 1)

        min_frac = float(self.get_parameter("roi_min_area_frac").value)
        if (roi_w * roi_h) < (min_frac * W * H):
            dbg["reason"] = "roi_too_small_probably_false_anchor"
            dbg["roi_bbox"] = [x_min, y_min, roi_w, roi_h]
            dbg["roi_quad"] = quad.tolist()
            return None, dbg

        roi = ROI(x0=x_min, y0=y_min, w=roi_w, h=roi_h, quad=quad)
        dbg["roi_bbox"] = [roi.x0, roi.y0, roi.w, roi.h]
        dbg["roi_quad"] = quad.tolist()
        dbg["reason"] = "ok"
        return roi, dbg

    def _crop_and_mask_roi(self, image_bgr: np.ndarray, roi: ROI) -> np.ndarray:
        crop = image_bgr[roi.y0:roi.y0 + roi.h, roi.x0:roi.x0 + roi.w].copy()

        poly = roi.quad.copy()
        poly[:, 0] -= float(roi.x0)
        poly[:, 1] -= float(roi.y0)
        poly_i = np.round(poly).astype(np.int32).reshape((-1, 1, 2))

        mask = np.zeros((roi.h, roi.w), dtype=np.uint8)
        cv2.fillPoly(mask, [poly_i], 255)

        return cv2.bitwise_and(crop, crop, mask=mask)

    # ------------------------- YOLO inference within ROI -------------------------

    def _infer_in_roi(self, image_bgr: np.ndarray, roi: ROI) -> List[Det]:
        crop = self._crop_and_mask_roi(image_bgr, roi)

        results = self._model.predict(
            source=crop,
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

    # ------------------------- translation alignment in ROI coords -------------------------

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

    @staticmethod
    def _apply_translation(sig: Dict[str, List[Det]], tx: float, ty: float) -> Dict[str, List[Det]]:
        if abs(tx) < 1e-12 and abs(ty) < 1e-12:
            return sig
        out: Dict[str, List[Det]] = {}
        for cls, lst in sig.items():
            out[cls] = [Det(cls=d.cls, x=d.x - tx, y=d.y - ty, w=d.w, h=d.h, conf=d.conf) for d in lst]
        return out

    def _estimate_translation_from_baseline(self, sig: Dict[str, List[Det]]) -> Tuple[float, float]:
        with self._base_lock:
            baseline = self._baseline
            base_counts = self._baseline_counts
        if baseline is None or base_counts is None:
            return 0.0, 0.0
        cur_counts = self._counts(sig)
        if cur_counts != base_counts:
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

    # ------------------------- calibration (build/update) -------------------------

    def _calibration_build(self, frames: List[np.ndarray]) -> Tuple[bool, Dict[str, Any], Optional[np.ndarray]]:
        self.get_logger().info(f"calibration(build): frames={len(frames)}")

        sigs: List[Dict[str, List[Det]]] = []
        counts_list: List[Dict[str, int]] = []
        rois_ok: List[bool] = []
        dbg_last: Dict[str, Any] = {}
        last_annot: Optional[np.ndarray] = None

        for i, img in enumerate(frames, start=1):
            roi, dbg = self._detect_roi(img)
            dbg_last = dbg
            if roi is None:
                self.get_logger().info(f"calibration(build): frame {i}/{len(frames)} ROI not found ({dbg.get('reason')})")
                sigs.append({})
                counts_list.append({})
                rois_ok.append(False)
                last_annot = self._draw_roi_debug(img, roi=None, dbg=dbg)
                continue

            dets = self._infer_in_roi(img, roi)
            sig = self._signature(dets)
            counts = self._counts(sig)
            sigs.append(sig)
            counts_list.append(counts)
            rois_ok.append(True)

            self.get_logger().info(f"calibration(build): frame {i}/{len(frames)} counts={counts}")
            last_annot = self._draw_overlay(img, roi, sig, baseline=None, base_counts=None,
                                           mismatches=[], missing=[], extra=[], tx=0.0, ty=0.0, roi_dbg=dbg)

        def key_of(c: Dict[str, int]) -> Tuple[Tuple[str, int], ...]:
            return tuple(sorted(c.items()))

        idx_valid = [i for i, ok in enumerate(rois_ok) if ok]
        if not idx_valid:
            report = {"command": "calibration", "baseline_set": False, "overall_ok": False, "reason": "roi_not_found_all_frames", "roi_debug_last": dbg_last}
            return False, report, last_annot

        freq: Dict[Tuple[Tuple[str, int], ...], int] = {}
        for i in idx_valid:
            k = key_of(counts_list[i])
            freq[k] = freq.get(k, 0) + 1

        mode_key = max(freq.items(), key=lambda kv: kv[1])[0]
        base_counts = dict(mode_key)
        good_idx = [i for i in idx_valid if key_of(counts_list[i]) == mode_key]

        if len(good_idx) < 2:
            report = {
                "command": "calibration",
                "baseline_set": False,
                "overall_ok": False,
                "reason": "not_enough_consistent_samples",
                "mode_counts": base_counts,
                "mode_have": len(good_idx),
                "need_at_least": 2,
                "roi_debug_last": dbg_last,
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
        }
        return True, report, last_annot

    def _calibration_update(self, frames: List[np.ndarray]) -> Tuple[bool, Dict[str, Any], Optional[np.ndarray]]:
        self.get_logger().info(f"calibration(update): frames={len(frames)}")

        with self._base_lock:
            base_counts = dict(self._baseline_counts or {})

        accepted = 0
        rejected = 0
        roi_fail = 0

        last_annot: Optional[np.ndarray] = None
        last_details: Dict[str, Any] = {}

        for i, img in enumerate(frames, start=1):
            roi, dbg = self._detect_roi(img)
            if roi is None:
                roi_fail += 1
                self.get_logger().info(f"calibration(update): frame {i}/{len(frames)} ROI not found ({dbg.get('reason')})")
                last_annot = self._draw_roi_debug(img, roi=None, dbg=dbg)
                continue

            sig = self._signature(self._infer_in_roi(img, roi))
            cur_counts = self._counts(sig)

            if cur_counts != base_counts:
                rejected += 1
                self.get_logger().info(f"calibration(update): frame {i}/{len(frames)} rejected counts={cur_counts} ref={base_counts}")
                last_annot = self._draw_overlay(img, roi, sig, baseline=None, base_counts=base_counts,
                                               mismatches=[], missing=[], extra=[], tx=0.0, ty=0.0, roi_dbg=dbg)
                continue

            tx, ty = self._estimate_translation_from_baseline(sig)
            sig_al = self._apply_translation(sig, tx, ty)
            self._update_baseline_with_sample(sig_al)
            accepted += 1
            self.get_logger().info(f"calibration(update): frame {i}/{len(frames)} accepted tx={tx:.6f} ty={ty:.6f}")

            ok, details, annot = self._check_and_annotate(img, roi, sig, roi_dbg=dbg)
            last_annot = annot
            last_details = details

        ok_final = accepted > 0
        report = {
            "command": "calibration",
            "baseline_set": True,
            "overall_ok": bool(ok_final),
            "reason": "baseline_updated" if ok_final else "no_accepted_samples",
            "counts_ref": base_counts,
            "accepted_frames": accepted,
            "rejected_frames": rejected,
            "roi_fail_frames": roi_fail,
            "total_frames": len(frames),
        }
        report.update(last_details)
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
            roi, dbg = self._detect_roi(img)
            if roi is None:
                ok = False
                details = {"reason": "roi_not_found", "roi_debug": dbg}
                annot = self._draw_roi_debug(img, roi=None, dbg=dbg)
                per_frame.append({"i": i, "ok": False, "reason": "roi_not_found"})
                self.get_logger().info(f"inference: frame {i}/{len(frames)} ROI not found ({dbg.get('reason')})")
            else:
                sig = self._signature(self._infer_in_roi(img, roi))
                ok, details, annot = self._check_and_annotate(img, roi, sig, roi_dbg=dbg)
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

    def _check_and_annotate(
        self, image_bgr: np.ndarray, roi: ROI, sig: Dict[str, List[Det]], roi_dbg: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any], Optional[np.ndarray]]:
        with self._base_lock:
            baseline = self._baseline
            base_counts = self._baseline_counts

        if baseline is None or base_counts is None:
            annot = self._draw_overlay(image_bgr, roi, sig, baseline=None, base_counts=None,
                                      mismatches=[], missing=[], extra=[], tx=0.0, ty=0.0, roi_dbg=roi_dbg)
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
            "roi_bbox": [roi.x0, roi.y0, roi.w, roi.h],
            "roi_quad": roi.quad.tolist(),
            "roi_debug": roi_dbg,
        }
        if missing:
            details["missing"] = missing
        if extra:
            details["extra"] = extra
        if mismatches:
            details["mismatches"] = mismatches[:50]

        annot = self._draw_overlay(image_bgr, roi, sig, baseline=baseline, base_counts=base_counts,
                                   mismatches=mismatches, missing=missing, extra=extra, tx=tx, ty=ty, roi_dbg=roi_dbg)
        return ok, details, annot

    # ------------------------- visualization -------------------------

    def _draw_roi_debug(self, img: np.ndarray, roi: Optional[ROI], dbg: Dict[str, Any]) -> np.ndarray:
        out = img.copy()
        if not bool(self.get_parameter("draw_roi_debug").value):
            return out

        txt = f"ROI: {'OK' if roi is not None else 'FAIL'} reason={dbg.get('reason','')} cand={dbg.get('anchors_candidates',0)}"
        cv2.putText(out, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        # draw candidate centers (magenta)
        cand = dbg.get("cand_pts", [])
        for c in cand[:30]:
            x, y = int(round(c[0])), int(round(c[1]))
            cv2.circle(out, (x, y), 5, (255, 0, 255), -1)

        bbox = dbg.get("roi_bbox", None)
        if bbox and isinstance(bbox, list) and len(bbox) == 4:
            x0, y0, w, h = bbox
            cv2.rectangle(out, (int(x0), int(y0)), (int(x0 + w), int(y0 + h)), (255, 255, 0), 2)

        quad = dbg.get("roi_quad", None)
        if quad and isinstance(quad, list) and len(quad) == 4:
            q = np.array(quad, dtype=np.int32).reshape(4, 2)
            cv2.polylines(out, [q.reshape(-1, 1, 2)], True, (0, 255, 255), 3)
            for i, (x, y) in enumerate(q):
                cv2.circle(out, (int(x), int(y)), 7, (0, 255, 255), -1)
                cv2.putText(out, f"{i}", (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        return out

    @staticmethod
    def _px_from_roi(det: Det, roi: ROI) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = _xywh_to_xyxy(det.x, det.y, det.w, det.h)
        x1p = int(round(roi.x0 + x1 * roi.w))
        y1p = int(round(roi.y0 + y1 * roi.h))
        x2p = int(round(roi.x0 + x2 * roi.w))
        y2p = int(round(roi.y0 + y2 * roi.h))
        return x1p, y1p, x2p, y2p

    @staticmethod
    def _px_from_roi_mean(m: BoxMean, roi: ROI) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = _xywh_to_xyxy(m.x, m.y, m.w, m.h)
        x1p = int(round(roi.x0 + x1 * roi.w))
        y1p = int(round(roi.y0 + y1 * roi.h))
        x2p = int(round(roi.x0 + x2 * roi.w))
        y2p = int(round(roi.y0 + y2 * roi.h))
        return x1p, y1p, x2p, y2p

    def _draw_overlay(
        self,
        img: np.ndarray,
        roi: ROI,
        sig: Dict[str, List[Det]],
        baseline: Optional[Dict[str, List[BaselineEntry]]],
        base_counts: Optional[Dict[str, int]],
        mismatches: List[Dict[str, Any]],
        missing: List[Dict[str, Any]],
        extra: List[Dict[str, Any]],
        tx: float,
        ty: float,
        roi_dbg: Dict[str, Any],
    ) -> np.ndarray:
        out = img.copy()

        # ROI polygon
        q = roi.quad.astype(np.int32).reshape(4, 2)
        cv2.polylines(out, [q.reshape(-1, 1, 2)], True, (0, 255, 255), 3)

        # expected baseline boxes
        if baseline is not None:
            for cls, entries in baseline.items():
                for i, be in enumerate(entries):
                    x1, y1, x2, y2 = self._px_from_roi_mean(be.mean, roi)
                    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(out, f"EXP {cls}[{i}]", (x1, y2 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        bad_set = {(m["class"], int(m["index"])) for m in mismatches}

        # current detections
        for cls, lst in sig.items():
            for i, d in enumerate(lst):
                x1, y1, x2, y2 = self._px_from_roi(d, roi)
                if (cls, i) in bad_set:
                    color = (0, 0, 255)
                    tag = "BAD"
                else:
                    color = (0, 255, 0)
                    tag = "OK"
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                cv2.putText(out, f"{tag} {cls}[{i}] {d.conf:.2f}", (x1, max(0, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # missing
        if baseline is not None:
            for m in missing:
                cls = str(m["class"])
                idx = int(m["index"])
                be_list = baseline.get(cls, [])
                if 0 <= idx < len(be_list):
                    x1, y1, x2, y2 = self._px_from_roi_mean(be_list[idx].mean, roi)
                    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(out, f"MISSING {cls}[{idx}]", (x1, max(0, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        # extra
        for e in extra:
            d = Det(cls=str(e["class"]), x=float(e["x"]), y=float(e["y"]), w=float(e["w"]), h=float(e["h"]),
                    conf=float(e.get("conf", 0.0)))
            x1, y1, x2, y2 = self._px_from_roi(d, roi)
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 165, 255), 3)
            cv2.putText(out, f"EXTRA {d.cls} {d.conf:.2f}", (x1, y2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2, cv2.LINE_AA)

        cv2.putText(out, f"align tx={tx:.6f} ty={ty:.6f}", (10, out.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # also draw candidate points (magenta) for quick tuning
        if bool(self.get_parameter("draw_roi_debug").value):
            cand = roi_dbg.get("cand_pts", [])
            for c in cand[:30]:
                x, y = int(round(c[0])), int(round(c[1]))
                cv2.circle(out, (x, y), 4, (255, 0, 255), -1)

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
