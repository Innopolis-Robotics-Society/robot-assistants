#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import time
import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_srvs.srv import Trigger

try:
    from cv_bridge import CvBridge
except Exception as e:
    CvBridge = None


@dataclass
class DetectResult:
    ok: bool
    reason: str
    circles: Optional[np.ndarray] = None   # shape (N, 3) float32/float64: x, y, r
    vector: Optional[np.ndarray] = None    # shape (expected_gears,) float64 radii (sorted)
    debug_img: Optional[np.ndarray] = None


def sort_circles_top_left_to_bottom_right(circles: np.ndarray, y_tol: int = 25) -> np.ndarray:
    """
    circles: (N, 3) -> sort by rows (y with tolerance), then by x
    """
    if circles is None or len(circles) == 0:
        return circles

    circles = np.asarray(circles, dtype=float)

    # First sort by y
    idx = np.argsort(circles[:, 1])
    circles = circles[idx]

    # Group by y bands
    groups: List[np.ndarray] = []
    current = [circles[0]]
    for c in circles[1:]:
        if abs(c[1] - current[-1][1]) <= y_tol:
            current.append(c)
        else:
            groups.append(np.array(current))
            current = [c]
    groups.append(np.array(current))

    # Sort each group by x, then flatten
    sorted_list = []
    for g in groups:
        g_idx = np.argsort(g[:, 0])
        sorted_list.append(g[g_idx])

    return np.vstack(sorted_list)


class GearsCheckNode(Node):
    def __init__(self):
        super().__init__("gears_check")

        # -------------------------
        # Parameters
        # -------------------------
        self.declare_parameter("image_topic", "/image_raw")
        self.declare_parameter("expected_gears", 4)
        self.declare_parameter("calib_frames", 5)
        self.declare_parameter("check_frames", 3)

        # Detection parameters
        self.declare_parameter("blur_ksize", 7)          # odd
        self.declare_parameter("hough_dp", 1.2)
        self.declare_parameter("hough_min_dist", 40.0)
        self.declare_parameter("hough_param1", 120.0)    # Canny high
        self.declare_parameter("hough_param2", 35.0)     # accumulator threshold
        self.declare_parameter("min_radius", 10)
        self.declare_parameter("max_radius", 0)          # 0 = no limit
        self.declare_parameter("sort_y_tol", 25)

        # Validation + comparison tolerances
        self.declare_parameter("radius_round", True)
        self.declare_parameter("max_radius_std", 9999.0)  # optional sanity check
        self.declare_parameter("diff_threshold", 2.0)      # for check vs calibration (per element)

        # IO topics
        self.declare_parameter("out_image_topic", "~/image")
        self.declare_parameter("out_data_topic", "~/data")

        self.image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        self.expected_gears = int(self.get_parameter("expected_gears").value)

        out_image_topic = self.get_parameter("out_image_topic").get_parameter_value().string_value
        out_data_topic = self.get_parameter("out_data_topic").get_parameter_value().string_value

        # -------------------------
        # ROS interfaces
        # -------------------------
        self.bridge = CvBridge() if CvBridge is not None else None
        if self.bridge is None:
            raise RuntimeError("cv_bridge is required but not available in this environment.")

        self.sub = self.create_subscription(Image, self.image_topic, self._on_image, 10)
        self.pub_img = self.create_publisher(Image, out_image_topic, 10)
        self.pub_data = self.create_publisher(String, out_data_topic, 10)

        self.srv_calib = self.create_service(Trigger, "~/calibration", self._on_calibration)
        self.srv_check = self.create_service(Trigger, "~/check", self._on_check)

        # -------------------------
        # Frame synchronization
        # -------------------------
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._last_frame_id = 0
        self._last_cv_img: Optional[np.ndarray] = None

        # Calibration state
        self._calib_vector: Optional[np.ndarray] = None

        self.get_logger().info(
            f"gears_check started: image_topic={self.image_topic}, out_image={out_image_topic}, out_data={out_data_topic}"
        )

    # -------------------------
    # Image callback
    # -------------------------
    def _on_image(self, msg: Image):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"imgmsg_to_cv2 failed: {e}")
            return

        with self._cond:
            self._last_cv_img = cv_img
            self._last_frame_id += 1
            self._cond.notify_all()

    def _wait_new_frame(self, prev_id: int, timeout_s: float = 2.0) -> Tuple[bool, int, Optional[np.ndarray]]:
        deadline = time.time() + timeout_s
        with self._cond:
            while self._last_frame_id <= prev_id:
                remaining = deadline - time.time()
                if remaining <= 0:
                    return False, self._last_frame_id, None
                self._cond.wait(timeout=remaining)
            return True, self._last_frame_id, self._last_cv_img.copy() if self._last_cv_img is not None else None

    # -------------------------
    # Detection
    # -------------------------
    def _detect_gears(self, bgr: np.ndarray) -> DetectResult:
        expected = int(self.get_parameter("expected_gears").value)

        blur_ksize = int(self.get_parameter("blur_ksize").value)
        blur_ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1

        dp = float(self.get_parameter("hough_dp").value)
        min_dist = float(self.get_parameter("hough_min_dist").value)
        p1 = float(self.get_parameter("hough_param1").value)
        p2 = float(self.get_parameter("hough_param2").value)
        min_r = int(self.get_parameter("min_radius").value)
        max_r = int(self.get_parameter("max_radius").value)
        y_tol = int(self.get_parameter("sort_y_tol").value)

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if blur_ksize > 1:
            gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=dp,
            minDist=min_dist,
            param1=p1,
            param2=p2,
            minRadius=min_r,
            maxRadius=max_r if max_r > 0 else 0,
        )

        if circles is None or circles.shape[1] == 0:
            return DetectResult(ok=False, reason="no_circles")

        circles = circles[0]  # (N, 3)
        # Basic cleanup
        circles = np.asarray(circles, dtype=float)

        # Sort stably to avoid permutation between frames
        circles = sort_circles_top_left_to_bottom_right(circles, y_tol=y_tol)

        # If we got more than expected, keep the best subset:
        # Here: take expected circles with smallest radius variance around median radius.
        # (simple heuristic; replace if you have better selection logic)
        if circles.shape[0] > expected:
            radii = circles[:, 2]
            med = np.median(radii)
            idx = np.argsort(np.abs(radii - med))[:expected]
            circles = circles[idx]
            circles = sort_circles_top_left_to_bottom_right(circles, y_tol=y_tol)

        if circles.shape[0] != expected:
            return DetectResult(ok=False, reason=f"gears_count_mismatch:{circles.shape[0]}")

        radii_vec = circles[:, 2].astype(np.float64)

        if bool(self.get_parameter("radius_round").value):
            radii_vec = np.rint(radii_vec)

        # Optional sanity check
        max_std = float(self.get_parameter("max_radius_std").value)
        if np.std(radii_vec) > max_std:
            return DetectResult(ok=False, reason="radius_std_too_high")

        # Debug image
        dbg = bgr.copy()
        for (x, y, r) in circles:
            cv2.circle(dbg, (int(round(x)), int(round(y))), int(round(r)), (0, 255, 0), 2)
            cv2.circle(dbg, (int(round(x)), int(round(y))), 2, (0, 0, 255), 3)

        return DetectResult(ok=True, reason="ok", circles=circles, vector=radii_vec, debug_img=dbg)

    # -------------------------
    # Aggregation
    # -------------------------
    @staticmethod
    def _mean_vector(vectors: List[np.ndarray]) -> np.ndarray:
        # vectors: list of (K,) float arrays
        mat = np.vstack([v.reshape(1, -1) for v in vectors]).astype(np.float64)
        return np.mean(mat, axis=0)

    def _publish_data(self, payload: dict):
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.pub_data.publish(msg)

    def _publish_debug_image(self, dbg_bgr: np.ndarray):
        try:
            ros_img = self.bridge.cv2_to_imgmsg(dbg_bgr, encoding="bgr8")
            self.pub_img.publish(ros_img)
        except Exception as e:
            self.get_logger().warn(f"cv2_to_imgmsg failed: {e}")

    # -------------------------
    # Services
    # -------------------------
    def _on_calibration(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        calib_frames = int(self.get_parameter("calib_frames").value)

        accepted_vectors: List[np.ndarray] = []
        prev_id = 0

        # Initialize prev_id to current to ensure we wait for new frames
        with self._cond:
            prev_id = self._last_frame_id

        for i in range(calib_frames):
            ok, prev_id, img = self._wait_new_frame(prev_id, timeout_s=2.0)
            if not ok or img is None:
                self.get_logger().info(f"calibration: frame {i+1}/{calib_frames} timeout")
                continue

            det = self._detect_gears(img)
            if det.ok and det.vector is not None:
                accepted_vectors.append(det.vector)
                self.get_logger().info(
                    f"calibration: frame {i+1}/{calib_frames} accepted gears={self.expected_gears}"
                )
                if det.debug_img is not None:
                    self._publish_debug_image(det.debug_img)
            else:
                # discard bad frame (do not affect computation except reducing sample count)
                gears_info = det.reason
                self.get_logger().info(
                    f"calibration: frame {i+1}/{calib_frames} ignored ({gears_info})"
                )
                if det.debug_img is not None:
                    self._publish_debug_image(det.debug_img)

        if len(accepted_vectors) == 0:
            response.success = False
            response.message = "calib_no_good_frames"
            self._calib_vector = None
            self._publish_data({"mode": "calibration", "ok": False, "reason": response.message})
            return response

        calib_vec = self._mean_vector(accepted_vectors)
        self._calib_vector = calib_vec

        response.success = True
        response.message = f"calib_ok n_good={len(accepted_vectors)} vector={calib_vec.tolist()}"
        self._publish_data(
            {"mode": "calibration", "ok": True, "n_good": len(accepted_vectors), "vector": calib_vec.tolist()}
        )
        return response

    def _on_check(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        if self._calib_vector is None:
            response.success = False
            response.message = "no_calibration"
            self._publish_data({"mode": "check", "ok": False, "reason": response.message})
            return response

        check_frames = int(self.get_parameter("check_frames").value)
        diff_thr = float(self.get_parameter("diff_threshold").value)

        accepted_vectors: List[np.ndarray] = []
        prev_id = 0
        with self._cond:
            prev_id = self._last_frame_id

        for i in range(check_frames):
            ok, prev_id, img = self._wait_new_frame(prev_id, timeout_s=2.0)
            if not ok or img is None:
                self.get_logger().info(f"check: frame {i+1}/{check_frames} timeout")
                continue

            det = self._detect_gears(img)
            if det.ok and det.vector is not None:
                accepted_vectors.append(det.vector)
                self.get_logger().info(f"check: frame {i+1}/{check_frames} accepted gears={self.expected_gears}")
                if det.debug_img is not None:
                    self._publish_debug_image(det.debug_img)
            else:
                self.get_logger().info(f"check: frame {i+1}/{check_frames} ignored ({det.reason})")
                if det.debug_img is not None:
                    self._publish_debug_image(det.debug_img)

        if len(accepted_vectors) == 0:
            response.success = False
            response.message = "check_no_good_frames"
            self._publish_data({"mode": "check", "ok": False, "reason": response.message})
            return response

        check_vec = self._mean_vector(accepted_vectors)

        diffs = np.abs(check_vec - self._calib_vector)
        is_ok = bool(np.all(diffs < diff_thr))

        response.success = is_ok
        response.message = (
            f"check_ok n_good={len(accepted_vectors)} diffs={diffs.tolist()}"
            if is_ok
            else f"check_fail n_good={len(accepted_vectors)} diffs={diffs.tolist()}"
        )

        self._publish_data(
            {
                "mode": "check",
                "ok": is_ok,
                "n_good": len(accepted_vectors),
                "calib_vector": self._calib_vector.tolist(),
                "check_vector": check_vec.tolist(),
                "diffs": diffs.tolist(),
                "diff_threshold": diff_thr,
            }
        )
        return response


def main():
    rclpy.init()
    node = GearsCheckNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()