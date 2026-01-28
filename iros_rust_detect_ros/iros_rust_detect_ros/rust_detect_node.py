#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import threading
import json
import time

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Bool
from std_srvs.srv import Trigger
from cv_bridge import CvBridge

import message_filters
import onnxruntime as ort


# -----------------------------
# ORT / math utils
# -----------------------------
IMG_PAD_MULT = 32


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def resize_shorter_side_keep_ar(img: np.ndarray, short: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return img
    m = min(h, w)
    if m == short:
        return img
    scale = float(short) / float(m)
    nh = int(round(h * scale))
    nw = int(round(w * scale))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)


def center_crop(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h < size or w < size:
        img = cv2.resize(img, (max(size, w), max(size, h)), interpolation=cv2.INTER_LINEAR)
        h, w = img.shape[:2]
    y0 = (h - size) // 2
    x0 = (w - size) // 2
    return img[y0:y0 + size, x0:x0 + size]


def one_bbox_union_from_prob(
    prob: np.ndarray,
    thr: float,
    min_area: int,
    close_k: int,
) -> Tuple[bool, Optional[Tuple[int, int, int, int]], int, np.ndarray]:
    mask01 = (prob >= float(thr)).astype(np.uint8)

    if close_k > 0:
        kernel = np.ones((int(close_k), int(close_k)), dtype=np.uint8)
        mask01 = cv2.morphologyEx(mask01 * 255, cv2.MORPH_CLOSE, kernel)
        mask01 = (mask01 > 0).astype(np.uint8)

    empty = np.zeros(prob.shape[:2], dtype=np.uint8)
    if mask01.sum() == 0:
        return False, None, 0, empty

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask01 * 255, connectivity=8)

    keep = []
    for lab in range(1, n):
        area = int(stats[lab, 4])
        if area >= int(min_area):
            keep.append(lab)

    if not keep:
        return False, None, 0, empty

    keep_mask = np.isin(labels, keep)
    ys, xs = np.where(keep_mask)
    if xs.size == 0:
        return False, None, 0, empty

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    area_total = int(keep_mask.sum())
    return True, (x0, y0, x1, y1), area_total, keep_mask.astype(np.uint8)


def region_confidence(prob: np.ndarray, keep_mask_u8: np.ndarray, mode: str, topk_frac: float) -> float:
    vals = prob[keep_mask_u8 > 0]
    if vals.size == 0:
        return 0.0

    mode = (mode or "p95").lower()

    if mode == "mean":
        return float(vals.mean())
    if mode == "max":
        return float(vals.max())

    if mode.startswith("p") and mode[1:].isdigit():
        p = int(mode[1:])
        p = max(0, min(100, p))
        return float(np.percentile(vals, p))

    if mode == "topk":
        frac = float(topk_frac)
        if not (0.0 < frac <= 1.0):
            frac = 0.01
        k = max(1, int(vals.size * frac))
        kth = np.partition(vals, -k)[-k]
        top = vals[vals >= kth]
        return float(top.mean()) if top.size > 0 else float(vals.max())

    return float(vals.mean())

def make_prob_vis(img_rgb: np.ndarray, prob: np.ndarray, thr: float, mode: str) -> Tuple[np.ndarray, str]:
    """
    mode:
      - prob_u8:      mono8 = prob*255
      - mask_x_gray:  mono8 = gray where prob>=thr else 0
      - prob_x_gray:  mono8 = gray * prob
      - red_overlay:  bgr8  = red mask over original
      - glow:         bgr8  = "glowing" highlight over original (eye-catching, not red)
      - invert_glow:  bgr8  = invert under mask + glow edge (very noticeable)
    """
    mode = (mode or "prob_u8").lower()
    prob01 = np.clip(prob, 0.0, 1.0).astype(np.float32)

    if mode == "prob_u8":
        out = (prob01 * 255.0).astype(np.uint8)
        return out, "mono8"

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    if mode == "mask_x_gray":
        mask01 = (prob01 >= float(thr)).astype(np.uint8)
        out = (gray * mask01).astype(np.uint8)
        return out, "mono8"

    if mode == "prob_x_gray":
        out = (gray.astype(np.float32) * prob01).clip(0, 255).astype(np.uint8)
        return out, "mono8"

    if mode == "red_overlay":
        mask01 = (prob01 >= float(thr)).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        overlay = img_bgr.copy()
        overlay[mask01 > 0] = (0, 0, 255)
        out = cv2.addWeighted(img_bgr, 0.7, overlay, 0.3, 0.0)
        return out, "bgr8"

    # ---------- new eye-catching overlays ----------
    if mode in ("glow", "invert_glow"):
        # base image
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # hard mask at thr + soft alpha from prob
        mask01 = (prob01 >= float(thr)).astype(np.uint8)

        # make a soft "glow" halo around mask by dilate + blur
        k = max(3, int(round(min(img_bgr.shape[:2]) * 0.01)))  # ~1% of min side
        if k % 2 == 0:
            k += 1
        kernel = np.ones((max(3, k // 2), max(3, k // 2)), dtype=np.uint8)

        dil = cv2.dilate(mask01, kernel, iterations=1)
        halo = cv2.GaussianBlur(dil.astype(np.float32), (k, k), 0)  # 0..1-ish
        halo = np.clip(halo, 0.0, 1.0)

        # edge for extra contrast
        edges = cv2.Canny((mask01 * 255).astype(np.uint8), 50, 150)
        edges = (edges > 0).astype(np.float32)

        # color palette (cyan-ish / neon)
        # OpenCV BGR: cyan = (255,255,0) is yellowish; real cyan is (255,255,0)?? actually BGR cyan=(255,255,0) -> (B=255,G=255,R=0)
        glow_color = np.array([255, 255, 0], dtype=np.float32)   # cyan
        edge_color = np.array([255, 255, 255], dtype=np.float32) # white

        base = img_bgr.astype(np.float32)

        # optional invert under mask (very noticeable)
        if mode == "invert_glow":
            inv = 255.0 - base
            a_inv = (prob01 * mask01.astype(np.float32)) * 0.55  # strength
            base = base * (1.0 - a_inv[..., None]) + inv * a_inv[..., None]

        # glow layer strength:
        # - inside mask: use prob as alpha
        # - around: use halo
        a_inside = (prob01 * mask01.astype(np.float32)) * 0.65
        a_halo = halo * 0.35
        a = np.clip(a_inside + a_halo, 0.0, 1.0)

        out = base * (1.0 - a[..., None]) + glow_color[None, None, :] * (a[..., None] * 1.0)

        # add bright edge
        a_e = edges * 0.9
        out = out * (1.0 - a_e[..., None]) + edge_color[None, None, :] * a_e[..., None]

        out = np.clip(out, 0, 255).astype(np.uint8)
        return out, "bgr8"

    # fallback
    out = (prob01 * 255.0).astype(np.uint8)
    return out, "mono8"


# -----------------------------
# ORT session container
# -----------------------------
@dataclass
class OrtxModels:
    sess_r: ort.InferenceSession
    r_in: str
    r_out: str
    sess_u: ort.InferenceSession
    u_in: str
    u_out: str


def make_session(path: str, use_cuda: bool) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]
    return ort.InferenceSession(path, sess_options=so, providers=providers)


# -----------------------------
# ROS2 Node (Trigger service)
# -----------------------------
class RustDetectNode(Node):
    """
    Совместимая по архитектуре нода:
      - подписка на image_topic (и опционально sync с CameraInfo) -> кешируем последний кадр
      - сервис Trigger: по вызову обрабатываем последний кадр
      - pub: rust/prob (Image), rust/detected (Bool)
    Новая логика:
      - ResNet (ONNX) делает классификацию rust/not rust
      - UNet (ONNX) запускается ТОЛЬКО если ResNet сказал rust
      - rust/detected публикуется от ResNet (классификация)
      - Trigger.message = JSON со всеми полями (ResNet + UNet метрики)
    """

    def __init__(self):
        super().__init__("rust_detect_node")

        # --- params (как раньше по стилю) ---
        self.declare_parameter("device", "auto")          # auto|cpu|cuda
        self.declare_parameter("amp", False)              # совместимость, не используется (ORT)

        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera_info")
        self.declare_parameter("use_camera_info", False)
        self.declare_parameter("queue_size", 5)
        self.declare_parameter("slop", 0.05)

        self.declare_parameter("prob_vis_mode", "prob_u8")
        self.declare_parameter("service_name", "/rust_detect/run")
        self.declare_parameter("publish_detected_topic", True)

        # --- ORT pipeline params ---
        self.declare_parameter("resnet_onnx", "")         # путь к resnet*.onnx
        self.declare_parameter("unet_onnx", "")           # путь к unet*.onnx
        self.declare_parameter("resnet_thr", 0.5)         # порог классификации
        self.declare_parameter("mask_thr", 0.5)           # порог для визуализации/бинаризации маски
        self.declare_parameter("warmup", 20)              # прогрев сессий
        self.declare_parameter("log_timing", True)

        # Candidate extraction + confidence (как раньше)
        self.declare_parameter("thr", 0.35)               # порог candidate extraction по prob-map
        self.declare_parameter("min_area", 200)
        self.declare_parameter("close_k", 0)

        self.declare_parameter("conf_thr", 0.85)
        self.declare_parameter("conf_mode", "p95")
        self.declare_parameter("topk_frac", 0.01)

        self.bridge = CvBridge()
        self._img_lock = threading.Lock()
        self._proc_lock = threading.Lock()
        self._last_img_msg: Optional[Image] = None

        # --- choose CUDA provider ---
        device_str = self.get_parameter("device").get_parameter_value().string_value.strip().lower()
        if device_str == "auto":
            device_str = "cuda" if ("CUDAExecutionProvider" in ort.get_available_providers()) else "cpu"
        if device_str not in ("cpu", "cuda"):
            device_str = "cpu"
        self.use_cuda = (device_str == "cuda")

        # --- resolve model paths ---
        resnet_path = self.get_parameter("resnet_onnx").get_parameter_value().string_value.strip()
        unet_path = self.get_parameter("unet_onnx").get_parameter_value().string_value.strip()

        if not resnet_path or not unet_path:
            # try take from share/<pkg>/onnx_models
            try:
                from ament_index_python.packages import get_package_share_directory
                share = Path(get_package_share_directory("iros_rust_detect_ros"))
                # prefer fp16
                cand_r = share / "onnx_models" / "resnet_fp16.onnx"
                cand_u = share / "onnx_models" / "unet_fp16.onnx"
                if not cand_r.exists():
                    cand_r = share / "onnx_models" / "resnet.onnx"
                if not cand_u.exists():
                    cand_u = share / "onnx_models" / "unet.onnx"
                if not resnet_path and cand_r.exists():
                    resnet_path = str(cand_r)
                if not unet_path and cand_u.exists():
                    unet_path = str(cand_u)
            except Exception:
                pass

        if not resnet_path or not unet_path:
            raise RuntimeError(
                "ONNX paths are empty. Set params resnet_onnx and unet_onnx, "
                "or install onnx_models into share/iros_rust_detect_ros/onnx_models."
            )

        if not Path(resnet_path).exists():
            raise RuntimeError(f"ResNet ONNX not found: {resnet_path}")
        if not Path(unet_path).exists():
            raise RuntimeError(f"UNet ONNX not found: {unet_path}")

        # --- create sessions ---
        self.models = self._load_ort_models(resnet_path, unet_path, self.use_cuda)

        self.get_logger().info(f"ORT providers available: {ort.get_available_providers()} | use_cuda={self.use_cuda}")
        self.get_logger().info(f"Loaded ONNX: resnet={resnet_path} unet={unet_path}")

        # --- warmup ---
        warmup = int(self.get_parameter("warmup").value)
        if warmup > 0:
            self._warmup(warmup)

        # Publishers
        self.pub_prob = self.create_publisher(Image, "rust/prob", 10)
        self.pub_detected = self.create_publisher(Bool, "rust/detected", 10)

        # Service
        srv_name = self.get_parameter("service_name").get_parameter_value().string_value
        self.srv = self.create_service(Trigger, srv_name, self.on_srv_run)
        self.get_logger().info(f"Service ready: {srv_name} (std_srvs/Trigger)")

        # Subscribers (cache only)
        image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        use_camera_info = bool(self.get_parameter("use_camera_info").value)

        if not use_camera_info:
            self.get_logger().info(f"Subscribing (cache only): image={image_topic}")
            self.create_subscription(Image, image_topic, self.on_image_only, qos_profile_sensor_data)
            return

        cam_info_topic = self.get_parameter("camera_info_topic").get_parameter_value().string_value
        queue_size = int(self.get_parameter("queue_size").value)
        slop = float(self.get_parameter("slop").value)

        sub_img = message_filters.Subscriber(self, Image, image_topic, qos_profile=qos_profile_sensor_data)
        sub_info = message_filters.Subscriber(self, CameraInfo, cam_info_topic, qos_profile=qos_profile_sensor_data)
        ats = message_filters.ApproximateTimeSynchronizer([sub_img, sub_info], queue_size=queue_size, slop=slop)
        ats.registerCallback(self.on_synced_cache)

        self.get_logger().info(f"Subscribing (cache sync): image={image_topic} camera_info={cam_info_topic}")

    def _load_ort_models(self, resnet_path: str, unet_path: str, use_cuda: bool) -> OrtxModels:
        sess_r = make_session(resnet_path, use_cuda)
        sess_u = make_session(unet_path, use_cuda)

        r_in = sess_r.get_inputs()[0].name
        r_out = sess_r.get_outputs()[0].name
        u_in = sess_u.get_inputs()[0].name
        u_out = sess_u.get_outputs()[0].name

        self.get_logger().info(f"ResNet providers: {sess_r.get_providers()}")
        self.get_logger().info(f"UNet providers:   {sess_u.get_providers()}")

        return OrtxModels(sess_r, r_in, r_out, sess_u, u_in, u_out)

    def _warmup(self, n: int):
        # ResNet input: (1,3,224,224)
        x_r = np.zeros((1, 3, 224, 224), dtype=np.float32)
        # UNet input: (1,3,480,480)
        x_u = np.zeros((1, 3, 480, 480), dtype=np.float32)

        t0 = time.perf_counter()
        for _ in range(n):
            _ = self.models.sess_r.run([self.models.r_out], {self.models.r_in: x_r})[0]
            _ = self.models.sess_u.run([self.models.u_out], {self.models.u_in: x_u})[0]
        dt = (time.perf_counter() - t0) * 1000.0
        self.get_logger().info(f"Warmup {n} iters done: {dt:.1f} ms")

    def on_image_only(self, msg_img: Image):
        with self._img_lock:
            self._last_img_msg = msg_img

    def on_synced_cache(self, msg_img: Image, _info: CameraInfo):
        with self._img_lock:
            self._last_img_msg = msg_img

    def on_srv_run(self, _req: Trigger.Request, res: Trigger.Response):
        with self._img_lock:
            msg_img = self._last_img_msg

        if msg_img is None:
            res.success = False
            res.message = json.dumps({"detected": False, "error": "no_image_yet"}, separators=(",", ":"))
            return res

        with self._proc_lock:
            info = self._process_once_and_publish_prob(msg_img)

        # success = ResNet классификация (detected)
        res.success = bool(info.get("detected", False))
        res.message = json.dumps(info, separators=(",", ":"))
        return res

    # ---------- Inference blocks ----------
    def _resnet_prob(self, rgb_u8: np.ndarray) -> float:
        """
        Как torchvision transforms:
          Resize(256) по короткой стороне (с сохранением AR),
          CenterCrop(224),
          Normalize(ImageNet),
          -> float32 NCHW (1,3,224,224)
        """
        img = resize_shorter_side_keep_ar(rgb_u8, 256)
        img = center_crop(img, 224)

        x = img.astype(np.float32) / 255.0  # HWC 0..1
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x = (x - mean) / std
        x = np.transpose(x, (2, 0, 1))[None, ...].astype(np.float32)  # 1x3x224x224

        logit = self.models.sess_r.run([self.models.r_out], {self.models.r_in: x})[0]
        logit = float(np.asarray(logit).reshape(-1)[0])
        return float(1.0 / (1.0 + np.exp(-logit)))

    def _unet_probmap_480(self, rgb_u8: np.ndarray) -> np.ndarray:
        """
        ВАЖНО: ровно 480x480, чтобы соответствовать ONNX (если экспорт был фиксированный).
        Вход float32 NCHW (1,3,480,480).
        Выход: prob map float32 (480,480) в диапазоне 0..1.
        """
        img = cv2.resize(rgb_u8, (480, 480), interpolation=cv2.INTER_LINEAR)
        x = img.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None, ...].astype(np.float32)

        out = self.models.sess_u.run([self.models.u_out], {self.models.u_in: x})[0]
        out = np.asarray(out)

        if out.ndim == 4:
            out = out[0, 0]
        elif out.ndim == 3:
            out = out[0]
        else:
            raise RuntimeError(f"Unexpected UNet output shape: {out.shape}")

        out = out.astype(np.float32)

        # если logits
        if out.min() < 0.0 or out.max() > 1.0:
            out = sigmoid_np(out)

        return np.clip(out, 0.0, 1.0)

    # ---------- Main processing ----------
    def _process_once_and_publish_prob(self, msg_img: Image) -> dict:
        prob_vis_mode = self.get_parameter("prob_vis_mode").get_parameter_value().string_value
        publish_detected_topic = bool(self.get_parameter("publish_detected_topic").value)

        resnet_thr = float(self.get_parameter("resnet_thr").value)
        mask_thr = float(self.get_parameter("mask_thr").value)

        thr = float(self.get_parameter("thr").value)
        min_area = int(self.get_parameter("min_area").value)
        close_k = int(self.get_parameter("close_k").value)

        conf_thr = float(self.get_parameter("conf_thr").value)
        conf_mode = self.get_parameter("conf_mode").get_parameter_value().string_value
        topk_frac = float(self.get_parameter("topk_frac").value)

        log_timing = bool(self.get_parameter("log_timing").value)

        t0 = time.perf_counter()

        bgr = self.bridge.imgmsg_to_cv2(msg_img, desired_encoding="bgr8")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        oh, ow = rgb.shape[:2]

        t1 = time.perf_counter()
        rust_prob = self._resnet_prob(rgb)
        rust_pred = bool(rust_prob >= resnet_thr)

        # detected публикуется от ResNet (классификация)
        detected_cls = rust_pred

        t2 = time.perf_counter()

        run_unet = rust_pred
        unet_found = False
        unet_bbox = None
        unet_area = 0
        unet_conf = 0.0
        unet_detected = False

        # prob_full публикуем на rust/prob
        prob_full = np.zeros((oh, ow), dtype=np.float32)

        if run_unet:
            prob_480 = self._unet_probmap_480(rgb)
            prob_full = cv2.resize(prob_480, (ow, oh), interpolation=cv2.INTER_LINEAR)

            unet_found, unet_bbox, unet_area, keep_mask_u8 = one_bbox_union_from_prob(
                prob_full, thr=thr, min_area=min_area, close_k=close_k
            )
            unet_conf = region_confidence(prob_full, keep_mask_u8, conf_mode, topk_frac) if unet_found else 0.0
            unet_detected = bool(unet_found and (unet_conf >= conf_thr))

        t3 = time.perf_counter()

        # publish prob image
        prob_vis, enc = make_prob_vis(rgb, prob_full, thr=mask_thr, mode=prob_vis_mode)
        if enc == "mono8":
            msg_prob = self.bridge.cv2_to_imgmsg(prob_vis, encoding="mono8")
        else:
            msg_prob = self.bridge.cv2_to_imgmsg(prob_vis, encoding="bgr8")
        msg_prob.header = msg_img.header
        self.pub_prob.publish(msg_prob)

        # publish Bool (ResNet)
        if publish_detected_topic:
            m = Bool()
            m.data = bool(detected_cls)
            self.pub_detected.publish(m)

        if log_timing:
            dt_total = (t3 - t0) * 1000.0
            dt_r = (t2 - t1) * 1000.0
            dt_u = (t3 - t2) * 1000.0
            self.get_logger().info(
                f"timing: total={dt_total:.2f}ms resnet={dt_r:.2f}ms unet+post={dt_u:.2f}ms "
                f"| rust_prob={rust_prob:.3f} resnet_pred={int(rust_pred)} unet_ran={int(run_unet)} unet_detected={int(unet_detected)}"
            )

        bbox_list = None if unet_bbox is None else [int(unet_bbox[0]), int(unet_bbox[1]), int(unet_bbox[2]), int(unet_bbox[3])]

        return {
            # ResNet classification = main decision (and rust/detected)
            "detected": bool(detected_cls),
            "rust_prob": float(rust_prob),
            "resnet_thr": float(resnet_thr),
            "run_unet": bool(run_unet),

            # UNet info (report only)
            "unet_found_candidate": bool(unet_found),
            "unet_conf": float(unet_conf),
            "unet_conf_thr": float(conf_thr),
            "unet_conf_mode": str(conf_mode),
            "unet_topk_frac": float(topk_frac),
            "unet_detected": bool(unet_detected),

            # thresholds / morphology
            "thr": float(thr),
            "mask_thr": float(mask_thr),
            "min_area": int(min_area),
            "close_k": int(close_k),

            "area": int(unet_area),
            "bbox_xyxy": bbox_list,

            "prob_topic": "rust/prob",
            "detected_topic": "rust/detected",
            "device": "cuda" if self.use_cuda else "cpu",
        }


def main():
    rclpy.init()
    node = RustDetectNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()