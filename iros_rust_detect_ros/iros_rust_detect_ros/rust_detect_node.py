#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import threading
import json

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Bool
from std_srvs.srv import Trigger
from cv_bridge import CvBridge

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import message_filters


# -----------------------------
# Model utils
# -----------------------------
def build_smp(arch: str, encoder: str, classes: int) -> nn.Module:
    arch = arch.lower()
    if arch == "linknet":
        return smp.Linknet(encoder_name=encoder, encoder_weights="imagenet", classes=classes, activation=None)
    if arch == "unet":
        return smp.Unet(encoder_name=encoder, encoder_weights="imagenet", classes=classes, activation=None)
    if arch in ("unetpp", "unet++"):
        return smp.UnetPlusPlus(encoder_name=encoder, encoder_weights="imagenet", classes=classes, activation=None)
    if arch == "fpn":
        return smp.FPN(encoder_name=encoder, encoder_weights="imagenet", classes=classes, activation=None)
    if arch in ("manet", "ma-net"):
        return smp.MAnet(encoder_name=encoder, encoder_weights="imagenet", classes=classes, activation=None)
    raise ValueError(f"Unknown arch: {arch}")


def _torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)  # type: ignore
    except Exception:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)  # type: ignore
        except TypeError:
            return torch.load(path, map_location="cpu")


def infer_classes_from_state_dict(state: dict) -> Optional[int]:
    key = "segmentation_head.0.weight"
    if key in state and torch.is_tensor(state[key]):
        return int(state[key].shape[0])
    for k, v in state.items():
        if "segmentation_head" in k and k.endswith(".weight") and torch.is_tensor(v):
            return int(v.shape[0])
    return None


def infer_classes_from_module(m: nn.Module) -> Optional[int]:
    try:
        sh = getattr(m, "segmentation_head", None)
        if sh is None:
            return None
        if hasattr(sh, "__getitem__"):
            w = sh[0].weight
            return int(w.shape[0])
    except Exception:
        pass
    return None


@dataclass
class LoadedModel:
    model: nn.Module
    encoder: str
    classes: int
    arch: str


def load_best_pt(weights_path: str, arch: str, encoder: str, classes: int, device: torch.device) -> LoadedModel:
    obj = _torch_load(weights_path)

    if isinstance(obj, nn.Module):
        m = obj.to(device).eval()
        if classes <= 0:
            cls = infer_classes_from_module(m) or 1
        else:
            cls = classes
        return LoadedModel(model=m, encoder=encoder, classes=int(cls), arch=arch)

    if not isinstance(obj, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(obj)}")

    state = obj["model"] if ("model" in obj and isinstance(obj["model"], dict)) else obj

    if classes <= 0:
        classes = int(obj.get("classes", 0) or infer_classes_from_state_dict(state) or 1)

    m = build_smp(arch=arch, encoder=encoder, classes=int(classes)).to(device)
    m.load_state_dict(state, strict=True)
    m.eval()
    return LoadedModel(model=m, encoder=encoder, classes=int(classes), arch=arch)


def pad_to_multiple(img: np.ndarray, m: int = 32) -> Tuple[np.ndarray, int, int]:
    h, w = img.shape[:2]
    nh = ((h + m - 1) // m) * m
    nw = ((w + m - 1) // m) * m
    pad_h = nh - h
    pad_w = nw - w
    if pad_h == 0 and pad_w == 0:
        return img, h, w
    out = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, borderType=cv2.BORDER_CONSTANT, value=0)
    return out, h, w


@torch.no_grad()
def predict_prob_map(
    model: nn.Module,
    encoder: str,
    img_rgb_u8: np.ndarray,
    device: torch.device,
    classes: int,
    rust_class_id: int,
    tile: int,
    stride: int,
    amp: bool,
) -> np.ndarray:
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, "imagenet")

    img = img_rgb_u8.astype(np.float32)
    img = preprocessing_fn(img)

    img_pad, oh, ow = pad_to_multiple(img, 32)
    H, W = img_pad.shape[:2]

    def forward_one(x_bchw: torch.Tensor) -> torch.Tensor:
        if amp and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                return model(x_bchw)
        return model(x_bchw)

    tile = int(tile)
    stride = int(stride) if stride > 0 else tile

    if tile <= 0:
        x = torch.from_numpy(img_pad.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
        logits = forward_one(x)
        if classes == 1:
            prob = torch.sigmoid(logits)[:, 0]
        else:
            prob = torch.softmax(logits, dim=1)[:, rust_class_id]
        return prob[0].float().cpu().numpy()[:oh, :ow]

    acc = np.zeros((H, W), dtype=np.float32)
    cnt = np.zeros((H, W), dtype=np.float32)

    ys = list(range(0, max(1, H - tile + 1), stride))
    xs = list(range(0, max(1, W - tile + 1), stride))
    if ys[-1] != H - tile:
        ys.append(H - tile)
    if xs[-1] != W - tile:
        xs.append(W - tile)

    for y in ys:
        for x0 in xs:
            patch = img_pad[y:y + tile, x0:x0 + tile]
            xt = torch.from_numpy(patch.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
            logits = forward_one(xt)

            if classes == 1:
                p = torch.sigmoid(logits)[:, 0]
            else:
                p = torch.softmax(logits, dim=1)[:, rust_class_id]

            p = p[0].float().cpu().numpy()
            acc[y:y + tile, x0:x0 + tile] += p
            cnt[y:y + tile, x0:x0 + tile] += 1.0

    cnt = np.maximum(cnt, 1.0)
    prob = acc / cnt
    return prob[:oh, :ow]


def one_bbox_union_from_prob(
    prob: np.ndarray,
    thr: float,
    min_area: int,
    close_k: int,
) -> Tuple[bool, Optional[Tuple[int, int, int, int]], int, np.ndarray]:
    """
    Candidate extraction only. Decision is done later by confidence.
    Returns: found_candidate, bbox, area_total, keep_mask_u8
    """
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
    """
    mode = (mode or "prob_u8").lower()
    prob01 = np.clip(prob, 0.0, 1.0).astype(np.float32)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    if mode == "prob_u8":
        out = (prob01 * 255.0).astype(np.uint8)
        return out, "mono8"

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

    out = (prob01 * 255.0).astype(np.uint8)
    return out, "mono8"


# -----------------------------
# ROS2 Node (service-driven, Trigger)
# -----------------------------
class RustDetectNode(Node):
    def __init__(self):
        super().__init__("rust_detect_node")

        # Model params
        self.declare_parameter("weights", "")
        self.declare_parameter("arch", "linknet")
        self.declare_parameter("encoder", "mobilenet_v2")
        self.declare_parameter("classes", 0)
        self.declare_parameter("rust_class_id", 1)

        self.declare_parameter("device", "auto")  # auto|cpu|cuda
        self.declare_parameter("amp", True)

        # Candidate extraction
        self.declare_parameter("tile", 320)
        self.declare_parameter("stride", 256)
        self.declare_parameter("thr", 0.35)
        self.declare_parameter("min_area", 200)
        self.declare_parameter("close_k", 0)

        # Decision by confidence
        self.declare_parameter("conf_thr", 0.85)      # 0..1
        self.declare_parameter("conf_mode", "p95")    # mean|max|p95|topk
        self.declare_parameter("topk_frac", 0.01)     # only for topk

        # ROS I/O
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera_info")
        self.declare_parameter("use_camera_info", False)
        self.declare_parameter("queue_size", 5)
        self.declare_parameter("slop", 0.05)

        # Prob publish mode
        self.declare_parameter("prob_vis_mode", "prob_u8")

        # Service name (Trigger)
        self.declare_parameter("service_name", "/rust_detect/run")

        # Optional publish /rust/detected
        self.declare_parameter("publish_detected_topic", True)

        self.bridge = CvBridge()

        # cache last image
        self._img_lock = threading.Lock()
        self._proc_lock = threading.Lock()
        self._last_img_msg: Optional[Image] = None

        # Load model
        weights = self.get_parameter("weights").get_parameter_value().string_value
        if not weights:
            raise RuntimeError("Parameter 'weights' is empty. Pass -p weights:=/abs/path/to/model.pt or .pth")

        arch = self.get_parameter("arch").get_parameter_value().string_value
        encoder = self.get_parameter("encoder").get_parameter_value().string_value
        classes = int(self.get_parameter("classes").value)

        device_str = self.get_parameter("device").get_parameter_value().string_value.strip().lower()
        if device_str == "auto":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        if device_str not in ("cpu", "cuda"):
            device_str = "cpu"
        self.device = torch.device(device_str)

        self.amp = bool(self.get_parameter("amp").value) and (self.device.type == "cuda")

        lm = load_best_pt(weights_path=weights, arch=arch, encoder=encoder, classes=classes, device=self.device)
        self.model = lm.model
        self.encoder = lm.encoder
        self.classes = lm.classes
        self.arch = lm.arch

        self.get_logger().info(
            f"Loaded: {weights} arch={self.arch} enc={self.encoder} classes={self.classes} device={self.device} amp={self.amp}"
        )

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

        res.success = bool(info.get("detected", False))
        res.message = json.dumps(info, separators=(",", ":"))
        return res

    def _process_once_and_publish_prob(self, msg_img: Image) -> dict:
        tile = int(self.get_parameter("tile").value)
        stride = int(self.get_parameter("stride").value)
        thr = float(self.get_parameter("thr").value)
        min_area = int(self.get_parameter("min_area").value)
        close_k = int(self.get_parameter("close_k").value)
        rust_class_id = int(self.get_parameter("rust_class_id").value)
        prob_vis_mode = self.get_parameter("prob_vis_mode").get_parameter_value().string_value

        conf_thr = float(self.get_parameter("conf_thr").value)
        conf_mode = self.get_parameter("conf_mode").get_parameter_value().string_value
        topk_frac = float(self.get_parameter("topk_frac").value)

        publish_detected_topic = bool(self.get_parameter("publish_detected_topic").value)

        bgr = self.bridge.imgmsg_to_cv2(msg_img, desired_encoding="bgr8")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        prob = predict_prob_map(
            model=self.model,
            encoder=self.encoder,
            img_rgb_u8=rgb,
            device=self.device,
            classes=self.classes,
            rust_class_id=rust_class_id,
            tile=tile,
            stride=stride,
            amp=self.amp,
        )

        found, bbox, area, keep_mask_u8 = one_bbox_union_from_prob(prob, thr=thr, min_area=min_area, close_k=close_k)
        conf = region_confidence(prob, keep_mask_u8, conf_mode, topk_frac) if found else 0.0
        detected = bool(found and (conf >= conf_thr))

        # publish prob image (on every service call)
        prob_vis, enc = make_prob_vis(rgb, prob, thr=thr, mode=prob_vis_mode)
        if enc == "mono8":
            msg_prob = self.bridge.cv2_to_imgmsg(prob_vis, encoding="mono8")
        else:
            msg_prob = self.bridge.cv2_to_imgmsg(prob_vis, encoding="bgr8")
        msg_prob.header = msg_img.header
        self.pub_prob.publish(msg_prob)

        # optional Bool topic
        if publish_detected_topic:
            m = Bool()
            m.data = bool(detected)
            self.pub_detected.publish(m)

        if bbox is None:
            bbox_list = None
        else:
            x0, y0, x1, y1 = bbox
            bbox_list = [int(x0), int(y0), int(x1), int(y1)]

        return {
            "detected": bool(detected),
            "found_candidate": bool(found),
            "conf": float(conf),
            "conf_thr": float(conf_thr),
            "conf_mode": str(conf_mode),
            "topk_frac": float(topk_frac),
            "thr": float(thr),
            "min_area": int(min_area),
            "close_k": int(close_k),
            "area": int(area),
            "bbox_xyxy": bbox_list,
            "tile": int(tile),
            "stride": int(stride),
            "rust_class_id": int(rust_class_id),
            "classes": int(self.classes),
            "arch": str(self.arch),
            "encoder": str(self.encoder),
            "prob_topic": "rust/prob",
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
