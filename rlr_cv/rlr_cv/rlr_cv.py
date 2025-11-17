#!/usr/bin/env python3

import time
from pathlib import Path
from typing import Optional, Tuple

import cv2 as cv
import numpy as np
from ultralytics import YOLO

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped

# ==== CONFIG ==== 
# Модель лежит рядом с этим файлом: ~/ros2_ws/src/hoba/best_fixed.pt
THIS_DIR = Path(__file__).resolve().parent
MODEL = str(THIS_DIR / "best_fixed.pt")

CAM_INDEX = 7
IMGZ = 640

# Intrinsics @ 1920x1080
FX, FY = 1769.9768, 1758.5401
CX, CY = 1038.8629, 534.7359
CAP_W, CAP_H = 1920, 1080

# Distortion (k1,k2,p1,p2,k3)
DIST = np.array([[
    -0.01703867068037522, 1.6164689285399254,
    -0.0020304471957141106, -0.0014182118041878213,
    -8.2797801162119313
]], dtype=np.float64).T

K = np.array([[FX, 0.0, CX],
              [0.0, FY, CY],
              [0.0, 0.0, 1.0]], dtype=np.float64)

# Default plane distance (камера перпендикулярна столу)
H_CAM_M = 0.75

PAD_REL = 0.07
MIN_PAD = 8
# ==============

_MODEL: Optional[YOLO] = None


def _get_model() -> YOLO:
    global _MODEL
    if _MODEL is None:
        _MODEL = YOLO(MODEL)
    return _MODEL


# ---------- camera ----------
def _lap_var(gray: np.ndarray) -> float:
    return float(cv.Laplacian(gray, cv.CV_64F).var())


def _grab_stable_frame(
    cam_index: int,
    warmup_s: float = 1.0,
    prefer_sharpest: bool = True,
) -> Optional[np.ndarray]:
    cap = cv.VideoCapture(cam_index, cv.CAP_V4L2)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, CAP_W)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, CAP_H)

    t0 = time.time()
    best_var, best, last = -1.0, None, None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        last = frame
        if prefer_sharpest:
            v = _lap_var(cv.cvtColor(frame, cv.COLOR_BGR2GRAY))
            if v > best_var:
                best_var, best = v, frame.copy()
        if time.time() - t0 >= warmup_s:
            break
    cap.release()
    if last is None:
        return None
    return best if (prefer_sharpest and best is not None) else last


# ---------- utils ----------
def undistort_pixel(u, v, K, dist):
    pts = np.array([[[float(u), float(v)]]], dtype=np.float64)
    und = cv.undistortPoints(pts, K, dist, P=K)  # pixel coords with P=K
    return float(und[0, 0, 0]), float(und[0, 0, 1])


def pixel_to_xy_known_depth(u_px: float, v_px: float, Z_m: float,
                            K: np.ndarray, dist: np.ndarray) -> Tuple[float, float]:
    u_u, v_u = undistort_pixel(u_px, v_px, K, dist)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    X = (u_u - cx) / fx * Z_m
    Y = (v_u - cy) / fy * Z_m
    return float(X), float(Y)


def _expand_box(x1: int, y1: int, x2: int, y2: int, W: int, H: int,
                pad_rel: float = PAD_REL, min_pad: int = MIN_PAD) -> Tuple[int, int, int, int]:
    w, h = x2 - x1, y2 - y1
    px = max(min_pad, int(w * pad_rel))
    py = max(min_pad, int(h * pad_rel))
    x1 = max(0, x1 - px)
    y1 = max(0, y1 - py)
    x2 = min(W - 1, x2 + px)
    y2 = min(H - 1, y2 + py)
    return x1, y1, x2, y2


def _gray_world(bgr: np.ndarray) -> np.ndarray:
    b, g, r = cv.split(bgr.astype(np.float32))
    m = (b.mean() + g.mean() + r.mean()) / 3.0
    b *= m / max(b.mean(), 1e-6)
    g *= m / max(g.mean(), 1e-6)
    r *= m / max(r.mean(), 1e-6)
    out = cv.merge([b, g, r])
    return np.clip(out, 0, 255).astype(np.uint8)


def remove_shadows_white_bg_auto(
    bgr_roi: np.ndarray,
    *,
    k_div_frac: float = 0.03,
    s_p: int = 90, v_p: int = 60,
    s_off: int = +5,
    s_clip: Tuple[int, int] = (20, 140),
    v_clip: Tuple[int, int] = (160, 245),
    morph_close: int = 5, morph_open: int = 3,
    use_grayworld: bool = False
) -> np.ndarray:
    bgr = _gray_world(bgr_roi) if use_grayworld else bgr_roi
    h, w = bgr.shape[:2]
    hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
    S, V = hsv[..., 1], hsv[..., 2]

    # фон по краям
    b = max(6, int(0.1 * min(h, w)))
    bm = np.zeros((h, w), dtype=bool)
    bm[:b, :] = True
    bm[-b:, :] = True
    bm[:, :b] = True
    bm[:, -b:] = True

    s_thr = int(np.clip(np.percentile(S[bm], s_p) + s_off, *s_clip))
    v_thr = int(np.clip(np.percentile(V[bm], v_p), *v_clip))

    # интенсивность
    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    k = max(31, (int(min(h, w) * k_div_frac) // 2) * 2 + 1)
    bg = cv.medianBlur(gray, k)
    norm = cv.GaussianBlur(cv.divide(gray, bg, scale=255), (0, 0), 1.0)
    _, bin_otsu = cv.threshold(norm, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    obj_int = (bin_otsu == 0)

    # цвет
    paper = (S < s_thr) & (V > v_thr)
    obj_color = (S > s_thr + 10) | (V < v_thr - 40)

    mask = ((obj_int | obj_color) & (~paper)).astype(np.uint8) * 255

    # морфология
    kC = cv.getStructuringElement(cv.MORPH_ELLIPSE, (morph_close, morph_close))
    kO = cv.getStructuringElement(cv.MORPH_ELLIPSE, (morph_open, morph_open))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kC)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kO)

    # оставляем самый большой компонент
    num, labels, stats, _ = cv.connectedComponentsWithStats(mask, 8)
    if num > 1:
        idx = 1 + np.argmax(stats[1:, cv.CC_STAT_AREA])
        mask = np.where(labels == idx, 255, 0).astype(np.uint8)

    return mask


def _com(mask_u8: np.ndarray) -> Optional[Tuple[float, float]]:
    _, binm = cv.threshold(mask_u8, 127, 255, cv.THRESH_BINARY)
    M = cv.moments(binm, binaryImage=True)
    if M["m00"] == 0:
        return None
    return M["m10"] / M["m00"], M["m01"] / M["m00"]


def _snap_inside(cx: float, cy: float, mask_u8: np.ndarray, erode_iters: int = 1) -> Optional[Tuple[float, float]]:
    _, binm = cv.threshold(mask_u8, 127, 255, cv.THRESH_BINARY)
    h, w = binm.shape
    ix, iy = np.clip(int(round(cx)), 0, w - 1), np.clip(int(round(cy)), 0, h - 1)
    if binm[iy, ix] > 0:
        return float(ix), float(iy)

    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    er = cv.erode(binm, k, iterations=erode_iters)
    src = er if cv.countNonZero(er) > 0 else binm

    ys, xs = np.nonzero(src)
    if xs.size == 0:
        return None
    pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    j = int(np.argmin((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2))
    return float(pts[j, 0]), float(pts[j, 1])


# ---------- main API ----------
def measure_xy_once(
    *,
    display: bool = False,
    device: str = "cpu",
    warmup_s: float = 1.5,
    prefer_sharpest: bool = True,
    s_p: int = 90, v_p: int = 60, s_off: int = 5,
    s_clip: Tuple[int, int] = (20, 140),
    v_clip: Tuple[int, int] = (160, 245),
    morph_close: int = 5, morph_open: int = 3,
    k_div_frac: float = 0.03,
    use_grayworld: bool = False,
    z_override_m: Optional[float] = None  # если задано, используем вместо H_CAM_M
) -> Optional[Tuple[float, float]]:
    frame = _grab_stable_frame(
            cam_index,
            warmup_s=warmup_s, prefer_sharpest=prefer_sharpest
    )
    if frame is None:
        return None

    Himg, Wimg = frame.shape[:2]
    _ = Himg, Wimg  # зарезервировано под возможные проверки

    model = _get_model()
    res = model.predict(source=frame, imgsz=IMGZ, device=device, conf=0.25, verbose=False)[0]
    if res.boxes is None or res.boxes.xyxy.numel() == 0:
        return None

    idx = int(res.boxes.conf.argmax().item())
    x1, y1, x2, y2 = res.boxes.xyxy[idx].detach().cpu().numpy().astype(int)
    x1, y1, x2, y2 = _expand_box(x1, y1, x2, y2, Wimg, Himg)
    if x2 <= x1 or y2 <= y1:
        return None

    roi = frame[y1:y2, x1:x2].copy()
    mask = remove_shadows_white_bg_auto(
        roi,
        k_div_frac=k_div_frac, s_p=s_p, v_p=v_p, s_off=s_off,
        s_clip=s_clip, v_clip=v_clip,
        morph_close=morph_close, morph_open=morph_open,
        use_grayworld=use_grayworld
    )

    c = _com(mask)
    if c is None:
        return None
    c = _snap_inside(*c, mask)
    if c is None:
        return None

    u_img, v_img = x1 + c[0], y1 + c[1]

    # Compute XY using chosen depth
    Z_used = float(z_override_m) if z_override_m is not None else float(H_CAM_M)
    X_m, Y_m = pixel_to_xy_known_depth(u_img, v_img, Z_used, K, DIST)

    if display:
        vis = frame.copy()
        cv.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv.circle(vis, (int(round(u_img)), int(round(v_img))), 6, (0, 0, 255), -1)
        cv.imshow("frame", vis)
        cv.waitKey(1)

    return float(X_m), float(Y_m)


# ---------- ROS2 Node ----------
class XYMeasureNode(Node):
    def __init__(self):
        super().__init__("hoba_xy_measure")

        # Параметры
        self.declare_parameter("device", "cpu")
        self.declare_parameter("camera_index", CAM_INDEX)
        self.declare_parameter("warmup_s", 1.5)
        self.declare_parameter("prefer_sharpest", True)
        self.declare_parameter("plane_depth", H_CAM_M)
        self.declare_parameter("display", False)
        self.declare_parameter("publish_rate", 1.0)
        self.declare_parameter("frame_id", "camera_optical_frame")
        self.declare_parameter("use_grayworld", False)

        self.declare_parameter("s_p", 90)
        self.declare_parameter("v_p", 60)
        self.declare_parameter("s_off", 5)
        self.declare_parameter("s_clip_low", 20)
        self.declare_parameter("s_clip_high", 140)
        self.declare_parameter("v_clip_low", 160)
        self.declare_parameter("v_clip_high", 245)
        self.declare_parameter("morph_close", 5)
        self.declare_parameter("morph_open", 3)
        self.declare_parameter("k_div_frac", 0.03)

        # Считать параметры
        self.device = self.get_parameter("device").value
        self.camera_index = int(self.get_parameter("camera_index").value)
        self.warmup_s = float(self.get_parameter("warmup_s").value)
        self.prefer_sharpest = bool(self.get_parameter("prefer_sharpest").value)
        self.plane_depth = float(self.get_parameter("plane_depth").value)
        self.display = bool(self.get_parameter("display").value)
        self.frame_id = self.get_parameter("frame_id").value
        self.use_grayworld = bool(self.get_parameter("use_grayworld").value)

        self.s_p = int(self.get_parameter("s_p").value)
        self.v_p = int(self.get_parameter("v_p").value)
        self.s_off = int(self.get_parameter("s_off").value)
        self.s_clip = (
            int(self.get_parameter("s_clip_low").value),
            int(self.get_parameter("s_clip_high").value),
        )
        self.v_clip = (
            int(self.get_parameter("v_clip_low").value),
            int(self.get_parameter("v_clip_high").value),
        )
        self.morph_close = int(self.get_parameter("morph_close").value)
        self.morph_open = int(self.get_parameter("morph_open").value)
        self.k_div_frac = float(self.get_parameter("k_div_frac").value)

        publish_rate = float(self.get_parameter("publish_rate").value)
        period = 1.0 / max(publish_rate, 1e-3)

        # паблишер
        self.pub_xy = self.create_publisher(PointStamped, "hoba/xy", 10)

        # таймер
        self.timer = self.create_timer(period, self.timer_cb)

        self.get_logger().info(
            f"XYMeasureNode started: device={self.device}, "
            f"camera_index={self.camera_index}, plane_depth={self.plane_depth:.3f} m"
        )

    def timer_cb(self):
        coords = measure_xy_once(
            display=self.display,
            device=self.device,
            warmup_s=self.warmup_s,
            prefer_sharpest=self.prefer_sharpest,
            s_p=self.s_p,
            v_p=self.v_p,
            s_off=self.s_off,
            s_clip=self.s_clip,
            v_clip=self.v_clip,
            morph_close=self.morph_close,
            morph_open=self.morph_open,
            k_div_frac=self.k_div_frac,
            use_grayworld=self.use_grayworld,
            z_override_m=self.plane_depth,
            cam_index=self.camera_index,

        )

        if coords is None:
            self.get_logger().debug("No detection / measurement failed")
            return

        x_m, y_m = coords

        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.point.x = x_m
        msg.point.y = y_m
        msg.point.z = self.plane_depth  # Z = высота камеры над столом

        self.pub_xy.publish(msg)
        self.get_logger().debug(
            f"Published XY: ({x_m:.4f}, {y_m:.4f}, {self.plane_depth:.4f}) in {self.frame_id}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = XYMeasureNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
