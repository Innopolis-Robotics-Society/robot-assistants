#!/usr/bin/env python3

from pathlib import Path
from typing import Optional, Tuple

import cv2 as cv
import numpy as np
from ultralytics import YOLO

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import PointStamped, TransformStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from tf2_ros import TransformBroadcaster, Buffer, TransformListener, TransformException

# ==== CONFIG ====
THIS_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = THIS_DIR / "best_fixed.pt"  # переопределяется через ROS-параметр model_path

IMGZ = 640

# Intrinsics @ 1920x1080 (rectified image)
FX, FY = 1769.9768, 1758.5401
CX, CY = 1038.8629, 534.7359

K = np.array([[FX, 0.0, CX],
              [0.0, FY, CY],
              [0.0, 0.0, 1.0]], dtype=np.float64)
K_INV = np.linalg.inv(K)

PAD_REL = 0.07
MIN_PAD = 8
# ==============

_MODEL: Optional[YOLO] = None
_MODEL_PATH: Path = DEFAULT_MODEL


def set_model_path(path: str) -> None:
    """Установить путь к модели YOLO через параметр ROS."""
    global _MODEL_PATH, _MODEL
    _MODEL_PATH = Path(path)
    _MODEL = None  # при смене пути модель будет загружена заново


def _get_model() -> YOLO:
    global _MODEL
    if _MODEL is None:
        _MODEL = YOLO(str(_MODEL_PATH))
    return _MODEL


# ---------- utils ----------
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


def quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Преобразование кватерниона (x,y,z,w) в матрицу поворота 3x3."""
    x, y, z, w = qx, qy, qz, qw
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    R = np.array([
        [1.0 - 2.0 * (yy + zz),     2.0 * (xy - wz),         2.0 * (xz + wy)],
        [2.0 * (xy + wz),           1.0 - 2.0 * (xx + zz),   2.0 * (yz - wx)],
        [2.0 * (xz - wy),           2.0 * (yz + wx),         1.0 - 2.0 * (xx + yy)],
    ], dtype=np.float64)
    return R


# ---------- main API (детекция в изображении) ----------
def measure_pixel_once(
    frame: np.ndarray,
    *,
    display: bool = False,
    device: str = "cpu",
    s_p: int = 90, v_p: int = 60, s_off: int = 5,
    s_clip: Tuple[int, int] = (20, 140),
    v_clip: Tuple[int, int] = (160, 245),
    morph_close: int = 5, morph_open: int = 3,
    k_div_frac: float = 0.03,
    use_grayworld: bool = False,
) -> Optional[Tuple[float, float, np.ndarray]]:
    """
    На вход: готовый кадр (BGR, rectified).
    На выход: пиксельные координаты центра объекта (u,v) + debug-изображение.
    """
    if frame is None:
        return None

    Himg, Wimg = frame.shape[:2]
    _ = Himg, Wimg

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

    # ---- визуализация с полной маской ----
    vis = frame.copy()
    mask_bool = mask > 0
    overlay = vis.copy()
    overlay_roi = overlay[y1:y2, x1:x2]
    overlay_roi[mask_bool] = (0, 255, 0)  # зелёный объект
    overlay[y1:y2, x1:x2] = overlay_roi
    vis = cv.addWeighted(overlay, 0.5, vis, 0.5, 0)

    # рамка и центр
    cv.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv.circle(vis, (int(round(u_img)), int(round(v_img))), 6, (0, 0, 255), -1)

    if display:
        cv.imshow("hoba_debug", vis)
        cv.waitKey(1)

    return float(u_img), float(v_img), vis


# ---------- ROS2 Node ----------
class XYMeasureNode(Node):
    def __init__(self):
        super().__init__("hoba_xy_measure")

        # Параметры
        self.declare_parameter("device", "cpu")
        self.declare_parameter("display", False)
        self.declare_parameter("publish_rate", 1.0)

        # ВАЖНО: твой фрейм камеры
        self.declare_parameter("frame_id", "camera")          # у тебя камера = "camera"
        self.declare_parameter("marker_frame_id", "marker1")  # AprilTag фрейм
        self.declare_parameter("target_frame_id", "hoba_target")

        self.declare_parameter("use_grayworld", False)
        self.declare_parameter("image_topic", "/image_rect")
        self.declare_parameter("model_path", str(DEFAULT_MODEL))

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
        self.display = bool(self.get_parameter("display").value)
        self.frame_id = self.get_parameter("frame_id").value
        self.marker_frame_id = self.get_parameter("marker_frame_id").value
        self.target_frame_id = self.get_parameter("target_frame_id").value
        self.use_grayworld = bool(self.get_parameter("use_grayworld").value)
        self.image_topic = self.get_parameter("image_topic").value
        self.model_path = self.get_parameter("model_path").value

        set_model_path(self.model_path)

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

        # bridge и хранение последнего кадра
        self.bridge = CvBridge()
        self.last_frame: Optional[np.ndarray] = None

        # TF2
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        # подписка на rectified image
        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_cb,
            10,
        )

        # паблишеры
        self.pub_xy = self.create_publisher(PointStamped, "hoba/xy", 10)
        self.pub_debug = self.create_publisher(Image, "hoba/debug_image", 10)

        # таймер
        self.timer = self.create_timer(period, self.timer_cb)

        self.get_logger().info(
            f"XYMeasureNode started: device={self.device}, "
            f"image_topic={self.image_topic}, "
            f"camera_frame={self.frame_id}, marker_frame={self.marker_frame_id}, "
            f"model_path={self.model_path}, target_frame_id={self.target_frame_id}"
        )

    def image_cb(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.last_frame = frame
        except Exception as e:
            self.get_logger().warn(f"Failed to convert image: {e}")

    def _compute_3d_on_table(self, u_px: float, v_px: float) -> Optional[np.ndarray]:
        """
        Пересечение луча из пикселя (u,v) с плоскостью стола, заданной AprilTag marker1.

        Используем TF: camera -> marker1.
        Считаем, что плоскость стола совпадает с плоскостью XY маркера (ось Z маркера — нормаль).
        """
        try:
            # transform: camera (target) <- marker1 (source)
            # т.е. поза marker1 в системе камеры
            trans = self.tf_buffer.lookup_transform(
                self.frame_id,          # target: camera
                self.marker_frame_id,   # source: marker1
                Time()
            )
        except TransformException as ex:
            self.get_logger().warn(f"TF lookup failed ({self.frame_id} <- {self.marker_frame_id}): {ex}")
            return None

        t = trans.transform.translation
        q = trans.transform.rotation

        # точка на плоскости стола (origin маркера) в системе камеры
        p0 = np.array([t.x, t.y, t.z], dtype=np.float64)

        # нормаль к плоскости стола в системе камеры: R * [0,0,1]
        R = quat_to_rot(q.x, q.y, q.z, q.w)
        n = R @ np.array([0.0, 0.0, 1.0], dtype=np.float64)

        # луч из камеры (камера в начале координат)
        uv1 = np.array([u_px, v_px, 1.0], dtype=np.float64)
        ray = K_INV @ uv1  # направление в координатах камеры
        # нормализуем для стабильности (не обязательно, но аккуратнее)
        ray_norm = ray / np.linalg.norm(ray)

        num = np.dot(n, p0)
        den = np.dot(n, ray_norm)
        if abs(den) < 1e-8:
            self.get_logger().warn("Ray is nearly parallel to table plane")
            return None

        s = num / den
        if s <= 0.0:
            self.get_logger().warn(f"Intersection behind camera or at origin: s={s}")
            return None

        X_cam = s * ray_norm  # 3D точка на плоскости стола в системе камеры
        return X_cam

    def timer_cb(self):
        if self.last_frame is None:
            self.get_logger().debug("No image received yet")
            return

        res = measure_pixel_once(
            self.last_frame,
            display=self.display,
            device=self.device,
            s_p=self.s_p,
            v_p=self.v_p,
            s_off=self.s_off,
            s_clip=self.s_clip,
            v_clip=self.v_clip,
            morph_close=self.morph_close,
            morph_open=self.morph_open,
            k_div_frac=self.k_div_frac,
            use_grayworld=self.use_grayworld,
        )

        if res is None:
            self.get_logger().debug("No detection / measurement failed")
            return

        u_px, v_px, vis = res

        # --- 3D точка через плоскость стола (marker1) ---
        X_cam = self._compute_3d_on_table(u_px, v_px)
        if X_cam is None:
            self.get_logger().debug("Failed to compute 3D point on table")
            return

        x_m, y_m, z_m = float(X_cam[0]), float(X_cam[1]), float(X_cam[2])

        # --- PointStamped в фрейме камеры ---
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.point.x = x_m
        msg.point.y = y_m
        msg.point.z = z_m

        self.pub_xy.publish(msg)

        # --- debug image with mask ---
        try:
            img_msg = self.bridge.cv2_to_imgmsg(vis, encoding="bgr8")
            img_msg.header = msg.header
            self.pub_debug.publish(img_msg)
        except Exception as e:
            self.get_logger().warn(f"Failed to publish debug image: {e}")

        # --- TF: camera -> hoba_target ---
        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = self.frame_id
        t.child_frame_id = self.target_frame_id

        t.transform.translation.x = x_m
        t.transform.translation.y = y_m
        t.transform.translation.z = z_m - 0.08

        # Пока ориентация = как у камеры (единичный кватернион)
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0

        self.tf_broadcaster.sendTransform(t)

        self.get_logger().debug(
            f"Published 3D & TF: ({x_m:.4f}, {y_m:.4f}, {z_m:.4f}) "
            f"in {self.frame_id} -> {self.target_frame_id}"
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
