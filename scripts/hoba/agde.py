#!/usr/bin/env python3
import cv2 as cv
import numpy as np
from ultralytics import YOLO

# ===== config =====
MODEL = "/home/ben/take/ur10-assistant/hoba/best_fixed.pt"
CAM_INDEX = 0
IMGZ = 640
PAD_REL = 0.07
MIN_PAD = 8

# Camera intrinsics (pixels). Fill from calibration.
FX = 1100.0
FY = 1100.0
CX = None   # if None, use image center
CY = None

# Camera height above ground plane (meters)
H_CAM_M = 0.35
# ==================

def expand_box(x1, y1, x2, y2, W, H, pad_rel=PAD_REL, min_pad=MIN_PAD):
    w = x2 - x1
    h = y2 - y1
    px = max(min_pad, int(w * pad_rel))
    py = max(min_pad, int(h * pad_rel))
    x1 = max(0, x1 - px)
    y1 = max(0, y1 - py)
    x2 = min(W - 1, x2 + px)
    y2 = min(H - 1, y2 + py)
    return x1, y1, x2, y2

def remove_shadows_white_bg_auto(bgr_roi, k_div_frac=0.03):
    h, w = bgr_roi.shape[:2]
    hsv = cv.cvtColor(bgr_roi, cv.COLOR_BGR2HSV)
    S, V = hsv[..., 1], hsv[..., 2]

    b = max(6, int(0.1 * min(h, w)))
    bmask = np.zeros((h, w), dtype=bool)
    bmask[:b, :] = True; bmask[-b:, :] = True; bmask[:, :b] = True; bmask[:, -b:] = True
    s_thr = int(np.clip(np.percentile(S[bmask], 90) + 5, 40, 120))
    v_thr = int(np.clip(np.percentile(V[bmask], 60), 170, 245))

    gray = cv.cvtColor(bgr_roi, cv.COLOR_BGR2GRAY)
    k_div = max(31, (int(min(h, w) * k_div_frac) // 2) * 2 + 1)
    bg = cv.medianBlur(gray, k_div)
    norm = cv.divide(gray, bg, scale=255)
    norm = cv.GaussianBlur(norm, (0, 0), 1.0)

    _, bin_otsu = cv.threshold(norm, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    obj_int = (bin_otsu == 0)

    paper = (S < s_thr) & (V > v_thr)
    obj_color = (S > s_thr + 10) | (V < v_thr - 40)

    mask = ((obj_int | obj_color) & (~paper)).astype(np.uint8) * 255

    k3 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    k5 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, k5)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, k3)

    num, labels, stats, _ = cv.connectedComponentsWithStats(mask, connectivity=8)
    if num > 1:
        largest = 1 + np.argmax(stats[1:, cv.CC_STAT_AREA])
        mask = np.where(labels == largest, 255, 0).astype(np.uint8)
    return mask

def com_of_mask(mask: np.ndarray):
    if mask.ndim == 3:
        mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    _, binm = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)
    M = cv.moments(binm, binaryImage=True)
    if M["m00"] == 0:
        return None
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    return float(cx), float(cy)

def com_of_largest_component(mask: np.ndarray):
    if mask.ndim == 3:
        mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    _, binm = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)
    num, labels, stats, centroids = cv.connectedComponentsWithStats(binm, connectivity=8)
    if num <= 1:
        return None
    largest = 1 + np.argmax(stats[1:, cv.CC_STAT_AREA])
    cx, cy = centroids[largest]
    return float(cx), float(cy)

def snap_point_to_mask(cx: float, cy: float, mask_u8: np.ndarray, prefer_interior: bool = True, erode_iters: int = 1):
    if mask_u8.ndim == 3:
        mask_u8 = cv.cvtColor(mask_u8, cv.COLOR_BGR2GRAY)
    _, binm = cv.threshold(mask_u8, 127, 255, cv.THRESH_BINARY)
    h, w = binm.shape[:2]
    ix, iy = int(round(cx)), int(round(cy))
    ix = np.clip(ix, 0, w - 1); iy = np.clip(iy, 0, h - 1)
    if binm[iy, ix] > 0:
        return float(ix), float(iy)
    cand = binm.copy()
    if prefer_interior:
        k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        eroded = cv.erode(cand, k, iterations=erode_iters)
        if cv.countNonZero(eroded) > 0:
            cand = eroded
    ys, xs = np.nonzero(cand)
    if xs.size == 0:
        ys, xs = np.nonzero(binm)
        if xs.size == 0:
            return None
    pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    d2 = (pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2
    j = int(np.argmin(d2))
    return float(pts[j, 0]), float(pts[j, 1])

# pixel -> ground plane (meters). Camera is perpendicular to plane.
def pixel_to_ground(u_px: float, v_px: float, fx: float, fy: float, cx: float, cy: float, H_m: float):
    X = (u_px - cx) / fx * H_m   # +X right
    Y = (v_px - cy) / fy * H_m   # +Y forward (image-down)
    return float(X), float(Y)

def draw_centroid(img_bgr: np.ndarray, c, color=(0, 0, 255), label="COM"):
    out = img_bgr.copy()
    if c is None:
        return out
    x, y = int(round(c[0])), int(round(c[1]))
    cv.drawMarker(out, (x, y), color, markerType=cv.MARKER_CROSS, markerSize=20, thickness=2)
    cv.putText(out, f"{label} ({x},{y})", (x + 8, y - 8),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv.LINE_AA)
    return out

def main():
    cap = cv.VideoCapture(CAM_INDEX, cv.CAP_V4L2)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("Camera read failed"); return
    H, W = frame.shape[:2]

    cx_intr = CX if CX is not None else (W - 1) / 2.0
    cy_intr = CY if CY is not None else (H - 1) / 2.0

    model = YOLO(MODEL)
    res = model.predict(source=frame, imgsz=IMGZ, device="cpu", conf=0.25, verbose=False)[0]
    if res.boxes is None or res.boxes.xyxy.numel() == 0:
        print("No detections"); cv.imshow("frame", frame); cv.waitKey(0); return

    idx = int(res.boxes.conf.argmax().item())
    box = res.boxes.xyxy[idx].detach().cpu().numpy().astype(int)
    cls = int(res.boxes.cls[idx].item())
    name = model.names.get(cls, str(cls))

    x1, y1, x2, y2 = expand_box(*box, W, H)
    if x2 <= x1 or y2 <= y1:
        print("Invalid bbox after expand"); cv.imshow("frame", frame); cv.waitKey(0); return

    roi = frame[y1:y2, x1:x2].copy()

    mask = remove_shadows_white_bg_auto(roi)
    com_roi = com_of_largest_component(mask) or com_of_mask(mask)
    com_roi = snap_point_to_mask(*com_roi, mask) if com_roi else None
    com_img = (x1 + com_roi[0], y1 + com_roi[1]) if com_roi else None

    vis = frame.copy()
    cv.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv.putText(vis, f"{name}", (x1, max(0, y1 - 6)), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv.LINE_AA)
    vis = draw_centroid(vis, com_img, color=(0, 0, 255), label="COM")

    if com_img is not None:
        # pixel -> meters on ground plane
        X_m, Y_m = pixel_to_ground(com_img[0], com_img[1], FX, FY, cx_intr, cy_intr, H_CAM_M)

        # print translation (SE2 with zero rotation)
        print(f"COM_world: X={X_m:.4f} m, Y={Y_m:.4f} m")
        print("T_origin_to_COM (SE2 homogeneous):")
        T = np.array([[1.0, 0.0, X_m],
                      [0.0, 1.0, Y_m],
                      [0.0, 0.0, 1.0]], dtype=np.float32)
        np.set_printoptions(precision=4, suppress=True)
        print(T)

        # overlay text
        ytxt = max(0, y1 - 10)
        for s in (f"X={X_m:.3f} m", f"Y={Y_m:.3f} m"):
            cv.putText(vis, s, (x1, ytxt), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv.LINE_AA)
            cv.putText(vis, s, (x1, ytxt), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv.LINE_AA)
            ytxt += 22
    else:
        print("Centroid not found")

    cv.imshow("frame + COM + world XY", vis)
    cv.imshow("roi", roi)
    cv.imshow("mask_shadowless_roi", mask)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
