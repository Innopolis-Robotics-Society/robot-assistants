#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
localizer.py
YOLO crop -> shadow-robust mask (tunable) -> COM inside mask -> pixel->ground XY (meters).
"""

import cv2 as cv
import numpy as np
from ultralytics import YOLO
from typing import Optional, Tuple, Dict
import time, math, subprocess, shutil, os

# ========= CONFIG =========
MODEL = "/home/ben/manip/best_fixed.pt"
CAM_INDEX = 0
IMGZ = 640

# Camera intrinsics (pixels)
FX = 1100.0
FY = 1100.0
CX: Optional[float] = None
CY: Optional[float] = None

# Camera height above the ground plane (m). Camera axis ⟂ plane.
H_CAM_M = 0.75

# BBox expansion
PAD_REL = 0.07
MIN_PAD = 8
# =========================

_MODEL: Optional[YOLO] = None

def _get_model() -> YOLO:
    global _MODEL
    if _MODEL is None:
        _MODEL = YOLO(MODEL)
    return _MODEL

# ---------- camera warmup / autofocus lock ----------

def _v4l2_devnode(cam_index: int) -> str:
    return f"/dev/video{cam_index}"

def _try_lock_autofocus(cap: cv.VideoCapture, cam_index: int, focus_abs: Optional[int]) -> None:
    dev = _v4l2_devnode(cam_index)
    if shutil.which("v4l2-ctl") and os.path.exists(dev):
        try:
            subprocess.run(["v4l2-ctl", "-d", dev, "--set-ctrl=focus_auto=0"], check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if focus_abs is not None:
                subprocess.run(["v4l2-ctl", "-d", dev, f"--set-ctrl=focus_absolute={int(focus_abs)}"], check=False,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except Exception:
            pass
    # Fallback: OpenCV props (may not work on all cams)
    try:
        cap.set(getattr(cv, "CAP_PROP_AUTOFOCUS", 39), 0)
        if focus_abs is not None:
            cap.set(getattr(cv, "CAP_PROP_FOCUS", 28), float(focus_abs))
    except Exception:
        pass

def _lap_var(gray: np.ndarray) -> float:
    return float(cv.Laplacian(gray, cv.CV_64F).var())

def _grab_stable_frame(
    cam_index: int,
    width: int = 1280,
    height: int = 720,
    warmup_s: float = 1.5,
    prefer_sharpest: bool = True,
    lock_af: bool = True,
    focus_abs: Optional[int] = None
) -> Optional[np.ndarray]:
    cap = cv.VideoCapture(cam_index, cv.CAP_V4L2)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)

    if lock_af:
        _try_lock_autofocus(cap, cam_index, focus_abs)

    t0 = time.time()
    best_var = -1.0
    best = None
    last = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        last = frame
        if prefer_sharpest:
            v = _lap_var(cv.cvtColor(frame, cv.COLOR_BGR2GRAY))
            if v > best_var:
                best_var = v
                best = frame.copy()
        if time.time() - t0 >= warmup_s:
            break

    cap.release()
    if last is None:
        return None
    return best if (prefer_sharpest and best is not None) else last

# ---------- geometry/utils ----------

def _expand_box(x1: int, y1: int, x2: int, y2: int, W: int, H: int,
                pad_rel: float = PAD_REL, min_pad: int = MIN_PAD) -> Tuple[int, int, int, int]:
    w, h = x2 - x1, y2 - y1
    px = max(min_pad, int(w * pad_rel))
    py = max(min_pad, int(h * pad_rel))
    x1 = max(0, x1 - px); y1 = max(0, y1 - py)
    x2 = min(W - 1, x2 + px); y2 = min(H - 1, y2 + py)
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
) -> Tuple[np.ndarray, Dict[str, float]]:
    bgr = _gray_world(bgr_roi) if use_grayworld else bgr_roi
    h, w = bgr.shape[:2]
    hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
    S, V = hsv[..., 1], hsv[..., 2]

    b = max(6, int(0.1 * min(h, w)))
    bm = np.zeros((h, w), dtype=bool)
    bm[:b, :] = True; bm[-b:, :] = True; bm[:, :b] = True; bm[:, -b:] = True

    s_thr = int(np.clip(np.percentile(S[bm], s_p) + s_off, *s_clip))
    v_thr = int(np.clip(np.percentile(V[bm], v_p), *v_clip))

    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    k = max(31, (int(min(h, w) * k_div_frac) // 2) * 2 + 1)
    bg = cv.medianBlur(gray, k)
    norm = cv.GaussianBlur(cv.divide(gray, bg, scale=255), (0, 0), 1.0)

    _, bin_otsu = cv.threshold(norm, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    obj_int = (bin_otsu == 0)
    paper = (S < s_thr) & (V > v_thr)
    obj_color = (S > s_thr + 10) | (V < v_thr - 40)

    mask = ((obj_int | obj_color) & (~paper)).astype(np.uint8) * 255

    kC = cv.getStructuringElement(cv.MORPH_ELLIPSE, (morph_close, morph_close))
    kO = cv.getStructuringElement(cv.MORPH_ELLIPSE, (morph_open, morph_open))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kC)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kO)

    num, labels, stats, _ = cv.connectedComponentsWithStats(mask, 8)
    if num > 1:
        idx = 1 + np.argmax(stats[1:, cv.CC_STAT_AREA])
        mask = np.where(labels == idx, 255, 0).astype(np.uint8)

    return mask, {"s_thr": s_thr, "v_thr": v_thr, "k_div": k}

def _com(mask_u8: np.ndarray) -> Optional[Tuple[float, float]]:
    _, binm = cv.threshold(mask_u8, 127, 255, cv.THRESH_BINARY)
    M = cv.moments(binm, binaryImage=True)
    if M["m00"] == 0:
        return None
    return M["m10"]/M["m00"], M["m01"]/M["m00"]

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

def _pixel_to_ground(u_px: float, v_px: float, fx: float, fy: float, cx: float, cy: float, H_m: float) -> Tuple[float, float]:
    # Camera perpendicular to plane. Origin at optical axis hit point.
    X = (u_px - cx) / fx * H_m  # +X right
    Y = (v_px - cy) / fy * H_m  # +Y forward (image-down)
    return float(X), float(Y)

def _draw_centroid(img: np.ndarray, pt_img: Tuple[float, float], color=(0, 0, 255), label="COM") -> np.ndarray:
    out = img.copy()
    x, y = int(round(pt_img[0])), int(round(pt_img[1]))
    cv.drawMarker(out, (x, y), color, markerType=cv.MARKER_CROSS, markerSize=20, thickness=2)
    cv.putText(out, f"{label} ({x},{y})", (x + 8, y - 8), cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv.LINE_AA)
    return out

# ---------- main API ----------

def measure_xy_once(
    *,
    display: bool = False,
    device: str = "cpu",
    # capture controls
    warmup_s: float = 1.5,
    prefer_sharpest: bool = True,
    lock_af: bool = True,
    focus_abs: Optional[int] = None,
    # mask tunables
    s_p: int = 90, v_p: int = 60, s_off: int = 5,
    s_clip: Tuple[int, int] = (20, 140),
    v_clip: Tuple[int, int] = (160, 245),
    morph_close: int = 5, morph_open: int = 3,
    k_div_frac: float = 0.03,
    use_grayworld: bool = False,
    # debug return
    return_debug: bool = False
) -> Optional[Tuple[float, float] | Tuple[Tuple[float, float], Dict[str, object]]]:
    """
    One-shot measurement.
    Returns:
      (X_m, Y_m)          if return_debug=False
      ((X_m, Y_m), debug) if return_debug=True, where debug has:
        frame, roi, mask, bbox, com_img, com_roi, thresholds(dict)
    """
    frame = _grab_stable_frame(
        CAM_INDEX, 1280, 720,
        warmup_s=warmup_s,
        prefer_sharpest=prefer_sharpest,
        lock_af=lock_af,
        focus_abs=focus_abs
    )
    if frame is None:
        return None

    Himg, Wimg = frame.shape[:2]
    cx_intr = CX if CX is not None else (Wimg - 1) / 2.0
    cy_intr = CY if CY is not None else (Himg - 1) / 2.0

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

    mask, dbg = remove_shadows_white_bg_auto(
        roi,
        k_div_frac=k_div_frac,
        s_p=s_p, v_p=v_p, s_off=s_off,
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
    X_m, Y_m = _pixel_to_ground(u_img, v_img, FX, FY, cx_intr, cy_intr, H_CAM_M)

    if display:
        vis = frame.copy()
        cv.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        vis = _draw_centroid(vis, (u_img, v_img), (0, 0, 255), "COM")
        cv.imshow("original+COM", vis)
        cv.imshow("roi", roi)
        cv.imshow("mask_final", mask)
        cv.waitKey(1)

    if not return_debug:
        return float(X_m), float(Y_m)

    debug = {
        "frame": frame,
        "roi": roi,
        "mask": mask,
        "bbox": (x1, y1, x2, y2),
        "com_img": (u_img, v_img),
        "com_roi": (float(c[0]), float(c[1])),
        "thresholds": dbg
    }
    return (float(X_m), float(Y_m)), debug

# ---------- CLI ----------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Measure COM XY in meters from a single frame.")
    ap.add_argument("--display", action="store_true", help="Show quick preview windows during measure.")
    ap.add_argument("--debug", action="store_true", help="After measuring, show original, ROI, and mask with COM and wait for key.")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--warmup", type=float, default=1.5)
    ap.add_argument("--no-sharpest", action="store_true")
    ap.add_argument("--no-lock-af", action="store_true")
    ap.add_argument("--focus", type=int, default=None)
    ap.add_argument("--s_p", type=int, default=90)
    ap.add_argument("--v_p", type=int, default=60)
    ap.add_argument("--s_off", type=int, default=5)
    ap.add_argument("--s_clip", type=int, nargs=2, default=(20, 140))
    ap.add_argument("--v_clip", type=int, nargs=2, default=(160, 245))
    ap.add_argument("--morph_close", type=int, default=5)
    ap.add_argument("--morph_open", type=int, default=3)
    ap.add_argument("--k_div_frac", type=float, default=0.03)
    ap.add_argument("--grayworld", action="store_true")
    args = ap.parse_args()

    out = measure_xy_once(
        display=args.display, device=args.device,
        warmup_s=args.warmup,
        prefer_sharpest=(not args.no_sharpest),
        lock_af=(not args.no_lock_af),
        focus_abs=args.focus,
        s_p=args.s_p, v_p=args.v_p, s_off=args.s_off,
        s_clip=tuple(args.s_clip), v_clip=tuple(args.v_clip),
        morph_close=args.morph_close, morph_open=args.morph_open,
        k_div_frac=args.k_div_frac, use_grayworld=args.grayworld,
        return_debug=args.debug
    )

    if out is None:
        print("FAIL")
    else:
        if args.debug:
            (x_m, y_m), dbg = out
            print(f"{x_m:.6f} {y_m:.6f}  |  thr={dbg['thresholds']}")
            # make full debug windows that wait for key
            frame = dbg["frame"]
            roi = dbg["roi"]
            mask = dbg["mask"]
            (x1, y1, x2, y2) = dbg["bbox"]
            u_img, v_img = dbg["com_img"]

            vis = frame.copy()
            cv.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            vis = _draw_centroid(vis, (u_img, v_img), (0, 0, 255), "COM")
            txt = f"X={x_m:.3f}m Y={y_m:.3f}m  S<{dbg['thresholds']['s_thr']} V>{dbg['thresholds']['v_thr']} k={dbg['thresholds']['k_div']}"
            cv.putText(vis, txt, (10, 28), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv.LINE_AA)
            cv.putText(vis, txt, (10, 28), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv.LINE_AA)

            cv.imshow("original+COM", vis)
            cv.imshow("roi", roi)
            cv.imshow("mask_final", mask)
            cv.waitKey(0)
            cv.destroyAllWindows()
        else:
            x_m, y_m = out
            print(f"{x_m:.6f} {y_m:.6f}")
