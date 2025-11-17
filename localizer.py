#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2 as cv
import numpy as np
from ultralytics import YOLO
from typing import Optional, Tuple, Dict
import time

# ==== CONFIG ====
MODEL = "/home/ben/take/ur10-assistant/hoba/best_fixed.pt"
CAM_INDEX = 0
IMGZ = 640

# Intrinsics @ 1280x960
FX, FY = 1413.5209083841803, 1406.0718500218677
CX, CY = 666.23745047219757, 483.9875748250534
CAP_W, CAP_H = 1280, 960

# Distortion (k1,k2,p1,p2,k3)
DIST = np.array([[
    -0.0016524370997080804,
     1.3223825867667165,
     0.0049886772832232638,
    -0.000404770364949342,
    -5.0863844544987762
]], dtype=np.float64).T

K = np.array([[FX, 0.0, CX],
              [0.0, FY, CY],
              [0.0, 0.0, 1.0]], dtype=np.float64)

# Default plane distance (camera ⟂ plane)
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
    return float(und[0,0,0]), float(und[0,0,1])

def pixel_to_xy_known_depth(u_px: float, v_px: float, Z_m: float,
                            K: np.ndarray, dist: np.ndarray) -> Tuple[float,float]:
    u_u, v_u = undistort_pixel(u_px, v_px, K, dist)
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    X = (u_u - cx) / fx * Z_m
    Y = (v_u - cy) / fy * Z_m
    
    return float(X), float(Y)

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
    b *= m / max(b.mean(), 1e-6); g *= m / max(g.mean(), 1e-6); r *= m / max(r.mean(), 1e-6)
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

def _annotate_debug(vis: np.ndarray,
                    K: np.ndarray,
                    Z_used_m: float,
                    bbox: Optional[Tuple[int,int,int,int]] = None,
                    raw: Optional[Tuple[float,float]] = None,
                    und: Optional[Tuple[float,float]] = None,
                    XY: Optional[Tuple[float,float]] = None,
                    stage: Optional[str] = None) -> np.ndarray:
    h, w = vis.shape[:2]
    fx, fy, cx, cy = float(K[0,0]), float(K[1,1]), float(K[0,2]), float(K[1,2])
    font = cv.FONT_HERSHEY_SIMPLEX
    out = vis.copy()

    if bbox is not None:
        x1,y1,x2,y2 = map(int, bbox)
        cv.rectangle(out, (x1,y1), (x2,y2), (0,255,0), 2)

    cv.putText(out, "img (0,0)", (8,18), font, 0.6, (0,255,255), 2, cv.LINE_AA)

    cpt = (int(round(cx)), int(round(cy)))
    cv.circle(out, cpt, 6, (255,0,0), -1)
    cv.putText(out, f"origin cx,cy=({int(cx)},{int(cy)})", (cpt[0]+8, cpt[1]-8),
               font, 0.6, (255,0,0), 2, cv.LINE_AA)

    ax_len = max(40, min(w,h)//8)
    cv.arrowedLine(out, cpt, (cpt[0]+ax_len, cpt[1]), (0,200,0), 2, tipLength=0.15)
    cv.arrowedLine(out, cpt, (cpt[0], cpt[1]+ax_len), (0,0,200), 2, tipLength=0.15)
    cv.putText(out, "+X", (cpt[0]+ax_len+6, cpt[1]+4), font, 0.6, (0,200,0), 2, cv.LINE_AA)
    cv.putText(out, "+Y", (cpt[0]+4, cpt[1]+ax_len+16), font, 0.6, (0,0,200), 2, cv.LINE_AA)

    # scale bar for 0.10 m at current Z
    px_per_0p10m = int(round((fx / Z_used_m) * 0.10))
    xb, yb = 40, h - 30
    cv.line(out, (xb, yb), (xb + px_per_0p10m, yb), (255,255,255), 3)
    cv.putText(out, "0.10 m", (xb + px_per_0p10m + 8, yb + 6), font, 0.6, (255,255,255), 2, cv.LINE_AA)

    if raw is not None:
        cv.circle(out, (int(raw[0]), int(raw[1])), 6, (0,255,255), 2)
    if und is not None:
        cv.circle(out, (int(und[0]), int(und[1])), 6, (0,0,255), -1)
        cv.line(out, cpt, (int(und[0]), int(und[1])), (0,0,255), 1)

    lines = []
    if stage is not None:
        lines.append(f"stage: {stage}")
    if raw is not None:
        lines.append(f"raw px: ({raw[0]:.1f}, {raw[1]:.1f})")
    if und is not None:
        lines.append(f"undist px: ({und[0]:.1f}, {und[1]:.1f})")
    lines.append(f"Z used: {Z_used_m:.3f} m")
    if XY is not None:
        lines.append(f"ground XY: ({XY[0]:.3f} m, {XY[1]:.3f} m)")
    lines.append("axes: +X right, +Y down; origin=principal point")

    x0, y0 = 10, 40
    for i, t in enumerate(lines):
        y = y0 + i*22
        cv.putText(out, t, (x0, y), font, 0.6, (0,0,0), 3, cv.LINE_AA)
        cv.putText(out, t, (x0, y), font, 0.6, (255,255,255), 1, cv.LINE_AA)

    return out

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
    return_debug: bool = False,
    z_override_m: Optional[float] = None,  # if given, use this depth instead of H_CAM_M
    coord_origin: str = "principal",       # "principal" или "center"
):
    dbg: Dict[str, object] = {"stage": "start"}

    # --- Кадр ---
    frame = _grab_stable_frame(
        CAM_INDEX,
        warmup_s=warmup_s, prefer_sharpest=prefer_sharpest
    )
    if frame is None:
        dbg["stage"] = "no_frame"
        return (None, dbg) if return_debug else None
    dbg["frame"] = frame

    Himg, Wimg = frame.shape[:2]
    dbg["frame_size"] = (int(Wimg), int(Himg))

    # --- Детекция YOLO ---
    model = _get_model()
    res = model.predict(source=frame, imgsz=IMGZ, device=device, conf=0.25, verbose=False)[0]
    if res.boxes is None or res.boxes.xyxy.numel() == 0:
        dbg["stage"] = "no_detection"
        return (None, dbg) if return_debug else None

    idx = int(res.boxes.conf.argmax().item())
    x1, y1, x2, y2 = res.boxes.xyxy[idx].detach().cpu().numpy().astype(int)
    x1, y1, x2, y2 = _expand_box(x1, y1, x2, y2, Wimg, Himg)
    if x2 <= x1 or y2 <= y1:
        dbg["stage"] = "invalid_bbox"
        dbg["bbox"]  = (int(x1), int(y1), int(x2), int(y2))
        return (None, dbg) if return_debug else None
    dbg["bbox"] = (int(x1), int(y1), int(x2), int(y2))

    roi = frame[y1:y2, x1:x2].copy()
    dbg["roi"] = roi

    # --- Маска / COM ---
    mask, thr = remove_shadows_white_bg_auto(
        roi,
        k_div_frac=k_div_frac, s_p=s_p, v_p=v_p, s_off=s_off,
        s_clip=s_clip, v_clip=v_clip,
        morph_close=morph_close, morph_open=morph_open,
        use_grayworld=use_grayworld
    )
    dbg["mask"] = mask
    dbg["thresholds"] = thr

    c = _com(mask)
    if c is None:
        dbg["stage"] = "no_com"
        return (None, dbg) if return_debug else None
    c = _snap_inside(*c, mask)
    if c is None:
        dbg["stage"] = "snap_failed"
        return (None, dbg) if return_debug else None

    # COM в координатах исходного изображения
    u_img, v_img = x1 + c[0], y1 + c[1]
    dbg["com_img_raw"] = (float(u_img), float(v_img))

    # --- Глубина и XY в системе "principal point" ---
    Z_used = float(z_override_m) if z_override_m is not None else float(H_CAM_M)
    dbg["Z_used"] = Z_used

    # Классическая модель камеры
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    # Undistort COM
    u_und, v_und = undistort_pixel(u_img, v_img, K, DIST)
    dbg["com_img_undist"] = (float(u_und), float(v_und))

    # XY_cam: ноль в principal point
    X_cam = (u_und - cx) / fx * Z_used
    Y_cam = (v_und - cy) / fy * Z_used
    dbg["XY_cam"] = (float(X_cam), float(Y_cam))

    # --- XY_center: ноль в центре изображения ---
    # Позиция центра кадра (W/2, H/2) в системе XY_cam:
    u_c = Wimg / 2.0
    v_c = Himg / 2.0
    # Внимание: центр кадра мы не undistort-им, потому что работаем в той же
    # системе, где undistortPoints выдаёт пиксели с матрицей K.
    # Для небольших дисторсий это достаточно, а точный вариант — отдельно
    # undistort-ить (u_c, v_c) таким же способом.
    u_c_und, v_c_und = undistort_pixel(u_c, v_c, K, DIST)

    X_center_cam = (u_c_und - cx) / fx * Z_used
    Y_center_cam = (v_c_und - cy) / fy * Z_used
    dbg["center_offset_cam"] = (float(X_center_cam), float(Y_center_cam))

    # Теперь координаты COM относительно центра кадра:
    X_center_rel = X_cam - X_center_cam
    Y_center_rel = Y_cam - Y_center_cam
    dbg["XY_center"] = (float(X_center_rel), float(Y_center_rel))

    dbg["stage"] = "ok"

    # --- выбор, что возвращать наружу ---
    if coord_origin == "center":
        X_out, Y_out = X_center_rel, Y_center_rel
    else:  # "principal" или что-то левое — по умолчанию principal
        X_out, Y_out = X_cam, Y_cam

    if display:
        vis = _annotate_debug(
            frame, K, Z_used,
            bbox=(x1, y1, x2, y2),
            raw=(u_img, v_img),
            und=(u_und, v_und),
            XY=(X_out, Y_out),
            stage=dbg["stage"],
        )
        cv.imshow("original(+debug)", vis)
        cv.imshow("roi", roi)
        cv.imshow("mask_final", mask)
        cv.waitKey(1)

    if not return_debug:
        return (float(X_out), float(Y_out))
    else:
        return ((float(X_out), float(Y_out)), dbg)


# ---------- CLI ----------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Measure COM XY in meters from a single frame.")
    ap.add_argument("--display", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--warmup", type=float, default=1.5)
    ap.add_argument("--no-sharpest", action="store_true")
    ap.add_argument("--s_p", type=int, default=90)
    ap.add_argument("--v_p", type=int, default=60)
    ap.add_argument("--s_off", type=int, default=5)
    ap.add_argument("--s_clip", type=int, nargs=2, default=(20, 140))
    ap.add_argument("--v_clip", type=int, nargs=2, default=(160, 245))
    ap.add_argument("--morph_close", type=int, default=5)
    ap.add_argument("--morph_open", type=int, default=3)
    ap.add_argument("--k_div_frac", type=float, default=0.03)
    ap.add_argument("--grayworld", action="store_true")
    ap.add_argument("--depth", type=float, default=None, help="Override depth Z [m]. If unset, uses H_CAM_M.")
    ap.add_argument(
    "--coord-origin",
    choices=["principal", "center"],
    default="principal",
    help="Система координат на выходе: principal (ноль в cx,cy) или center (ноль в центре кадра).",)

    args = ap.parse_args()

    out = measure_xy_once(
        display=args.display, device=args.device,
        warmup_s=args.warmup, prefer_sharpest=(not args.no_sharpest),
        s_p=args.s_p, v_p=args.v_p, s_off=args.s_off,
        s_clip=tuple(args.s_clip), v_clip=tuple(args.v_clip),
        morph_close=args.morph_close, morph_open=args.morph_open,
        k_div_frac=args.k_div_frac, use_grayworld=args.grayworld,
        return_debug=args.debug, z_override_m=args.depth,
        coord_origin=args.coord_origin,
    )

    if args.debug:
        coords, dbg = out if (isinstance(out, tuple) and isinstance(out[1], dict)) else (None, {"stage":"unknown"})
        stage = dbg.get("stage", "unknown")
        frame = dbg.get("frame"); roi = dbg.get("roi"); mask = dbg.get("mask")
        bbox  = dbg.get("bbox");  raw = dbg.get("com_img_raw"); und  = dbg.get("com_img_undist")
        Z_used = dbg.get("Z_used", (args.depth if args.depth is not None else H_CAM_M))

        xy_cam    = dbg.get("XY_cam")
        xy_center = dbg.get("XY_center")

        if frame is not None:
            vis = _annotate_debug(frame, K, Z_used, bbox=bbox, raw=raw, und=und,
                                XY=(coords if coords is not None else None),
                                stage=stage)
            cv.imshow("original(+debug)", vis)
        if roi is not None:
            cv.imshow("roi", roi)
        if mask is not None:
            cv.imshow("mask_final", mask)

        if coords is not None:
            x_m, y_m = coords
            print(f"OUT ({args.coord_origin}): {x_m:.6f} {y_m:.6f}  |  Z_used={Z_used:.3f}  stage={stage} thr={dbg.get('thresholds')}")
            if xy_cam is not None:
                print(f"  cam-origin XY:   {xy_cam[0]:.6f} {xy_cam[1]:.6f}")
            if xy_center is not None:
                print(f"  center-origin XY:{xy_center[0]:.6f} {xy_center[1]:.6f}")
        else:
            print(f"FAIL  |  Z_used={Z_used:.3f}  stage={stage} thr={dbg.get('thresholds')}")
        cv.waitKey(0); cv.destroyAllWindows()
