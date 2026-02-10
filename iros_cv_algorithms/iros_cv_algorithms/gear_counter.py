# gear_counter.py
from __future__ import annotations

import math
import itertools
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class GearCounterConfig:
    # Preprocess
    brighten_region: Tuple[int, int, int, int] =  None# (x1,y1,x2,y2)
    brighten_factor: float = 1.2
    brighten_blend: bool = True
    bilateral_d: int = 3
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0

    # Hough small circles (gear centers)
    hough_dp: float = 1.5
    hough_min_dist: float = 100.0
    hough_param1: float = 60.0
    hough_param2: float = 50.0
    hough_min_radius: int = 5
    hough_max_radius: int = 25

    # Improve Hough stability
    use_clahe_for_hough: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: int = 8

    # Filter: drop isolated centers
    max_neighbor_dist: float = 200.0

    # Validation: reject overlapping/contained center circles
    center_circle_min_clearance: float = 0.0

    # Sorting
    sort_y_tol: int = 25

    # Stable ordering across frames (uses first good detection as reference)
    stable_order: bool = True
    stable_order_max_dist: float = 80.0
    stable_order_reset: bool = False

    # Radius search for big/small circle (edge-based)
    big_r_min: int = 35
    big_r_max: int = 300
    small_r_min: int = 10
    small_r_max: int = 35

    # FIX: limit r_max by nearest neighbor distance (prevents too-big radii)
    big_r_max_from_neighbor_frac: float = 0.45
    small_r_max_from_neighbor_frac: float = 0.25

    radius_step: int = 1
    radius_thickness: int = 3
    radius_angles: int = 720
    canny1: int = 40
    canny2: int = 120
    blur_ksize: int = 5
    refine_subpixel: bool = True

    # Optional penalty to avoid sticking to large radii (0 = off)
    radius_size_penalty: float = 0.0

    # Teeth formula
    teeth_k1: float = 7.0
    teeth_k2_num: float = 1.15
    teeth_k2_den: float = 1.8


@dataclass
class GearCountResult:
    ok: bool
    reason: str
    teeth: List[float]
    centers: List[Tuple[int, int]]
    r_big: List[float]
    r_small: List[float]
    preprocessed_bgr: np.ndarray
    annotated_bgr: np.ndarray
    debug: Dict[str, Any]


def brighten_region(image: np.ndarray, region, factor: float, blend: bool = True) -> np.ndarray:
    if isinstance(region, tuple) and len(region) == 4:
        x1, y1, x2, y2 = region
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(int(x2), image.shape[1]), min(int(y2), image.shape[0])

        if blend:
            mask = np.zeros(image.shape[:2], dtype=np.float32)
            mask[y1:y2, x1:x2] = 1.0
            kernel_size = max(3, min(x2 - x1, y2 - y1) // 10)
            kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
            mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
        else:
            mask = np.zeros(image.shape[:2], dtype=np.float32)
            mask[y1:y2, x1:x2] = 1.0
    else:
        mask = region.astype(np.float32) / 255.0

    brightened = cv2.convertScaleAbs(image, alpha=factor, beta=0)

    result = image.astype(np.float32)
    brightened = brightened.astype(np.float32)
    for c in range(3):
        result[:, :, c] = result[:, :, c] * (1 - mask) + brightened[:, :, c] * mask
    return np.clip(result, 0, 255).astype(np.uint8)


def blur(img: np.ndarray, d: int = 7, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
    return cv2.bilateralFilter(img, int(d), float(sigma_color), float(sigma_space))


def filter_circles_by_max_distance(circles_array, max_allowed_distance: float):
    """Remove circles whose nearest neighbor is farther than max_allowed_distance."""
    if circles_array is None or len(circles_array[0]) <= 1:
        return circles_array

    circles = circles_array[0].copy()
    filtered = []

    for i, (x1, y1, r1) in enumerate(circles):
        min_dist = float("inf")
        for j, (x2, y2, _r2) in enumerate(circles):
            if i == j:
                continue
            dist = math.hypot(float(x2 - x1), float(y2 - y1))
            if dist < min_dist:
                min_dist = dist
        if min_dist <= max_allowed_distance:
            filtered.append([x1, y1, r1])

    return np.array([filtered]) if filtered else None


def sort_circles(circles, y_tol: int = 25) -> np.ndarray:
    """
    Return circles as (N,3) sorted roughly top-left to bottom-right.
    Accepts: None, (1,N,3), (N,3), (3,)
    """
    if circles is None:
        return np.empty((0, 3), dtype=np.float32)

    c = np.asarray(circles)

    if c.ndim == 3 and c.shape[-1] == 3:
        c = c[0]
    elif c.ndim == 1 and c.size == 3:
        c = c.reshape(1, 3)
    elif c.ndim == 2 and c.shape[1] == 3:
        pass
    else:
        raise ValueError(f"Unexpected circles shape: {c.shape}")

    if c.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    x = np.rint(c[:, 0]).astype(np.int32)
    y = np.rint(c[:, 1]).astype(np.int32)
    r = c[:, 2].astype(np.float32)

    row = (y // max(1, int(y_tol))).astype(np.int32)
    order = np.lexsort((x, y, row))
    return np.stack([x[order], y[order], r[order]], axis=1).astype(np.float32)


# --- Stable ordering across frames (module-level cache) ---
_REF_CENTER_ORDER: Optional[List[Tuple[int, int]]] = None


def _reorder_circles_by_reference(
    circles_xy_r: np.ndarray,
    ref_centers: List[Tuple[int, int]],
    *,
    max_dist_px: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Reorder circles (N,3) to best match ref_centers (N,2) by minimal total distance.

    Uses brute-force permutations; for N=4 it is only 24 permutations.
    Returns (reordered_circles, debug_dict). If no assignment within max_dist_px,
    returns input circles unchanged with debug reason.
    """
    c = np.asarray(circles_xy_r, dtype=np.float32)
    if c.ndim != 2 or c.shape[1] != 3 or len(ref_centers) != int(c.shape[0]):
        return c, {"ok": False, "reason": "bad_shape_or_len", "shape": list(c.shape), "n_ref": len(ref_centers)}

    ref = np.asarray(ref_centers, dtype=np.float32)
    cur = c[:, :2]

    best_perm = None
    best_cost = float("inf")
    best_d = None

    idxs = list(range(int(c.shape[0])))
    for perm in itertools.permutations(idxs):
        perm = list(perm)
        d = np.linalg.norm(cur[perm] - ref, axis=1)
        if float(np.max(d)) > float(max_dist_px):
            continue
        cost = float(np.sum(d))
        if cost < best_cost:
            best_cost = cost
            best_perm = perm
            best_d = d

    if best_perm is None:
        return c, {"ok": False, "reason": "no_assignment_within_max_dist", "max_dist_px": float(max_dist_px)}

    return c[best_perm], {
        "ok": True,
        "reason": "ok",
        "perm": best_perm,
        "dists": best_d.tolist() if best_d is not None else None,
        "cost": float(best_cost),
        "max_dist_px": float(max_dist_px),
    }


def _stable_order_circles(
    circles_xy_r: np.ndarray,
    cfg: GearCounterConfig,
    debug: Dict[str, Any],
) -> np.ndarray:
    """Apply stable ordering to circles using module-level reference."""
    global _REF_CENTER_ORDER

    if not bool(cfg.stable_order):
        return circles_xy_r

    if bool(cfg.stable_order_reset):
        _REF_CENTER_ORDER = None
        debug["stable_order"] = {"ok": True, "reason": "reset"}

    c = np.asarray(circles_xy_r)
    n = int(c.shape[0]) if c.ndim == 2 else 0
    if n == 0:
        return circles_xy_r

    centers = [(int(round(x)), int(round(y))) for x, y in c[:, :2]]

    # Initialize reference on first non-empty detection
    if _REF_CENTER_ORDER is None:
        _REF_CENTER_ORDER = centers
        debug["stable_order"] = {"ok": True, "reason": "reference_set", "ref_centers": _REF_CENTER_ORDER}
        return circles_xy_r

    # Only reorder when lengths match
    if len(_REF_CENTER_ORDER) != n:
        debug["stable_order"] = {
            "ok": False,
            "reason": "ref_len_mismatch",
            "n_ref": len(_REF_CENTER_ORDER),
            "n_cur": n,
        }
        return circles_xy_r

    reordered, dbg = _reorder_circles_by_reference(c, _REF_CENTER_ORDER, max_dist_px=float(cfg.stable_order_max_dist))
    debug["stable_order"] = dbg
    return reordered


def validate_circles_geometry(
    circles: np.ndarray,
    *,
    min_clearance: float = 0.0,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Validate that detected circles do not overlap and that no circle center lies inside another.

    circles: (N,3) array-like [x,y,r]
    Rules (with optional min_clearance):
      - No overlap: distance(center_i, center_j) >= r_i + r_j + min_clearance
      - No center-inside-other: distance(center_i, center_j) >= max(r_i, r_j) + min_clearance
        (equivalently: neither center is inside the other circle).
    """
    c = np.asarray(circles, dtype=np.float32)
    if c.size == 0:
        return True, "ok", {"n": 0}

    if c.ndim != 2 or c.shape[1] != 3:
        return False, "bad_shape", {"shape": list(c.shape)}

    n = int(c.shape[0])
    bad_pairs = []
    for i in range(n):
        x1, y1, r1 = float(c[i, 0]), float(c[i, 1]), float(c[i, 2])
        for j in range(i + 1, n):
            x2, y2, r2 = float(c[j, 0]), float(c[j, 1]), float(c[j, 2])
            d = math.hypot(x2 - x1, y2 - y1)

            # centers inside each other
            if d < max(r1, r2) + float(min_clearance):
                bad_pairs.append(
                    {
                        "i": i,
                        "j": j,
                        "type": "center_inside_other",
                        "d": float(d),
                        "r_i": float(r1),
                        "r_j": float(r2),
                        "threshold": float(max(r1, r2) + float(min_clearance)),
                    }
                )
                continue

            # circles overlap
            if d < (r1 + r2) + float(min_clearance):
                bad_pairs.append(
                    {
                        "i": i,
                        "j": j,
                        "type": "overlap",
                        "d": float(d),
                        "r_i": float(r1),
                        "r_j": float(r2),
                        "threshold": float((r1 + r2) + float(min_clearance)),
                    }
                )

    if bad_pairs:
        return False, "invalid_geometry", {"n": n, "bad_pairs": bad_pairs}

    return True, "ok", {"n": n}


def _bilinear_sample(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """img: (H,W) float32, xs/ys float coords, out-of-bounds -> 0"""
    h, w = img.shape
    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    wx = xs - x0
    wy = ys - y0

    def inb(x, y):
        return (x >= 0) & (x < w) & (y >= 0) & (y < h)

    m00 = inb(x0, y0)
    m10 = inb(x1, y0)
    m01 = inb(x0, y1)
    m11 = inb(x1, y1)

    out = np.zeros_like(xs, dtype=np.float32)

    if np.any(m00):
        out[m00] += img[y0[m00], x0[m00]] * (1 - wx[m00]) * (1 - wy[m00])
    if np.any(m10):
        out[m10] += img[y0[m10], x1[m10]] * wx[m10] * (1 - wy[m10])
    if np.any(m01):
        out[m01] += img[y1[m01], x0[m01]] * (1 - wx[m01]) * wy[m01]
    if np.any(m11):
        out[m11] += img[y1[m11], x1[m11]] * wx[m11] * wy[m11]

    return out


def find_best_circle_radius(
    image: np.ndarray,
    center_xy: Tuple[float, float],
    r_min: int,
    r_max: Optional[int],
    step: int = 1,
    thickness: int = 2,
    angles: int = 720,
    blur_ksize: int = 5,
    canny1: int = 50,
    canny2: int = 150,
    refine_subpixel: bool = True,
    size_penalty: float = 0.0,
) -> Tuple[float, Dict[str, Any]]:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    if gray.dtype != np.uint8:
        g = np.clip(gray.astype(np.float32), 0, 255)
        gray_u8 = g.astype(np.uint8)
    else:
        gray_u8 = gray

    if blur_ksize and blur_ksize >= 3 and blur_ksize % 2 == 1:
        gray_u8 = cv2.GaussianBlur(gray_u8, (blur_ksize, blur_ksize), 0)

    edges = cv2.Canny(gray_u8, int(canny1), int(canny2)).astype(np.float32) / 255.0
    h, w = edges.shape
    cx, cy = float(center_xy[0]), float(center_xy[1])

    if r_max is None:
        r_max = int(min(cx, cy, (w - 1) - cx, (h - 1) - cy))
    r_max = max(int(r_min), int(r_max))

    radii = np.arange(int(r_min), int(r_max) + 1, int(step), dtype=np.float32)
    if radii.size == 0:
        raise ValueError("Empty radius range. Check r_min/r_max/step.")

    thetas = np.linspace(0.0, 2.0 * np.pi, int(angles), endpoint=False, dtype=np.float32)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    thickness = max(1, int(thickness))
    half = thickness // 2
    offsets = np.arange(-half, half + 1, dtype=np.float32)

    scores = np.zeros_like(radii, dtype=np.float32)
    for i, r in enumerate(radii):
        acc = 0.0
        cnt = 0
        for dr in offsets:
            rr = r + dr
            if rr <= 0:
                continue
            xs = cx + rr * cos_t
            ys = cy + rr * sin_t
            vals = _bilinear_sample(edges, xs, ys)
            acc += float(vals.mean())
            cnt += 1
        scores[i] = acc / max(1, cnt)

    # Optional: penalize very large radii
    if size_penalty and size_penalty > 0.0 and r_max > 0:
        rr = radii / float(r_max)  # 0..1
        scores = scores / (1.0 + float(size_penalty) * (rr * rr))

    best_idx = int(np.argmax(scores))
    best_r = float(radii[best_idx])
    best_score = float(scores[best_idx])

    # Subpixel refinement (parabola fit)
    if refine_subpixel and 0 < best_idx < len(radii) - 1:
        y0, y1, y2 = scores[best_idx - 1], scores[best_idx], scores[best_idx + 1]
        denom = float(y0 - 2.0 * y1 + y2)
        if abs(denom) > 1e-8:
            delta = 0.5 * float(y0 - y2) / denom
            delta = float(np.clip(delta, -1.0, 1.0))
            best_r = float(radii[best_idx] + delta * int(step))

    info = {"best_idx": best_idx, "best_score": best_score, "r_min": int(r_min), "r_max": int(r_max)}
    return best_r, info


def _nearest_center_dist(cx: int, cy: int, centers: List[Tuple[int, int]]) -> float:
    dmin = float("inf")
    for (x2, y2) in centers:
        if x2 == cx and y2 == cy:
            continue
        d = math.hypot(float(x2 - cx), float(y2 - cy))
        dmin = min(dmin, d)
    return dmin


def count_teeth(image_bgr: np.ndarray, cfg: Optional[GearCounterConfig] = None) -> GearCountResult:
    cfg = cfg or GearCounterConfig()
    debug: Dict[str, Any] = {}

    if image_bgr is None or not isinstance(image_bgr, np.ndarray) or image_bgr.size == 0:
        z = np.zeros((1, 1, 3), dtype=np.uint8)
        return GearCountResult(False, "empty_image", [], [], [], [], z, z, {"reason": "empty_image"})

    try:
        blurred = blur(
            image_bgr,
            d=cfg.bilateral_d,
            sigma_color=cfg.bilateral_sigma_color,
            sigma_space=cfg.bilateral_sigma_space,
        )
        pre = brighten_region(
            blurred, cfg.brighten_region, factor=cfg.brighten_factor, blend=cfg.brighten_blend
        )

        # --- Hough centers with CLAHE ---
        gray = cv2.cvtColor(pre, cv2.COLOR_BGR2GRAY)
        if cfg.use_clahe_for_hough:
            clahe = cv2.createCLAHE(
                clipLimit=float(cfg.clahe_clip_limit),
                tileGridSize=(int(cfg.clahe_tile_grid), int(cfg.clahe_tile_grid)),
            )
            gray = clahe.apply(gray)

        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=float(cfg.hough_dp),
            minDist=float(cfg.hough_min_dist),
            param1=float(cfg.hough_param1),
            param2=float(cfg.hough_param2),
            minRadius=int(cfg.hough_min_radius),
            maxRadius=int(cfg.hough_max_radius),
        )
        circles = filter_circles_by_max_distance(circles, float(cfg.max_neighbor_dist))
        circles_sorted = sort_circles(circles, y_tol=int(cfg.sort_y_tol))
        circles_u16 = np.uint16(np.around(circles_sorted)) if circles_sorted is not None else np.empty((0, 3), dtype=np.uint16)

        # Keep circle order stable across frames (optional)
        if isinstance(circles_u16, np.ndarray) and circles_u16.size > 0:
            if circles_u16.ndim == 3 and circles_u16.shape[-1] == 3:
                circles_u16 = circles_u16[0]
            circles_u16 = _stable_order_circles(circles_u16.astype(np.float32), cfg, debug)
            circles_u16 = np.uint16(np.around(circles_u16))

        pre_draw = pre.copy()

        # Validate: reject overlapping/contained detected center circles
        circles_for_check = circles_u16
        if isinstance(circles_for_check, np.ndarray) and circles_for_check.size > 0:
            if circles_for_check.ndim == 3 and circles_for_check.shape[-1] == 3:
                circles_for_check = circles_for_check[0]
            ok_geom, reason_geom, geom_dbg = validate_circles_geometry(
                circles_for_check, min_clearance=float(cfg.center_circle_min_clearance)
            )
            debug["center_circles_geom"] = {"ok": ok_geom, "reason": reason_geom, **geom_dbg}
            if not ok_geom:
                tmp = pre_draw.copy()
                for (xg, yg, rg) in np.uint16(np.around(circles_for_check)):
                    cv2.circle(tmp, (int(xg), int(yg)), int(rg), (0, 255, 255), 2)
                    cv2.circle(tmp, (int(xg), int(yg)), 2, (0, 0, 255), 2)
                return GearCountResult(
                    ok=False,
                    reason="bad_center_circles",
                    teeth=[],
                    centers=[],
                    r_big=[],
                    r_small=[],
                    preprocessed_bgr=tmp,
                    annotated_bgr=tmp,
                    debug=debug,
                )
        else:
            debug["center_circles_geom"] = {"ok": True, "reason": "no_circles", "n": 0}

        centers: List[Tuple[int, int]] = []

        if isinstance(circles_u16, np.ndarray) and circles_u16.size > 0:
            if circles_u16.ndim == 3 and circles_u16.shape[-1] == 3:
                circles_u16 = circles_u16[0]
            for (x, y, r) in circles_u16:
                x, y, r = int(x), int(y), int(r)
                centers.append((x, y))
                cv2.circle(pre_draw, (x, y), r, (0, 255, 0), 2)
                cv2.circle(pre_draw, (x, y), 2, (0, 0, 255), 2)

        debug["centers"] = centers
        debug["n_centers"] = len(centers)

        annotated = pre_draw.copy()
        teeth: List[float] = []
        r_big_list: List[float] = []
        r_small_list: List[float] = []
        r_big_max_local_list: List[int] = []
        r_small_max_local_list: List[int] = []

        for (x, y) in centers:
            center = (int(x), int(y))

            # --- FIX: limit r_max by nearest neighbor distance ---
            dmin = _nearest_center_dist(center[0], center[1], centers)
            if math.isfinite(dmin):
                big_r_max_local = min(int(cfg.big_r_max), int(cfg.big_r_max_from_neighbor_frac * dmin))
                small_r_max_local = min(int(cfg.small_r_max), int(cfg.small_r_max_from_neighbor_frac * dmin))
            else:
                big_r_max_local = int(cfg.big_r_max)
                small_r_max_local = int(cfg.small_r_max)

            big_r_max_local = max(big_r_max_local, int(cfg.big_r_min) + 5)
            small_r_max_local = max(small_r_max_local, int(cfg.small_r_min) + 2)
            r_big_max_local_list.append(int(big_r_max_local))
            r_small_max_local_list.append(int(small_r_max_local))

            r_big, info_big = find_best_circle_radius(
                pre,
                center,
                r_min=int(cfg.big_r_min),
                r_max=int(big_r_max_local),
                step=int(cfg.radius_step),
                thickness=int(cfg.radius_thickness),
                angles=int(cfg.radius_angles),
                blur_ksize=int(cfg.blur_ksize),
                canny1=int(cfg.canny1),
                canny2=int(cfg.canny2),
                refine_subpixel=bool(cfg.refine_subpixel),
                size_penalty=float(cfg.radius_size_penalty),
            )
            r_small, info_small = find_best_circle_radius(
                pre,
                center,
                r_min=int(cfg.small_r_min),
                r_max=int(small_r_max_local),
                step=int(cfg.radius_step),
                thickness=int(cfg.radius_thickness),
                angles=int(cfg.radius_angles),
                blur_ksize=int(cfg.blur_ksize),
                canny1=int(cfg.canny1),
                canny2=int(cfg.canny2),
                refine_subpixel=bool(cfg.refine_subpixel),
                size_penalty=float(cfg.radius_size_penalty),
            )

            r_big_list.append(float(r_big))
            r_small_list.append(float(r_small))

            cv2.circle(annotated, center, int(round(r_big)), (0, 0, 255), 2)
            cv2.circle(annotated, center, int(round(r_small)), (255, 0, 0), 2)
            cv2.circle(annotated, center, 4, (0, 255, 0), -1)

            denom = (cfg.teeth_k2_den / max(1e-9, cfg.teeth_k2_num))
            t = ((float(r_big) / max(1e-9, float(r_small))) * cfg.teeth_k1 / denom)
            teeth.append(float(t))

            debug.setdefault("r_big_scores", []).append(float(info_big.get("best_score", 0.0)))
            debug.setdefault("r_small_scores", []).append(float(info_small.get("best_score", 0.0)))

        debug["r_big_max_local"] = r_big_max_local_list
        debug["r_small_max_local"] = r_small_max_local_list
        debug["r_big"] = r_big_list
        debug["r_small"] = r_small_list
        debug["teeth"] = teeth

        if not teeth:
            return GearCountResult(
                ok=False,
                reason="no_gears_found",
                teeth=[],
                centers=centers,
                r_big=r_big_list,
                r_small=r_small_list,
                preprocessed_bgr=pre_draw,
                annotated_bgr=annotated,
                debug=debug,
            )

        return GearCountResult(
            ok=True,
            reason="ok",
            teeth=teeth,
            centers=centers,
            r_big=r_big_list,
            r_small=r_small_list,
            preprocessed_bgr=pre_draw,
            annotated_bgr=annotated,
            debug=debug,
        )

    except Exception as e:
        return GearCountResult(
            ok=False,
            reason="internal_error",
            teeth=[],
            centers=[],
            r_big=[],
            r_small=[],
            preprocessed_bgr=image_bgr.copy(),
            annotated_bgr=image_bgr.copy(),
            debug={"error": repr(e)},
        )


def main_func(image_bgr: np.ndarray):
    """
    Notebook-compatible wrapper:
      returns (preprocessed_with_centers, teeth_list, annotated_with_radii)
    """
    res = count_teeth(image_bgr, cfg=None)
    return res.preprocessed_bgr, res.teeth, res.annotated_bgr