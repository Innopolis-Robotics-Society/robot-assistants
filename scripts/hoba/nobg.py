
# #!/usr/bin/env python3
# import cv2 as cv
# import numpy as np
# from ultralytics import YOLO

# MODEL = "/home/ben/manip/best_fixed.pt"
# CAM_INDEX = 0
# IMGZ = 640
# PAD_REL = 0.07   # доля от размера бокса на расширение
# MIN_PAD = 8      # минимум пикселей расширения

# def expand_box(x1, y1, x2, y2, W, H, pad_rel=PAD_REL, min_pad=MIN_PAD):
#     w = x2 - x1
#     h = y2 - y1
#     px = max(min_pad, int(w * pad_rel))
#     py = max(min_pad, int(h * pad_rel))
#     x1 = max(0, x1 - px)
#     y1 = max(0, y1 - py)
#     x2 = min(W - 1, x2 + px)
#     y2 = min(H - 1, y2 + py)
#     return x1, y1, x2, y2

# def remove_shadows_white_bg(bgr_roi, s_thr=None, v_thr=None, k_div_frac=0.03, keep_largest=True):
#     h, w = bgr_roi.shape[:2]

#     # 0) HSV
#     hsv = cv.cvtColor(bgr_roi, cv.COLOR_BGR2HSV)
#     S, V = hsv[..., 1], hsv[..., 2]

#     # 0.1) auto-estimate thresholds from a border band of the ROI
#     if s_thr is None or v_thr is None:
#         b = max(6, int(0.1 * min(h, w)))
#         border = np.zeros((h, w), np.uint8)
#         border[:b, :] = 1; border[-b:, :] = 1; border[:, :b] = 1; border[:, -b:] = 1
#         s_thr = int(np.clip(np.percentile(S[border == 1], 90) + 5, 40, 120))
#         v_thr = int(np.clip(np.percentile(V[border == 1], 60), 170, 245))
#     # clamp in case user passes nonsense
#     s_thr = int(np.clip(s_thr, 0, 255))
#     v_thr = int(np.clip(v_thr, 0, 255))

#     # 1) illumination normalization
#     gray = cv.cvtColor(bgr_roi, cv.COLOR_BGR2GRAY)
#     k_div = max(31, (int(min(h, w) * k_div_frac) | 1))
#     bg = cv.medianBlur(gray, k_div)
#     norm = cv.divide(gray, bg, scale=255)
#     norm = cv.GaussianBlur(norm, (0, 0), 1.0)

#     # 2) Otsu on normalized gray → object darker than paper
#     _, bin_otsu = cv.threshold(norm, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
#     obj_by_intensity = cv.bitwise_not(bin_otsu)  # 255=object

#     # 3) white-paper mask and color/lowlight object cue
#     bg_white = ((S < s_thr) & (V > v_thr)).astype(np.uint8)  # 1=paper
#     obj_color = ((S > min(255, s_thr + 10)) | (V < max(0, v_thr - 40))).astype(np.uint8)

#     # 4) fuse cues: keep anything that looks like object and is not paper
#     mask = ((obj_by_intensity > 0) | (obj_color > 0)) & (bg_white == 0)
#     mask = (mask.astype(np.uint8) * 255)

#     # 5) light cleanup
#     k3 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
#     k5 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
#     mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, k5)
#     mask = cv.morphologyEx(mask, cv.MORPH_OPEN, k3)

#     # 6) largest component (optional)
#     if keep_largest:
#         num, labels, stats, _ = cv.connectedComponentsWithStats(mask, connectivity=8)
#         if num > 1:
#             largest = 1 + np.argmax(stats[1:, cv.CC_STAT_AREA])
#             mask = np.where(labels == largest, 255, 0).astype(np.uint8)

#     return mask


# def get_segmask_for_idx(res, idx, out_shape, fallback=None):
#     """Достать маску сегментации Ultralytics к размеру исходного кадра. Если нет — вернуть fallback."""
#     try:
#         m = getattr(res, "masks", None)
#         if m is None:
#             return fallback
#         data = getattr(m, "data", None)
#         if data is None or data.shape[0] <= idx:
#             return fallback
#         seg = data[idx].detach().cpu().numpy()  # float 0..1
#         seg = (seg > 0.5).astype(np.uint8) * 255
#         oh, ow = out_shape[:2]
#         seg = cv.resize(seg, (ow, oh), interpolation=cv.INTER_NEAREST)
#         return seg
#     except Exception:
#         return fallback

# def main():
#     # 1) кадр с камеры
#     cap = cv.VideoCapture(CAM_INDEX, cv.CAP_V4L2)
#     cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
#     cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
#     ok, frame = cap.read()
#     cap.release()
#     if not ok:
#         print("Camera read failed"); return

#     H, W = frame.shape[:2]

#     # 2) YOLO
#     model = YOLO(MODEL)
#     res = model.predict(source=frame, imgsz=IMGZ, device="cpu", conf=0.25, verbose=False)[0]
#     if res.boxes is None or res.boxes.xyxy.numel() == 0:
#         cv.imshow("frame (no detections)", frame); cv.waitKey(0); return

#     # 3) лучший бокс
#     idx = int(res.boxes.conf.argmax().item())
#     box = res.boxes.xyxy[idx].detach().cpu().numpy().astype(int)
#     cls = int(res.boxes.cls[idx].item())
#     name = model.names.get(cls, str(cls))

#     x1, y1, x2, y2 = box
#     x1, y1, x2, y2 = expand_box(x1, y1, x2, y2, W, H)
#     if x2 <= x1 or y2 <= y1:
#         cv.imshow("frame (invalid bbox)", frame); cv.waitKey(0); return

#     roi = frame[y1:y2, x1:x2].copy()

#     # 4) маска сегментации от YOLO, если есть
#     seg_full = get_segmask_for_idx(res, idx, frame.shape, fallback=None)
#     seg_roi = None
#     if seg_full is not None:
#         seg_roi = seg_full[y1:y2, x1:x2]
#         # немного расширим сегмент для сохранения тонких краев
#         k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
#         seg_roi = cv.morphologyEx(seg_roi, cv.MORPH_DILATE, k)

#     # 5) анти-тень маска
#     mask_shadowless = remove_shadows_white_bg(roi)

#     # 6) комбинированная маска: если есть сегментация — пересечение
#     if seg_roi is not None:
#         mask = cv.bitwise_and(mask_shadowless, seg_roi)
#         # если пересечение слишком маленькое, вернемся к shadowless
#         if cv.countNonZero(mask) < 0.25 * cv.countNonZero(mask_shadowless):
#             mask = mask_shadowless
#     else:
#         mask = mask_shadowless

#     # 7) производные: вырезки и бинарные инверсии
#     cut_white_bg = np.full_like(roi, 255)
#     cut_black_bg = np.zeros_like(roi)
#     cut_white_bg[mask == 255] = roi[mask == 255]
#     cut_black_bg[mask == 255] = roi[mask == 255]

#     bin_white_obj_black_bg = np.zeros((roi.shape[0], roi.shape[1]), np.uint8)
#     bin_white_obj_black_bg[mask == 255] = 255
#     bin_black_obj_white_bg = 255 - bin_white_obj_black_bg

#     # 8) визуализация
#     vis = frame.copy()
#     cv.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     cv.putText(vis, f"{name}", (x1, max(0, y1 - 6)),
#                cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv.LINE_AA)

#     cv.imshow("frame + bbox", vis)
#     cv.imshow("roi", roi)
#     if seg_roi is not None:
#         cv.imshow("yolo_seg_roi", seg_roi)
#     cv.imshow("mask_shadowless", mask_shadowless)
#     cv.imshow("mask_final", mask)
#     cv.imshow("cut_white_bg", cut_white_bg)
#     cv.imshow("cut_black_bg", cut_black_bg)
#     cv.imshow("bin_white_obj_black_bg", bin_white_obj_black_bg)
#     cv.imshow("bin_black_obj_white_bg", bin_black_obj_white_bg)
#     cv.waitKey(0)
#     cv.destroyAllWindows()

# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
# import cv2 as cv
# import numpy as np
# from ultralytics import YOLO

# MODEL = "/home/ben/manip/best_fixed.pt"
# CAM_INDEX = 0
# IMGZ = 640
# PAD_REL = 0.07   # expand bbox by 7%
# MIN_PAD = 8      # at least 8 px

# def expand_box(x1, y1, x2, y2, W, H, pad_rel=PAD_REL, min_pad=MIN_PAD):
#     w = x2 - x1
#     h = y2 - y1
#     px = max(min_pad, int(w * pad_rel))
#     py = max(min_pad, int(h * pad_rel))
#     x1 = max(0, x1 - px)
#     y1 = max(0, y1 - py)
#     x2 = min(W - 1, x2 + px)
#     y2 = min(H - 1, y2 + py)
#     return x1, y1, x2, y2

# def remove_shadows_white_bg_auto(bgr_roi, k_div_frac=0.03):
#     """Shadow-robust binary mask of object on white background (ROI only)."""
#     h, w = bgr_roi.shape[:2]
#     hsv = cv.cvtColor(bgr_roi, cv.COLOR_BGR2HSV)
#     S, V = hsv[..., 1], hsv[..., 2]

#     # auto thresholds from a border band that likely sees paper
#     b = max(6, int(0.1 * min(h, w)))
#     bmask = np.zeros((h, w), dtype=bool)
#     bmask[:b, :] = True; bmask[-b:, :] = True; bmask[:, :b] = True; bmask[:, -b:] = True
#     if not bmask.any():  # fallback
#         bmask[:] = True
#     s_thr = int(np.clip(np.percentile(S[bmask], 90) + 5, 40, 120))
#     v_thr = int(np.clip(np.percentile(V[bmask], 60), 170, 245))

#     # illumination normalization
#     gray = cv.cvtColor(bgr_roi, cv.COLOR_BGR2GRAY)
#     k_div = max(31, (int(min(h, w) * k_div_frac) // 2) * 2 + 1)  # odd
#     bg = cv.medianBlur(gray, k_div)
#     norm = cv.divide(gray, bg, scale=255)
#     norm = cv.GaussianBlur(norm, (0, 0), 1.0)

#     # intensity cue
#     _, bin_otsu = cv.threshold(norm, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
#     obj_int = (bin_otsu == 0)

#     # paper and color/lowlight cues
#     paper = (S < s_thr) & (V > v_thr)
#     obj_color = (S > s_thr + 10) | (V < v_thr - 40)

#     # fuse and exclude paper
#     mask = ((obj_int | obj_color) & (~paper)).astype(np.uint8) * 255

#     # cleanup
#     k3 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
#     k5 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
#     mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, k5)
#     mask = cv.morphologyEx(mask, cv.MORPH_OPEN, k3)

#     # keep largest CC
#     num, labels, stats, _ = cv.connectedComponentsWithStats(mask, connectivity=8)
#     if num > 1:
#         largest = 1 + np.argmax(stats[1:, cv.CC_STAT_AREA])
#         mask = np.where(labels == largest, 255, 0).astype(np.uint8)
#     return mask

# def main():
#     # capture
#     cap = cv.VideoCapture(CAM_INDEX, cv.CAP_V4L2)
#     cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
#     cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
#     ok, frame = cap.read()
#     cap.release()
#     if not ok:
#         print("Camera read failed"); return
#     H, W = frame.shape[:2]

#     # YOLO detect best bbox
#     model = YOLO(MODEL)
#     res = model.predict(source=frame, imgsz=IMGZ, device="cpu", conf=0.25, verbose=False)[0]
#     if res.boxes is None or res.boxes.xyxy.numel() == 0:
#         print("No detections")
#         cv.imshow("frame", frame); cv.waitKey(0); return

#     idx = int(res.boxes.conf.argmax().item())
#     box = res.boxes.xyxy[idx].detach().cpu().numpy().astype(int)
#     cls = int(res.boxes.cls[idx].item())
#     name = model.names.get(cls, str(cls))

#     x1, y1, x2, y2 = box
#     x1, y1, x2, y2 = expand_box(x1, y1, x2, y2, W, H)
#     if x2 <= x1 or y2 <= y1:
#         print("Invalid bbox after expand")
#         cv.imshow("frame", frame); cv.waitKey(0); return

#     roi = frame[y1:y2, x1:x2].copy()

#     # shadowless mask on ROI
#     mask = remove_shadows_white_bg_auto(roi)

#     # moments and centroid in ROI coords
#     M = cv.moments(mask, binaryImage=True)
#     if M["m00"] > 0:
#         cx_roi = M["m10"] / M["m00"]
#         cy_roi = M["m01"] / M["m00"]
#         centroid_roi = (cx_roi, cy_roi)
#         # map to image coords
#         cx_img = x1 + cx_roi
#         cy_img = y1 + cy_roi
#         centroid_img = (cx_img, cy_img)
#     else:
#         centroid_roi = None
#         centroid_img = None

#     # visualization
#     vis = frame.copy()
#     cv.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     cv.putText(vis, f"{name}", (x1, max(0, y1-6)),
#                cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv.LINE_AA)
#     if centroid_img is not None:
#         cxi, cyi = int(round(centroid_img[0])), int(round(centroid_img[1]))
#         cv.circle(vis, (cxi, cyi), 6, (0, 0, 255), -1)
#         cv.putText(vis, f"COM ({cxi},{cyi})", (cxi+8, cyi-8),
#                    cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv.LINE_AA)

#     cut_white_bg = np.full_like(roi, 255)
#     cut_white_bg[mask == 255] = roi[mask == 255]

#     # outputs
#     print(f"class={name} bbox=({x1},{y1},{x2},{y2})")
#     if centroid_roi is not None:
#         print(f"centroid_roi=({centroid_roi[0]:.2f},{centroid_roi[1]:.2f}) "
#               f"centroid_img=({centroid_img[0]:.2f},{centroid_img[1]:.2f})")
#     else:
#         print("centroid not found (empty mask)")

#     cv.imshow("frame+COM", vis)
#     cv.imshow("roi", roi)
#     cv.imshow("mask_shadowless_roi", mask)
#     cv.imshow("roi_cut_white_bg", cut_white_bg)
#     cv.waitKey(0)
#     cv.destroyAllWindows()

# if __name__ == "__main__":
#     main()



#!/usr/bin/env python3
#!/usr/bin/env python3
# import cv2 as cv
# import numpy as np
# from ultralytics import YOLO

# # --- config ---
# MODEL = "/home/ben/manip/best_fixed.pt"
# CAM_INDEX = 0
# IMGZ = 640
# PAD_REL = 0.07   # expand bbox by 7%
# MIN_PAD = 8      # at least 8 px

# # --- utils: bbox expand ---
# def expand_box(x1, y1, x2, y2, W, H, pad_rel=PAD_REL, min_pad=MIN_PAD):
#     w = x2 - x1
#     h = y2 - y1
#     px = max(min_pad, int(w * pad_rel))
#     py = max(min_pad, int(h * pad_rel))
#     x1 = max(0, x1 - px)
#     y1 = max(0, y1 - py)
#     x2 = min(W - 1, x2 + px)
#     y2 = min(H - 1, y2 + py)
#     return x1, y1, x2, y2

# # --- shadow-robust mask inside ROI ---
# def remove_shadows_white_bg_auto(bgr_roi, k_div_frac=0.03):
#     h, w = bgr_roi.shape[:2]
#     hsv = cv.cvtColor(bgr_roi, cv.COLOR_BGR2HSV)
#     S, V = hsv[..., 1], hsv[..., 2]

#     # auto thresholds from a border band (models white paper)
#     b = max(6, int(0.1 * min(h, w)))
#     bmask = np.zeros((h, w), dtype=bool)
#     bmask[:b, :] = True; bmask[-b:, :] = True; bmask[:, :b] = True; bmask[:, -b:] = True
#     s_thr = int(np.clip(np.percentile(S[bmask], 90) + 5, 40, 120))
#     v_thr = int(np.clip(np.percentile(V[bmask], 60), 170, 245))

#     # illumination normalization (divide)
#     gray = cv.cvtColor(bgr_roi, cv.COLOR_BGR2GRAY)
#     k_div = max(31, (int(min(h, w) * k_div_frac) // 2) * 2 + 1)  # odd
#     bg = cv.medianBlur(gray, k_div)
#     norm = cv.divide(gray, bg, scale=255)
#     norm = cv.GaussianBlur(norm, (0, 0), 1.0)

#     # intensity cue
#     _, bin_otsu = cv.threshold(norm, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
#     obj_int = (bin_otsu == 0)

#     # paper and color/lowlight cues
#     paper = (S < s_thr) & (V > v_thr)
#     obj_color = (S > s_thr + 10) | (V < v_thr - 40)

#     # fuse and exclude paper
#     mask = ((obj_int | obj_color) & (~paper)).astype(np.uint8) * 255

#     # cleanup
#     k3 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
#     k5 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
#     mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, k5)
#     mask = cv.morphologyEx(mask, cv.MORPH_OPEN, k3)

#     # keep largest CC
#     num, labels, stats, _ = cv.connectedComponentsWithStats(mask, connectivity=8)
#     if num > 1:
#         largest = 1 + np.argmax(stats[1:, cv.CC_STAT_AREA])
#         mask = np.where(labels == largest, 255, 0).astype(np.uint8)
#     return mask

# # --- COM helpers (on binary mask: white=object) ---
# def com_of_mask(mask: np.ndarray):
#     if mask.ndim == 3:
#         mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
#     _, binm = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)
#     M = cv.moments(binm, binaryImage=True)
#     if M["m00"] == 0:
#         return None
#     cx = M["m10"] / M["m00"]
#     cy = M["m01"] / M["m00"]
#     return float(cx), float(cy)

# def com_of_largest_component(mask: np.ndarray):
#     if mask.ndim == 3:
#         mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
#     _, binm = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)
#     num, labels, stats, centroids = cv.connectedComponentsWithStats(binm, connectivity=8)
#     if num <= 1:
#         return None
#     largest = 1 + np.argmax(stats[1:, cv.CC_STAT_AREA])  # skip background 0
#     cx, cy = centroids[largest]
#     return float(cx), float(cy)

# # --- snap any point to be inside white mask ---
# def snap_point_to_mask(cx: float, cy: float, mask_u8: np.ndarray, prefer_interior: bool = True, erode_iters: int = 1):
#     """
#     Returns a point inside the white region. If (cx,cy) is outside,
#     it snaps to the nearest interior white pixel (uses a small erosion if possible).
#     """
#     if mask_u8.ndim == 3:
#         mask_u8 = cv.cvtColor(mask_u8, cv.COLOR_BGR2GRAY)
#     _, binm = cv.threshold(mask_u8, 127, 255, cv.THRESH_BINARY)

#     h, w = binm.shape[:2]
#     ix, iy = int(round(cx)), int(round(cy))
#     ix = np.clip(ix, 0, w - 1); iy = np.clip(iy, 0, h - 1)
#     if binm[iy, ix] > 0:
#         return float(ix), float(iy)

#     # Prefer interior pixels (avoid border) by eroding
#     cand = binm.copy()
#     if prefer_interior:
#         k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
#         eroded = cv.erode(cand, k, iterations=erode_iters)
#         if cv.countNonZero(eroded) > 0:
#             cand = eroded

#     ys, xs = np.nonzero(cand)
#     if xs.size == 0:
#         ys, xs = np.nonzero(binm)
#         if xs.size == 0:
#             return None  # no white region at all

#     pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
#     d2 = (pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2
#     j = int(np.argmin(d2))
#     return float(pts[j, 0]), float(pts[j, 1])

# def draw_centroid(img_bgr: np.ndarray, c, color=(0, 0, 255), label="COM"):
#     out = img_bgr.copy()
#     if c is None:
#         return out
#     x, y = int(round(c[0])), int(round(c[1]))
#     cv.drawMarker(out, (x, y), color, markerType=cv.MARKER_CROSS, markerSize=20, thickness=2)
#     cv.putText(out, f"{label} ({x},{y})", (x + 8, y - 8),
#                cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv.LINE_AA)
#     return out

# # --- main ---
# def main():
#     # capture one frame
#     cap = cv.VideoCapture(CAM_INDEX, cv.CAP_V4L2)
#     cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
#     cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
#     ok, frame = cap.read()
#     cap.release()
#     if not ok:
#         print("Camera read failed"); return
#     H, W = frame.shape[:2]

#     # YOLO detect best bbox
#     model = YOLO(MODEL)
#     res = model.predict(source=frame, imgsz=IMGZ, device="cpu", conf=0.25, verbose=False)[0]
#     if res.boxes is None or res.boxes.xyxy.numel() == 0:
#         print("No detections"); cv.imshow("frame", frame); cv.waitKey(0); return

#     idx = int(res.boxes.conf.argmax().item())
#     box = res.boxes.xyxy[idx].detach().cpu().numpy().astype(int)
#     cls = int(res.boxes.cls[idx].item())
#     name = model.names.get(cls, str(cls))

#     x1, y1, x2, y2 = expand_box(*box, W, H)
#     if x2 <= x1 or y2 <= y1:
#         print("Invalid bbox after expand"); cv.imshow("frame", frame); cv.waitKey(0); return

#     roi = frame[y1:y2, x1:x2].copy()

#     # shadowless mask in ROI
#     mask = remove_shadows_white_bg_auto(roi)

#     # COMs in ROI (raw)
#     com_all_roi = com_of_mask(mask)
#     com_lrg_roi = com_of_largest_component(mask)

#     # snap COMs to be inside white region
#     com_all_roi_in = snap_point_to_mask(*com_all_roi, mask) if com_all_roi else None
#     com_lrg_roi_in = snap_point_to_mask(*com_lrg_roi, mask) if com_lrg_roi else None

#     # map to full image coords
#     com_all_img = (x1 + com_all_roi_in[0], y1 + com_all_roi_in[1]) if com_all_roi_in else None
#     com_lrg_img = (x1 + com_lrg_roi_in[0], y1 + com_lrg_roi_in[1]) if com_lrg_roi_in else None

#     # visualization
#     vis = frame.copy()
#     cv.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     cv.putText(vis, f"{name}", (x1, max(0, y1 - 6)), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv.LINE_AA)
#     vis = draw_centroid(vis, com_all_img, color=(0, 0, 255), label="COM all (inside)")
#     vis = draw_centroid(vis, com_lrg_img, color=(0, 255, 0), label="COM largest (inside)")

#     cut_white_bg = np.full_like(roi, 255)
#     cut_white_bg[mask == 255] = roi[mask == 255]

#     # outputs
#     print(f"class={name} bbox=({x1},{y1},{x2},{y2})")
#     if com_all_roi_in:
#         print(f"COM_all ROI=({com_all_roi_in[0]:.2f},{com_all_roi_in[1]:.2f})  "
#               f"IMG=({com_all_img[0]:.2f},{com_all_img[1]:.2f})")
#     else:
#         print("COM_all: none")
#     if com_lrg_roi_in:
#         print(f"COM_largest ROI=({com_lrg_roi_in[0]:.2f},{com_lrg_roi_in[1]:.2f})  "
#               f"IMG=({com_lrg_img[0]:.2f},{com_lrg_img[1]:.2f})")
#     else:
#         print("COM_largest: none")

#     cv.imshow("frame + COMs (inside)", vis)
#     cv.imshow("roi", roi)
#     cv.imshow("mask_shadowless_roi", mask)
#     cv.imshow("roi_cut_white_bg", cut_white_bg)
#     cv.waitKey(0)
#     cv.destroyAllWindows()

# if __name__ == "__main__":
#     main()




#!/usr/bin/env python3
import cv2 as cv
import numpy as np
from ultralytics import YOLO
import math

# ===================== CONFIG =====================
MODEL = "/home/ben/manip/best_fixed.pt"
CAM_INDEX = 0
IMGZ = 640

PAD_REL = 0.07   # expand bbox by 7%
MIN_PAD = 8      # at least 8 px

# Camera intrinsics (pixels). Fill from calibration.
FX = 1100.0
FY = 1100.0
CX = None  # if None, will use image center
CY = None

# Known physical size that matches the chosen pixel size metric below (meters)
REAL_SIZE_M = 0.050   # e.g., 5 cm

# Which pixel size to use for ranging: "long", "short", or "equiv_diam"
SIZE_MODE = "long"
# ==================================================

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
    """Shadow-robust binary mask of object on white background (ROI only)."""
    h, w = bgr_roi.shape[:2]
    hsv = cv.cvtColor(bgr_roi, cv.COLOR_BGR2HSV)
    S, V = hsv[..., 1], hsv[..., 2]

    # thresholds from border band (paper model)
    b = max(6, int(0.1 * min(h, w)))
    bmask = np.zeros((h, w), dtype=bool)
    bmask[:b, :] = True; bmask[-b:, :] = True; bmask[:, :b] = True; bmask[:, -b:] = True
    s_thr = int(np.clip(np.percentile(S[bmask], 90) + 5, 40, 120))
    v_thr = int(np.clip(np.percentile(V[bmask], 60), 170, 245))

    gray = cv.cvtColor(bgr_roi, cv.COLOR_BGR2GRAY)
    k_div = max(31, (int(min(h, w) * k_div_frac) // 2) * 2 + 1)  # odd
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
    largest = 1 + np.argmax(stats[1:, cv.CC_STAT_AREA])  # skip background 0
    cx, cy = centroids[largest]
    return float(cx), float(cy)

def snap_point_to_mask(cx: float, cy: float, mask_u8: np.ndarray, prefer_interior: bool = True, erode_iters: int = 1):
    """Snap to nearest interior white pixel to guarantee 'inside'."""
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

def mask_size_px(mask: np.ndarray, mode: str = "long") -> float:
    """Measure object pixel size from mask via contour min-area rectangle."""
    if mask.ndim == 3:
        mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    _, binm = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)
    cnts, _ = cv.findContours(binm, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    c = max(cnts, key=cv.contourArea)
    rect = cv.minAreaRect(c)  # ((cx,cy),(w,h),angle)
    wpx, hpx = rect[1]
    if wpx == 0 or hpx == 0:
        return 0.0
    if mode == "long":
        return float(max(wpx, hpx))
    if mode == "short":
        return float(min(wpx, hpx))
    if mode == "equiv_diam":  # diameter from area
        area = cv.contourArea(c)
        return float(math.sqrt(4.0 * area / math.pi))
    return float(max(wpx, hpx))

def depth_from_size(fx_px: float, real_size_m: float, size_px: float) -> float | None:
    """Monocular depth along optical axis (meters) from pinhole model."""
    if fx_px <= 0 or real_size_m <= 0 or size_px <= 0:
        return None
    return fx_px * real_size_m / size_px

def distance_along_ray(Z: float, u_px: float, v_px: float, fx: float, fy: float, cx: float, cy: float) -> float:
    """Euclidean distance camera->point at pixel (u,v) lying at depth Z."""
    x = (u_px - cx) / fx
    y = (v_px - cy) / fy
    return float(Z * math.sqrt(1.0 + x * x + y * y))

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
    # capture one frame
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

    # YOLO detect best bbox
    model = YOLO(MODEL)
    res = model.predict(source=frame, imgsz=IMGZ, device="cpu", conf=0.25, verbose=False)[0]
    if res.boxes is None or res.boxes.xyxy.numel() == 0:
        print("No detections")
        cv.imshow("frame", frame); cv.waitKey(0); return

    idx = int(res.boxes.conf.argmax().item())
    box = res.boxes.xyxy[idx].detach().cpu().numpy().astype(int)
    cls = int(res.boxes.cls[idx].item())
    name = model.names.get(cls, str(cls))

    x1, y1, x2, y2 = expand_box(*box, W, H)
    if x2 <= x1 or y2 <= y1:
        print("Invalid bbox after expand"); cv.imshow("frame", frame); cv.waitKey(0); return

    roi = frame[y1:y2, x1:x2].copy()

    # mask and COM in ROI
    mask = remove_shadows_white_bg_auto(roi)
    com_roi = com_of_largest_component(mask) or com_of_mask(mask)
    com_roi = snap_point_to_mask(*com_roi, mask) if com_roi else None
    com_img = (x1 + com_roi[0], y1 + com_roi[1]) if com_roi else None

    # pixel size for ranging
    size_px = mask_size_px(mask, SIZE_MODE)

    # depth along optical axis via pinhole
    Z = depth_from_size(FX, REAL_SIZE_M, size_px)

    # Euclidean distance along ray through COM
    if Z is not None and com_img is not None:
        D = distance_along_ray(Z, com_img[0], com_img[1], FX, FY, cx_intr, cy_intr)
    else:
        D = None

    # visualization
    vis = frame.copy()
    cv.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv.putText(vis, f"{name}", (x1, max(0, y1 - 6)), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv.LINE_AA)
    vis = draw_centroid(vis, com_img, color=(0, 0, 255), label="COM")

    overlay = []
    overlay.append(f"size_px({SIZE_MODE})={size_px:.1f}" if size_px > 0 else "size_px=NA")
    overlay.append(f"Z={Z:.3f} m" if Z is not None else "Z=NA")
    overlay.append(f"D(COM)={D:.3f} m" if D is not None else "D=NA")
    ytxt = y1 - 26
    for s in overlay:
        ytxt = max(0, ytxt)
        cv.putText(vis, s, (x1, ytxt), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv.LINE_AA)
        cv.putText(vis, s, (x1, ytxt), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv.LINE_AA)
        ytxt += 22

    cv.imshow("frame + COM + distance", vis)
    cv.imshow("roi", roi)
    cv.imshow("mask_shadowless_roi", mask)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
