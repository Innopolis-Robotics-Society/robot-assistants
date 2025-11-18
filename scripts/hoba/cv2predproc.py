#!/usr/bin/env python3
import cv2 as cv
import numpy as np
from ultralytics import YOLO

MODEL = "/home/ben/take/ur10-assistant/hoba/best_fixed.pt"
CAM_INDEX = 0
IMGZ = 640
S_THR, V_THR = 140, 170  # white ≈ low S, high V

def mask_from_white_bg_debug(bgr, s_thr=S_THR, v_thr=V_THR):
    """Return all intermediate masks for inspection."""
    hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
    S, V = hsv[..., 1], hsv[..., 2]

    # 1) background mask (white-ish): S low AND V high
    bg_bool = (S < s_thr) & (V > v_thr)
    bg_u8   = bg_bool.astype(np.uint8) * 255

    # 2) foreground raw = NOT background
    fg_raw_bool = ~bg_bool
    fg_raw_u8   = fg_raw_bool.astype(np.uint8) * 255

    # 3) morphology
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    fg_open  = cv.morphologyEx(fg_raw_u8,  cv.MORPH_OPEN,  k, iterations=1)
    fg_close = cv.morphologyEx(fg_raw_u8,  cv.MORPH_CLOSE, k, iterations=1)

    # 4) keep largest connected component (from close)
    num, labels, stats, _ = cv.connectedComponentsWithStats(fg_close)
    if num > 1:
        largest = 1 + np.argmax(stats[1:, cv.CC_STAT_AREA])
        fg_largest = (labels == largest).astype(np.uint8) * 255
    else:
        fg_largest = fg_close

    # kernel visualization (scaled up so imshow can display)
    k_vis = cv.resize((k*255).astype(np.uint8), (100, 100), interpolation=cv.INTER_NEAREST)

    return {
        "bg": bg_u8,
        "fg_raw": fg_raw_u8,
        "fg_open": fg_open,
        "fg_close": fg_close,
        "fg_largest": fg_largest,
        "k_vis": k_vis,
    }

def main():
    # 0) grab one frame
    cap = cv.VideoCapture(CAM_INDEX, cv.CAP_V4L2)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1980)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1020)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("Camera read failed"); return

    # 1) detect once
    model = YOLO(MODEL)
    res = model.predict(source=frame, imgsz=IMGZ, device="cpu", conf=0.25, verbose=False)[0]
    if res.boxes is None or res.boxes.xyxy.numel() == 0:
        cv.imshow("frame (no detections)", frame); cv.waitKey(0); return

    # 2) pick highest-confidence detection only
    idx = int(res.boxes.conf.argmax().item())
    box = res.boxes.xyxy[idx].cpu().numpy().astype(int)
    cls = int(res.boxes.cls[idx].item())
    name = model.names.get(cls, str(cls))

    H, W = frame.shape[:2]
    x1,y1,x2,y2 = box
    x1 = max(0, min(x1, W-1)); x2 = max(0, min(x2, W-1))
    y1 = max(0, min(y1, H-1)); y2 = max(0, min(y2, H-1))
    if x2 <= x1 or y2 <= y1:
        cv.imshow("frame (invalid bbox)", frame); cv.waitKey(0); return

    roi = frame[y1:y2, x1:x2].copy()

    # 3) preprocessing and intermediates
    masks = mask_from_white_bg_debug(roi)  # bg, fg_raw, fg_open, fg_close, fg_largest, k_vis


    # 5) show everything, no saving
    vis = frame.copy()
    cv.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
    cv.putText(vis, f"{name}", (x1, max(0,y1-6)), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv.LINE_AA)

    cv.imshow("frame + first bbox", vis)
    cv.imshow("roi", roi)
    cv.imshow("bg (white regions)", masks["bg"])
    cv.imshow("fg_raw (not white)", masks["fg_raw"])
    cv.imshow("fg_open", masks["fg_open"])
    cv.imshow("fg_close", masks["fg_close"])
    cv.imshow("fg_largest (final)", masks["fg_largest"])
    cv.imshow("kernel (vis)", masks["k_vis"])

    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
