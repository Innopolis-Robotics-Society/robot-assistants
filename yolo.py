# pip install ultralytics opencv-python
import sys
import cv2 as cv
import torch
from ultralytics import YOLO

cam = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # индекс камеры
w   = int(sys.argv[2]) if len(sys.argv) > 2 else 1280
h   = int(sys.argv[3]) if len(sys.argv) > 3 else 720
CONF = 0.25  # порог уверенности

cap = cv.VideoCapture(cam)

cap.set(cv.CAP_PROP_FRAME_WIDTH,  w)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, h)
cap.set(cv.CAP_PROP_FPS, 30)
cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))  # often gives better color

# Warm-up to let AE/AWB settle
for _ in range(45):
    cap.read()
    cv.waitKey(1)

# Try to lock after warm-up (support varies by driver)
cap.set(cv.CAP_PROP_AUTO_WB, 0)                 # 0=manual
cap.set(cv.CAP_PROP_WB_TEMPERATURE, 4600)       # tweak 3000–6500 as needed

# OpenCV’s AUTO_EXPOSURE mapping is odd on V4L2: 0.25=manual, 0.75=auto
cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 0.25)        # manual
cap.set(cv.CAP_PROP_EXPOSURE, 200)              # units are driver-specific
cap.set(cv.CAP_PROP_GAIN, 0)


model = YOLO("yolov8s.pt")
if torch.cuda.is_available():
    model.to("cuda")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    res = model(frame, conf=CONF, verbose=False)[0]
    annotated = res.plot()  # BGR

    cv.imshow("YOLO webcam", annotated)
    # выход по Esc или 'q'
    key = cv.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
