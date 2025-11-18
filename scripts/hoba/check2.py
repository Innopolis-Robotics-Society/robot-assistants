from ultralytics import YOLO
m = YOLO("/home/ben/manip/best_fixed.pt")
r = m.predict(source="/home/ben/manip/hoba.png", imgsz=640, device="cpu", conf=0.25, verbose=False)
res = r[0]
print(res.boxes.xyxy, res.boxes.conf, res.boxes.cls, m.names)

