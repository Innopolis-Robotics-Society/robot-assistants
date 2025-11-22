from ultralytics import YOLO
m = YOLO("/home/ben/manip/best_fixed.pt")
for r in m.predict(source=0, imgsz=640, device="cpu", conf=0.25, stream=True, show=True, verbose=False):
    pass
