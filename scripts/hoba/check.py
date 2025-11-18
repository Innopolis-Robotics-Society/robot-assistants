# check_ultra_load.py
from torch.serialization import add_safe_globals
from ultralytics.nn.tasks import DetectionModel
import torch

add_safe_globals([DetectionModel])  # allow YOLO class during unpickle

ckpt = torch.load("/home/ben/manip/best.zip", map_location="cpu", weights_only=False)
print("ckpt keys:", list(ckpt)[:8])

model_obj = ckpt.get("model", None)  # may be a DetectionModel
if model_obj is not None and hasattr(model_obj, "yaml"):
    model_obj.eval()
    # save a proper .pt for Ultralytics API
    torch.save(ckpt, "/home/ben/manip/best_fixed.pt")
    print("Saved /home/ben/manip/best_fixed.pt")
else:
    print("No full model object; likely weights-only state_dict.")
