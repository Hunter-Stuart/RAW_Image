import torch
torch.backends.cudnn.enabled = False
import torch.optim
_AdamW = torch.optim.AdamW
def _patched_AdamW(params, *args, **kwargs):
    kwargs["fused"] = False
    return _AdamW(params, *args, **kwargs)
torch.optim.AdamW = _patched_AdamW
from ultralytics import YOLO
model = YOLO("/home/zeng/binocular_isp/data/yolov8n.pt")
model.train(data="/home/zeng/datasets/aodraw/data.yaml", epochs=50, imgsz=640, batch=8, device=0, workers=0, amp=False, project="/home/zeng/train_results", name="aodraw_v1")