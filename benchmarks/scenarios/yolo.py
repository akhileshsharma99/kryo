"""YOLOv8n load + one dummy predict on GPU."""

import numpy as np
import torch
from _base import checkpoint_or_exit
from ultralytics import YOLO

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

_ = torch.zeros(1, device="cuda")
model = YOLO("yolov8n.pt")
dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)


def infer() -> None:
    model.predict(dummy_image, verbose=False)


infer()
checkpoint_or_exit(resume=infer)
