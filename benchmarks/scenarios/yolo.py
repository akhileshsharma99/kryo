"""YOLOv8n load + one dummy predict on GPU."""

import numpy as np
import torch
from _base import maybe_checkpoint
from ultralytics import YOLO

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

_ = torch.zeros(1, device="cuda")
model = YOLO("yolov8n.pt")
dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
model.predict(dummy_image, verbose=False)

maybe_checkpoint()
