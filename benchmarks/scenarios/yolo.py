"""Ultralytics YOLO scenario."""

from _base import Timing, output_results

with Timing("import"):
    from ultralytics import YOLO

with Timing("cuda_init"):
    import torch

    if torch.cuda.is_available():
        _ = torch.zeros(1, device="cuda")

with Timing("model_load"):
    model = YOLO("yolov8n.pt")

with Timing("first_inference"):
    import numpy as np

    dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    model.predict(dummy_image, verbose=False)

output_results()
