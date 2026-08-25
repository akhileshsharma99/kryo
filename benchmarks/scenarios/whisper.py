"""Whisper-tiny load + one dummy transcription on GPU."""

import numpy as np
import torch
from _base import checkpoint_or_exit
from transformers import WhisperForConditionalGeneration, WhisperProcessor

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

_ = torch.zeros(1, device="cuda")
model_name = "openai/whisper-tiny"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name).to("cuda")
dummy_audio = np.random.randn(16000).astype(np.float32)


def infer() -> None:
    inputs = processor(dummy_audio, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to("cuda")
    with torch.no_grad():
        generated_ids = model.generate(input_features, max_new_tokens=10)
    _ = processor.batch_decode(generated_ids, skip_special_tokens=True)


infer()
checkpoint_or_exit(resume=infer)
