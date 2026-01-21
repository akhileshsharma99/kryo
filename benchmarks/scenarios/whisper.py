"""OpenAI Whisper scenario using transformers."""

from _base import Timing, output_results

with Timing("import"):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

with Timing("cuda_init"):
    import torch

    if torch.cuda.is_available():
        _ = torch.zeros(1, device="cuda")
        device = "cuda"
    else:
        device = "cpu"

with Timing("model_load"):
    model_name = "openai/whisper-tiny"
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)

with Timing("first_inference"):
    import numpy as np

    # Generate 1 second of dummy audio at 16kHz
    dummy_audio = np.random.randn(16000).astype(np.float32)
    inputs = processor(dummy_audio, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device)
    with torch.no_grad():
        generated_ids = model.generate(input_features, max_new_tokens=10)
    _ = processor.batch_decode(generated_ids, skip_special_tokens=True)

output_results()
