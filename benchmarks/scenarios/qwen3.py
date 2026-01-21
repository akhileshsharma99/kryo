"""Qwen3 LLM scenario using transformers."""

from _base import Timing, output_results

with Timing("import"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

with Timing("cuda_init"):
    import torch

    if torch.cuda.is_available():
        _ = torch.zeros(1, device="cuda")
        device = "cuda"
    else:
        device = "cpu"

with Timing("model_load"):
    model_name = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map=device,
    )

with Timing("first_inference"):
    inputs = tokenizer("Hello, world!", return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10)
    _ = tokenizer.decode(outputs[0], skip_special_tokens=True)

output_results()
