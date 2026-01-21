"""Setup script - loads model and warms up CUDA kernels."""

from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B").cuda()
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

# Warm up CUDA kernels
model.generate(**tokenizer("warmup", return_tensors="pt").cuda(), max_new_tokens=1)

print("Setup complete - ready for snapshot")
