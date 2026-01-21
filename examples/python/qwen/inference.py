"""Inference script - runs from restored snapshot."""

from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B").cuda()
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

inputs = tokenizer("Hello, I am a language model", return_tensors="pt").cuda()
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
