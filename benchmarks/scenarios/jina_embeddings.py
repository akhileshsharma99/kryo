"""Jina Embeddings v3 scenario using transformers."""

from _base import Timing, output_results

with Timing("import"):
    from transformers import AutoModel, AutoTokenizer

with Timing("cuda_init"):
    import torch

    if torch.cuda.is_available():
        _ = torch.zeros(1, device="cuda")
        device = "cuda"
    else:
        device = "cpu"

with Timing("model_load"):
    model_name = "jinaai/jina-embeddings-v3"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)

with Timing("first_inference"):
    text = "Hello, world! This is a test sentence for embedding."
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    # Get embeddings from last hidden state
    _ = outputs.last_hidden_state.mean(dim=1)

output_results()
