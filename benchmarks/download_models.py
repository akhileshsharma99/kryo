"""Pre-download model weights for cold start benchmarking.

This ensures cold start benchmarks measure startup time, not download time.
Run during image build: python download_models.py
"""

MODELS = {
    "yolo": {
        "download": lambda: __import__("ultralytics").YOLO("yolov8n.pt"),
    },
    "qwen3": {
        "download": lambda: (
            __import__("transformers").AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B"),
            __import__("transformers").AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B"),
        ),
    },
    "whisper": {
        "download": lambda: (
            __import__("transformers").WhisperProcessor.from_pretrained("openai/whisper-tiny"),
            __import__("transformers").WhisperForConditionalGeneration.from_pretrained(
                "openai/whisper-tiny"
            ),
        ),
    },
    "jina_embeddings": {
        "download": lambda: (
            __import__("transformers").AutoTokenizer.from_pretrained(
                "jinaai/jina-embeddings-v3", trust_remote_code=True
            ),
            __import__("transformers").AutoModel.from_pretrained(
                "jinaai/jina-embeddings-v3", trust_remote_code=True
            ),
        ),
    },
}


def main() -> None:
    """Download all model weights."""
    for name, config in MODELS.items():
        print(f"Downloading {name}...")
        try:
            config["download"]()
            print(f"  {name}: OK")
        except Exception as e:
            print(f"  {name}: FAILED ({e})")


if __name__ == "__main__":
    main()
