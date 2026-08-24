"""Pre-download weights so timed runs do not include network I/O."""

from pathlib import Path

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
from ultralytics import YOLO

SCENARIOS_DIR = Path(__file__).parent / "scenarios"


def main() -> None:
    """Fetch YOLO, Qwen, and Whisper weights into the local cache."""
    print("Downloading yolov8n...")
    YOLO(str(SCENARIOS_DIR / "yolov8n.pt"))
    print("Downloading Qwen/Qwen2.5-0.5B...")
    AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
    print("Downloading openai/whisper-tiny...")
    WhisperProcessor.from_pretrained("openai/whisper-tiny")
    WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    print("Done")


if __name__ == "__main__":
    main()
