"""Pre-download weights so timed runs do not include network I/O."""

from __future__ import annotations

import argparse
from pathlib import Path

from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
from ultralytics import YOLO

SCENARIOS_DIR = Path(__file__).parent / "scenarios"

SCENARIO_LLMS = {
    "qwen": "Qwen/Qwen2.5-0.5B",
    "qwen7": "Qwen/Qwen2.5-7B",
    "qwen32": "Qwen/Qwen2.5-32B",
    "torch_compile": "Qwen/Qwen2.5-7B",
    "vllm7": "Qwen/Qwen2.5-7B",
    "vllm32": "Qwen/Qwen2.5-32B",
    "triton7": "Qwen/Qwen2.5-7B",
    "triton32": "Qwen/Qwen2.5-32B",
}


def download_llm(name: str) -> None:
    """Fetch one causal LM into the HuggingFace cache without loading weights."""
    from huggingface_hub import snapshot_download

    print(f"Downloading {name}...")
    snapshot_download(name)


def main() -> None:
    """Fetch default CI weights, or specific LLMs for optional probes."""
    parser = argparse.ArgumentParser(description="Pre-download benchmark weights")
    parser.add_argument(
        "--llm",
        action="append",
        default=[],
        help="HuggingFace model id (repeatable). Skips YOLO/Whisper when set.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        choices=sorted(SCENARIO_LLMS),
        help="Download the LLM for this scenario (repeatable)",
    )
    args = parser.parse_args()

    llms = list(args.llm)
    llms.extend(SCENARIO_LLMS[name] for name in args.scenario)
    if llms:
        for name in dict.fromkeys(llms):
            download_llm(name)
        print("Done")
        return

    print("Downloading yolov8n...")
    YOLO(str(SCENARIOS_DIR / "yolov8n.pt"))
    download_llm("Qwen/Qwen2.5-0.5B")
    print("Downloading openai/whisper-tiny...")
    WhisperProcessor.from_pretrained("openai/whisper-tiny")
    WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    print("Done")


if __name__ == "__main__":
    main()
