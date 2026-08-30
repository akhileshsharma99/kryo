"""In-process vLLM engine + HTTP, from the official docker rootfs.

CUDA graphs are captured during LLM() init. After Kryo restore those graphs
stay in GPU memory, so the cold-start recapture is skipped.
"""

from __future__ import annotations

import json
import os
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")

MODEL = os.environ.get("BENCH_MODEL", "Qwen/Qwen2.5-7B")
PORT = int(os.environ.get("BENCH_PORT", "8000"))
MAX_MODEL_LEN = int(os.environ.get("BENCH_MAX_MODEL_LEN", "512"))
GPU_UTIL = float(os.environ.get("BENCH_GPU_UTIL", "0.90"))

READY = threading.Event()
LLM = None
PARAMS = None


def checkpoint() -> None:
    """Same protocol as the Kryo Python package, stdlib only."""
    cli_pid_value = os.environ.get("KRYO_CLI_PID", "")
    if not cli_pid_value:
        return
    cli_pid = int(cli_pid_value)
    restored = False

    def wake(*_: object) -> None:
        nonlocal restored
        restored = True

    previous = []
    wake_signals = [signal.SIGUSR2]
    rt_min = getattr(signal, "SIGRTMIN", None)
    if isinstance(rt_min, int):
        wake_signals.append(rt_min + 1)
    for sig in wake_signals:
        previous.append((sig, signal.signal(sig, wake)))
    try:
        os.kill(cli_pid, signal.SIGUSR1)
        while not restored:
            time.sleep(0.1)
    finally:
        for sig, handler in previous:
            signal.signal(sig, handler)


def infer(prompt: str = "Hello, world!") -> str:
    from vllm import SamplingParams

    global PARAMS
    if PARAMS is None:
        PARAMS = SamplingParams(max_tokens=1, temperature=0.0)
    outputs = LLM.generate([prompt], PARAMS, use_tqdm=False)
    return outputs[0].outputs[0].text


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: object) -> None:
        return

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in {"/health", "/v1/models"}:
            self._send(200, {"ok": READY.is_set(), "model": MODEL})
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path not in {"/generate", "/v1/completions"}:
            self._send(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode() or "{}")
        except json.JSONDecodeError:
            body = {}
        prompt = body.get("prompt") or body.get("text") or "Hello, world!"
        text = infer(str(prompt))
        self._send(200, {"text": text, "model": MODEL})


def main() -> None:
    global LLM
    from vllm import LLM as VllmLLM

    kwargs = dict(
        model=MODEL,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=4,
        gpu_memory_utilization=GPU_UTIL,
        disable_log_stats=True,
        enforce_eager=False,
        distributed_executor_backend="uni",
    )
    try:
        LLM = VllmLLM(
            **kwargs,
            compilation_config={"cudagraph_capture_sizes": [1, 2, 4]},
        )
    except TypeError:
        LLM = VllmLLM(**kwargs)
    infer()
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    READY.set()
    print(f"READY model={MODEL} port={PORT}", flush=True)
    checkpoint()
    # Stay up after restore so the external timer can hit /generate.
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
