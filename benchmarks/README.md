# Benchmarks

Time to first inference on a real NVIDIA GPU: a normal Python start versus restoring a Kryo snapshot.

Cold start loads the model from disk and runs one inference. Kryo restores a snapshot, then runs the same inference. The Linux page cache is dropped first, so this is a new process reading files from disk, not a warm machine.

Creating the snapshot is a deploy step. It is not included in the restore time.

Numbers live on the [main README](../README.md). They were measured on NVIDIA A10. The larger models below need an 80GB GPU.

| Workload | What starts |
|----------|-------------|
| PyTorch CUDA | import, CUDA init, one GPU matmul |
| YOLOv8n | load weights, one predict |
| Qwen 2.5-0.5B | load weights, first generate |
| Whisper-tiny | load weights, one transcription |
| Qwen 2.5-7B | load weights, first generate |
| torch.compile on 7B | compile + CUDA graphs, first generate |
| vLLM on 7B | engine start + CUDA-graph capture, first generate |
| Qwen 2.5-32B | load weights, first generate (80GB GPU) |

These need Linux, an NVIDIA GPU, CRIU, `cuda-checkpoint`, and root. They will not run on a Mac.

```bash
cd benchmarks
python runner.py --scenario torch_cuda
```
