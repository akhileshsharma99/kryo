# Kryo

Kryo (pronounced like 'cry-oh') is a Python runtime for sub-second cold starts.

## The Problem

Python ML inference has slow cold starts. A model that runs inference in milliseconds can take tens of seconds to start. This breaks serverless: you keep instances running to avoid startup overhead, which defeats autoscaling to zero.

Slow cold starts force you to choose between high latency (cold start every request) or low utilization (keep instances warm). Neither works for bursty inference workloads.

## Why Fast Cold Starts Matter

**Serverless becomes viable.** Pay only for inference time. Scale to zero between requests. Handle traffic spikes without overprovisioning.

**Higher GPU utilization.** Run more instances on-demand instead of fewer instances always-on.

**Faster iteration.** Restart your inference server in seconds, not minutes. Develop without waiting through framework initialization.

Kryo is a runtime for making Python cold starts fast enough that these patterns work. 
