# Windows model loading and release audit

Date: 2026-10-21. Scope: the pinned Windows / PyTorch 2.8.0+cu128 / RTX 5080 setup, both Qwen3.5 sizes. These checks concern memory ownership and functional inference, not lyric quality.

## What caused the failure

No earlier project model processes were left running when the audit started. Only unrelated Python services were present, and they were left alone. The original failure also reproduced in the previous committed code, but that did not mean the project could do nothing about it.

Three concrete issues were identified and addressed:

1. **Excessive loading staging.** Transformers 5.20.2 materialized weights through safetensors 0.8's pread slice path. That path reads a full tensor into a Rust buffer and then copies it to a Python bytearray. A 9B embedding is approximately 2.03 GB, so the temporary duplication exceeds available system commit headroom. Smaller files alone cannot split an individual large tensor. Full-tensor reads avoid the extra copy; the final NF4 path streams tensors larger than 16 MiB directly into their CUDA destination using a 16 MiB CPU byte buffer. The safetensors reader validates the layout before offsets are used. Source weights are unchanged.
2. **Retained failures.** When a caller keeps an exception, its traceback can retain a partially loaded model. A wrapper exception can keep the same allocation through its cause/context. Failed loads now clear allocation-owning frames throughout those chains, close checkpoint readers, drop partial references, collect unreachable objects, and release unused CUDA cache before propagating the error.
3. **A cached cuBLAS workspace pinning an allocator segment.** An initial explicit 4B unload dropped model tensors but retained an approximately 8 MiB workspace inside an approximately 3 GB reserved CUDA block. Explicit `unload(release_runtime_cache=True)` now synchronizes, clears PyTorch 2.8's process-wide cached cuBLAS workspace through its available internal hook, and then empties the unused CUDA cache. The default unload does not purge global workspaces; the opt-in requires external CUDA work in the same process to be stopped. This does not reset the GPU or terminate other processes.

There is also one-time CUDA/library initialization that remains in the process after model unload. It was measured separately from model ownership and did not grow by a model-sized amount per cycle. Process-private bytes on Windows include committed driver/accelerator resources; they are not a measurement of physical RAM alone.

## Reproduction

Install the tested model environment and cache the selected weights, then run serially:

```powershell
$env:HF_HUB_OFFLINE = "1"
python scripts/check_model_memory.py --model 9b
python scripts/check_model_memory.py --model 4b
```

Each audit injects a MemoryError after 99 actual tensor reads, deliberately keeps that error alive, checks the failed instance is unloaded, then performs two full load/score/unload cycles in the **same process**. The already-loaded instance must reuse the same object; the opt-in full-cache unload must destroy the model weak reference and release both allocated and reserved CUDA memory. Process-private growth between the two released states must remain below 256 MiB; this threshold is an integration-check bound, not a universal memory guarantee.

## Final results (explicit full-cache unload)

| Measurement | 9B | 4B |
| --- | --- | --- |
| Injected failure followed by retry | Passed | Passed |
| Complete same-process cycles | 2 | 2 |
| CUDA allocated after each unload | 1,024 bytes | 1,024 bytes |
| CUDA reserved after each unload | 2,097,152 bytes | 2,097,152 bytes |
| Process-private growth, released cycle 2 vs 1 | 23.18 MiB | 21.18 MiB |

Raw measurements: [9B lifecycle](model-memory-9b.json), [4B lifecycle](model-memory-4b.json).

Functional regression runs also passed after the streaming change: [9B direct](model-smoke-memory-fixed.json), [9B MCP](mcp-model-smoke-memory-fixed.json), [4B direct](model-smoke-memory-fixed-4b.json), [4B MCP](mcp-model-smoke-memory-fixed-4b.json). They cover actual generation, contextual likelihood, right-context gap scoring, and structured stdio transport. Earlier version measurements remain in their original files.

An attempted concurrent CPU test environment plus 9B smoke exhausted the host's available resources; it was rerun serially and passed. The fix reduces avoidable staging and ownership leaks, but does not create unlimited system or GPU memory.

## Lifetime contract

An MCP process intentionally keeps its model between calls to avoid reloading on every request. Stop the server process to release its CUDA context. Python users who want to drop weights while keeping their process alive can call:

```python
service.model.unload()  # Drop this adapter's weights; do not purge shared workspaces.
# Only when external CUDA work in this process is stopped:
service.model.unload(release_runtime_cache=True)
# A subsequent model operation loads the same pinned model again.
```

`unload()` shares a process-wide project lock with all LocalModel loads and inference calls. Full workspace cleanup is optional because the cuBLAS cache is process-wide: external PyTorch callers must stop their concurrent CUDA work before requesting it. The project lock cannot serialize unrelated code. Default unload and automatic load-error recovery therefore do not clear shared workspaces. It drops references owned by the adapter; if user code separately retains the model or tensors, those references still own memory. A CUDA context and loaded runtime libraries can remain until process exit. The tiny final allocation/reservation above is the observed result, not a cross-platform promise of exact zero.
