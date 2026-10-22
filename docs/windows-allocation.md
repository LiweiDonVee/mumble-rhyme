# Windows 7.22 GiB allocation investigation

Date: 2026-10-22. Fix: 0.3.1. Hardware: RTX 5080, Windows WDDM, driver 610.62, torch 2.8.0+cu128, transformers 5.20.2. All probes ran locally, without web/search tools.

## Exact allocation source

The reported request was reproduced by raw PyTorch: a single **7,752,651,232 byte** (7.220 GiB) CUDA allocation failed despite 14.64 GiB reported free.

A separate instrumented 9B load identified exactly that byte count at:

```text
LocalModel.load
  transformers.from_pretrained
    _load_pretrained_model
      caching_allocator_warmup (modeling_utils.py:5118)
        torch.empty(int(byte_count // 2), dtype=float16, device=cuda)
```

This is a disposable speed optimization before actual weights are loaded. It is not a required parameter, NF4 quantization output, or attention activation. The audit rejected the request before making it, confirming the call site independently of the native probe. Raw evidence: [allocation-investigation.json](allocation-investigation.json).

## Separate system commit pressure

The Windows GraphicsPerfSvc host had approximately **54.4 GiB of private committed memory** with a small physical working set. The follow-up 9B load, after the large warmup was capped, could fail even on a 96 MiB allocation. Free GPU capacity alone was therefore insufficient evidence that the driver could satisfy allocations.

The user authorized restarting GraphicsPerfSvc. Restart succeeded, and the old host subsequently exited on its own; no forced kill was performed. The system's available commit rose to about 45 GiB. The project did not change paging settings, service startup type, or stop unrelated applications. System service maintenance is not performed by the package.

On the same GPU and driver after restart, **both 4.5 GiB and 7.22 GiB native requests succeeded**. Even before restart, the local 4.5 GiB probe succeeded. Thus these observations do **not establish a universal fixed 4 GiB WDDM contiguous-allocation ceiling**. They show an allocation failure sensitive to the Windows environment and commit pressure. The service's internal reason for retaining memory was not established. Raw after-restart measurements: [native-allocation-after-service-restart.json](native-allocation-after-service-restart.json).

## Project fix

Windows loading now limits the upstream **optional allocator warmup** to **256 MiB**. This removes an avoidable single-block request even when the driver/system has sufficient headroom.

The guard uses thread-local `TorchFunctionMode`, matching the exact pinned `caching_allocator_warmup` function code on the calling stack. Only its oversized `torch.empty` is replaced with a smaller disposable warmup tensor. The upstream code discards that result without using its shape. Actual weight, quantization, and activation allocations are unchanged. There is no global function patch, site-packages edit, model substitution, or reduction in the default beam20/top20 search.

`model_status.loading_warmup` reports the original and effective byte counts. For 9B the record is:

```json
{"requested_bytes": 7752651232, "allocated_bytes": 268435456}
```

The records list is attached before loading, so it remains available after a later failed load. Existing allocation-error handling still releases partial models and readers. Enough system commit and GPU memory remain necessary for the real weights and computation.

## Was pairwise_decay also creating a 7.22 GiB tensor?

Not in the pinned supported search path. The fallback chunks sequence positions in groups of 64. Its pairwise float32 tensor has shape:

```text
[microbatch, value_heads, ceil(sequence_length/64), 64, 64]
```

For the maximum supported search microbatch/context (**8 × 2048**, 32 heads), that tensor is **128 MiB**. The actual single-layer fallback was run at that shape, with finite output and final state. Largest observed tensor: **256 MiB**; peak GPU allocation for that isolated layer: approximately **3.30 GiB**. That peak is not a full-model VRAM requirement. The default short search observed a 1 MiB masked-fill tensor. Raw record: [attention-allocation-bound.json](attention-allocation-bound.json).

No attention algorithm was replaced, and no masked-fill failure was silently ignored. A different dependency version, unsupported context/batch, or different operation needs its own trace; the figure above applies to the pinned fallback and supported search limits.

## Verification and reproduction

```console
python scripts/check_loading_allocations.py --model 9b
python scripts/check_loading_allocations.py --model 4b
python scripts/check_attention_allocation.py
python scripts/check_search_model.py --model 9b --output docs/search-model-bounded-warmup-9b.json
python scripts/check_search_mcp.py --model 9b --output docs/search-mcp-bounded-warmup-9b.json
```

The first two run actual NF4 load and the complete default beam20/top20/12-token search under an externally enforced **4 GiB single-factory allocation ceiling**. This is an audit condition, not a claimed hardware specification. The guard sees the smaller optional warmup and does not block any actual allocation. Records include maximum observed tensor/factory sizes, masked-fill sizes, warmup records and candidates:

- [9B allocation audit](loading-allocation-9b.json)
- [4B allocation audit](loading-allocation-4b.json)
- [9B independent probability check](search-model-bounded-warmup-9b.json)
- [9B cold production MCP](search-mcp-bounded-warmup-9b.json)
- [4B cold production MCP](search-mcp-bounded-warmup-4b.json)

The independent model check recomputes the returned path log probabilities; the cold MCP checks call search before any other model operation. CPU/meta regression tests confirm that ordinary allocations and small warmups are unchanged, another thread is unaffected, and the mode is restored after an exception.

After updating, restart the MCP service and check `model_status.package_version=0.3.1`. The failed old project server was stopped for isolated testing; the client must create a fresh connection. If failures recur despite the cap, inspect the actual allocation and system commit rather than inferring a cause from the GPU free-memory number alone.
