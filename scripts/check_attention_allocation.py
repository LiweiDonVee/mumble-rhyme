"""Actual pinned Qwen PyTorch fallback at the supported maximum search shape.

No model weights are loaded: this isolates pairwise_decay/masked_fill from
allocator warmup and quantization, using B=8, L=2048, 32 value heads.
"""

import json
import time
from pathlib import Path

import torch
from check_loading_allocations import AllocationAudit
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule


def main():
    torch.manual_seed(42)
    shape = (8, 2048, 32, 128)
    audit = AllocationAudit()
    start = time.perf_counter()
    with audit, torch.inference_mode():
        query, key, value = (torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3))
        decay = torch.full(shape[:-1], -.01, device="cuda", dtype=torch.float32)
        beta = torch.full(shape[:-1], .1, device="cuda", dtype=torch.bfloat16)
        output, state = torch_chunk_gated_delta_rule(
            query, key, value, decay, beta, output_final_state=True, use_qk_l2norm_in_kernel=True,
        )
        assert output.shape == query.shape and torch.isfinite(output).all().item()
        assert state is not None and torch.isfinite(state).all().item()
        torch.cuda.synchronize()
    theoretical_pairwise = 8 * 32 * 32 * 64 * 64 * 4
    assert audit.maximum_masked_fill_bytes == theoretical_pairwise
    assert audit.maximum_tensor_bytes < 4 * 1024**3
    result = {
        "input_shape": list(shape), "chunk_size": 64, "dtype": "bfloat16",
        "pairwise_dtype": "float32", "pairwise_masked_fill_bytes": audit.maximum_masked_fill_bytes,
        "maximum_observed_tensor_bytes": audit.maximum_tensor_bytes,
        "maximum_factory_bytes": audit.maximum_factory_bytes,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "elapsed_seconds": time.perf_counter() - start, "checks_passed": True,
        "scope": "One actual fallback layer at maximum supported search microbatch/context; not full-model VRAM",
    }
    Path("docs/attention-allocation-bound.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
