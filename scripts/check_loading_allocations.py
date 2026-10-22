"""Actual Windows search with an enforced 4 GiB single-factory allocation ceiling.

Run serially with other model checks. The ceiling is a test condition, not a
claim that all WDDM systems have a fixed 4 GiB hardware limit.
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.overrides import TorchFunctionMode

from mumble_rhyme.model import LocalModel
from mumble_rhyme.service import LyricService


class AllocationAudit(TorchFunctionMode):
    def __init__(self, ceiling=4 * 1024**3):
        super().__init__()
        self.ceiling = ceiling
        self.maximum_factory_bytes = 0
        self.maximum_tensor_bytes = 0
        self.maximum_masked_fill_bytes = 0
        self.blocked = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in (torch.empty, torch.zeros, torch.ones, torch.full):
            device = torch.device(kwargs.get("device", "cpu"))
            if device.type == "cuda":
                shape = kwargs.get("size", args[0] if len(args) == 1 else args)
                if func is torch.full:
                    shape = args[0]
                dtype = kwargs.get("dtype") or torch.get_default_dtype()
                size = (shape if isinstance(shape, int) else math.prod(shape)) * dtype.itemsize
                self.maximum_factory_bytes = max(self.maximum_factory_bytes, size)
                if size > self.ceiling:
                    self.blocked.append(size)
                    raise torch.OutOfMemoryError(f"[audit] Single factory allocation exceeds {self.ceiling} bytes")
        result = func(*args, **kwargs)
        if isinstance(result, torch.Tensor) and result.device.type == "cuda":
            size = result.numel() * result.element_size()
            self.maximum_tensor_bytes = max(self.maximum_tensor_bytes, size)
            if "masked_fill" in func.__name__:
                self.maximum_masked_fill_bytes = max(self.maximum_masked_fill_bytes, size)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["9b", "4b"], default="9b")
    args = parser.parse_args()
    service = LyricService(model=LocalModel(model=args.model))
    audit = AllocationAudit()
    started = time.perf_counter()
    with audit:
        result = service.search(
            "light", left="the city lights are", intent="深夜城市的孤独感",
            beam_width=20, branch_top_k=20, max_new_tokens=12, microbatch_size=1,
        )
    assert result["candidates"] and not audit.blocked
    assert audit.maximum_tensor_bytes < audit.ceiling
    assert result["model"]["loading_warmup"]
    assert all(r["allocated_bytes"] <= 256 * 1024**2 for r in result["model"]["loading_warmup"])
    record = {
        "elapsed_seconds_including_load": time.perf_counter() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "test_single_allocation_ceiling_bytes": audit.ceiling,
        "maximum_factory_bytes": audit.maximum_factory_bytes,
        "maximum_observed_tensor_bytes": audit.maximum_tensor_bytes,
        "maximum_masked_fill_bytes": audit.maximum_masked_fill_bytes,
        "blocked_allocations": audit.blocked, "checks_passed": True, "result": result,
    }
    path = Path(__file__).resolve().parents[1] / f"docs/loading-allocation-{args.model}.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in record.items() if k != "result"}), flush=True)
    service.model.unload(release_runtime_cache=True)


if __name__ == "__main__":
    main()
