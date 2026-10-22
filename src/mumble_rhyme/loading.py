"""Bound the optional Transformers allocator warmup without patching global functions."""

import math
import sys
from contextlib import contextmanager

WARMUP_MAX_BYTES = 256 * 1024**2


@contextmanager
def bounded_allocator_warmup(max_bytes=WARMUP_MAX_BYTES, *, warmup_function=None, devices=("cuda",)):
    """Limit only torch.empty inside the pinned loader's disposable warmup.

    That allocation is a speed optimization, not a parameter/activation. Its
    shape is never used. A thread-local TorchFunctionMode recognizes the exact
    upstream function code, preserving all other allocations and their shapes.
    No site-packages or module-global functions are changed.
    """
    import torch
    from torch.overrides import TorchFunctionMode

    if warmup_function is None:
        from transformers.modeling_utils import caching_allocator_warmup

        warmup_function = caching_allocator_warmup
    warmup_code = warmup_function.__code__
    records = []

    class WarmupMode(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            requested_device = kwargs.get("device")
            if (func is torch.empty and requested_device is not None
                    and torch.device(requested_device).type in devices):
                shape = kwargs.get("size", args[0] if len(args) == 1 else args)
                dtype = kwargs.get("dtype") or torch.get_default_dtype()
                requested = (shape if isinstance(shape, int) else math.prod(shape)) * dtype.itemsize
                if requested > max_bytes:
                    frame = sys._getframe(1)
                    try:
                        while frame is not None and frame.f_code is not warmup_code:
                            frame = frame.f_back
                        if frame is not None:
                            count = max_bytes // dtype.itemsize
                            records.append({"requested_bytes": requested, "allocated_bytes": count * dtype.itemsize})
                            bounded_kwargs = dict(kwargs)
                            bounded_kwargs.pop("size", None)
                            return func((count,), **bounded_kwargs)
                    finally:
                        del frame
            return func(*args, **kwargs)

    with WarmupMode():
        yield records
