"""Opt-in Windows/CUDA failure-recovery and repeated-load audit.

This loads real model weights several times. Core pytest does not run it.
"""

import argparse
import ctypes
import gc
import json
import sys
import threading
import time
import weakref
from ctypes import wintypes
from pathlib import Path
from unittest.mock import patch

import torch

from mumble_rhyme.checkpoint import _FullTensor
from mumble_rhyme.model import LocalModel


class PerformanceInfo(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD)] + [
        (name, ctypes.c_size_t) for name in (
            "CommitTotal", "CommitLimit", "CommitPeak", "PhysicalTotal", "PhysicalAvailable",
            "SystemCache", "KernelTotal", "KernelPaged", "KernelNonpaged", "PageSize",
        )
    ] + [(name, wintypes.DWORD) for name in ("HandleCount", "ProcessCount", "ThreadCount")]


class ProcessMemory(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
        (name, ctypes.c_size_t) for name in (
            "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
            "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage",
            "PrivateUsage",
        )
    ]


def memory_snapshot():
    ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    performance, process = PerformanceInfo(), ProcessMemory()
    performance.cb, process.cb = ctypes.sizeof(performance), ctypes.sizeof(process)
    if not ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(performance), performance.cb):
        raise ctypes.WinError()
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    if not ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(process), process.cb):
        raise ctypes.WinError()
    return {
        "process_private_bytes": process.PrivateUsage,
        "system_commit_available_bytes": (
            performance.CommitLimit - performance.CommitTotal
        ) * performance.PageSize,
        "cuda_allocated_bytes": torch.cuda.memory_allocated(),
        "cuda_reserved_bytes": torch.cuda.memory_reserved(),
        "host_allocator": dict(torch.cuda.memory.host_memory_stats()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["9b", "4b"], default="9b")
    args = parser.parse_args()
    if sys.platform != "win32" or not torch.cuda.is_available():
        raise RuntimeError("This integration check requires Windows and CUDA")
    torch.cuda.init()
    baseline = memory_snapshot()
    print(json.dumps({"stage": "baseline", **baseline}), flush=True)
    samples, stop = [], threading.Event()

    def sample():
        while not stop.wait(0.05):
            samples.append(memory_snapshot())

    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    adapter = LocalModel(model=args.model)
    result = {"model": adapter.status(), "baseline": baseline, "cycles": []}
    started = time.perf_counter()
    try:
        original = _FullTensor.__getitem__
        reads = 0

        def fail_after_partial_load(view, index):
            nonlocal reads
            reads += 1
            if reads == 100:
                raise MemoryError("injected after 99 real tensor reads")
            return original(view, index)

        retained_error = None
        with patch.object(_FullTensor, "__getitem__", fail_after_partial_load):
            try:
                adapter.load()
            except MemoryError as error:
                retained_error = error
        assert retained_error is not None, "Failure injection did not run"
        assert not adapter.status()["loaded"]
        failure = memory_snapshot()
        print(json.dumps({"stage": "after_failure", **failure}), flush=True)
        assert failure["cuda_allocated_bytes"] <= baseline["cuda_allocated_bytes"] + 128 * 1024**2
        result["after_failed_load_with_error_retained"] = failure
        for number in range(2):
            adapter.load()
            owned = weakref.ref(adapter.model)
            score = adapter.score(["head"], left="I keep your voice inside my")
            assert score[0]["scored_tokens"] > 0
            adapter.load()
            assert adapter.model is owned(), "An already loaded instance loaded a second model"
            loaded = memory_snapshot()
            print(json.dumps({"stage": f"loaded_{number + 1}", **loaded}), flush=True)
            adapter.unload(release_runtime_cache=True)
            assert owned() is None
            released = memory_snapshot()
            print(json.dumps({"stage": f"released_{number + 1}", **released}), flush=True)
            assert released["cuda_allocated_bytes"] <= baseline["cuda_allocated_bytes"] + 128 * 1024**2
            assert released["cuda_reserved_bytes"] <= baseline["cuda_reserved_bytes"] + 128 * 1024**2
            result["cycles"].append({"number": number + 1, "loaded": loaded, "released": released})
        retained_error = None
        gc.collect()
        result["released_private_growth_bytes"] = (
            result["cycles"][1]["released"]["process_private_bytes"]
            - result["cycles"][0]["released"]["process_private_bytes"]
        )
        assert result["released_private_growth_bytes"] < 256 * 1024**2
        result["elapsed_seconds"] = time.perf_counter() - started
    finally:
        adapter.unload(release_runtime_cache=True)
        stop.set()
        monitor.join()
    result["peak_process_private_bytes"] = max(s["process_private_bytes"] for s in samples)
    result["minimum_system_commit_available_bytes"] = min(
        s["system_commit_available_bytes"] for s in samples
    )
    result["checks_passed"] = True
    result["release_runtime_cache"] = True
    path = Path(__file__).resolve().parents[1] / f"docs/model-memory-{args.model}.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
