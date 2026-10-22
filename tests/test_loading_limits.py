"""Optional real torch dispatch checks; no GPU or weights needed."""

import threading

import pytest

torch = pytest.importorskip("torch")


def _warmup(size, device="meta"):
    return torch.empty(size, dtype=torch.float16, device=device)


def test_only_optional_allocator_warmup_is_bounded():
    from mumble_rhyme.loading import bounded_allocator_warmup

    with bounded_allocator_warmup(max_bytes=256, warmup_function=_warmup, devices=("meta",)) as records:
        warmed = _warmup(10000)
        actual_weight = torch.empty(10000, dtype=torch.float16, device="meta")
        ordinary_cpu = torch.empty(1, device=None)
        small = _warmup(8)
    assert warmed.numel() == 128
    assert actual_weight.numel() == 10000
    assert small.numel() == 8
    assert ordinary_cpu.device.type == "cpu"
    assert records == [{"requested_bytes":20000, "allocated_bytes":256}]
    assert _warmup(10000).numel() == 10000


def test_warmup_guard_is_thread_local_and_restored_after_error():
    from mumble_rhyme.loading import bounded_allocator_warmup

    other = []
    with (pytest.raises(RuntimeError, match="test failure"),
          bounded_allocator_warmup(max_bytes=256, warmup_function=_warmup, devices=("meta",))):
        thread = threading.Thread(target=lambda: other.append(_warmup(10000).numel()))
        thread.start()
        thread.join()
        assert _warmup(10000).numel() == 128
        raise RuntimeError("test failure")
    assert other == [10000]
    assert _warmup(10000).numel() == 10000


def test_fallback_pairwise_bound_for_supported_search_budget():
    # [batch, value_heads, ceil(sequence/64), 64, 64], float32
    from mumble_rhyme.model import MAX_INPUT_TOKENS

    maximum_bytes = 8 * 32 * ((MAX_INPUT_TOKENS + 63) // 64) * 64 * 64 * 4
    assert maximum_bytes == 128 * 1024**2
