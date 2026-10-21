"""Explicit GPU integration smoke; never part of the offline pytest suite."""

import json
import time
from pathlib import Path

import torch

from mumble_rhyme.service import LyricService


def main():
    started = time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    probe = torch.ones(8, device="cuda")
    assert (probe * 2).sum().item() == 16
    service = LyricService()
    ranked = service.rank(["head", "chest", "dreams"], "bed", left="I keep your voice inside my")
    assert ranked["scoring_mode"] == "local_model"
    assert all(row["lm_mean_logprob"] < 0 and row["scored_tokens"] > 0 for row in ranked["candidates"])
    generation = service.suggest(
        "night", intent="深夜失眠，想打电话却没勇气。短句英文歌词，口语化。", count=3
    )
    assert generation["candidates"]
    assert all(row["lm_mean_logprob"] is not None for row in generation["candidates"])
    result = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "ranking": ranked,
        "generation": generation,
    }
    path = Path(__file__).resolve().parents[1] / "docs/model-smoke.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "result_path": str(path),
                "elapsed_seconds": result["elapsed_seconds"],
                "peak_allocated_gib": result["peak_allocated_gib"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
