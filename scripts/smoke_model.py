"""Explicit GPU integration smoke; never part of the offline pytest suite."""

import argparse
import json
import re
import time
from pathlib import Path

import torch

from mumble_rhyme.model import MODEL_SPECS, LocalModel
from mumble_rhyme.service import LyricService


def main():
    parser = argparse.ArgumentParser(description="Actual selected Qwen model GPU integration")
    parser.add_argument("--model", choices=list(MODEL_SPECS), default="9b")
    args = parser.parse_args()
    torch.manual_seed(42)
    started = time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    probe = torch.ones(8, device="cuda")
    assert (probe * 2).sum().item() == 16
    service = LyricService(model=LocalModel(model=args.model))
    ranked = service.rank(["head", "chest", "dreams"], "bed", left="I keep your voice inside my")
    assert ranked["scoring_mode"] == "local_model"
    assert all(row["lm_mean_logprob"] < 0 and row["scored_tokens"] > 0 for row in ranked["candidates"])
    generation = service.suggest(
        "night", intent="深夜失眠，想打电话却没勇气。短句英文歌词，口语化。", count=3
    )
    assert generation["candidates"]
    assert all(row["lm_mean_logprob"] is not None for row in generation["candidates"])
    # A narrow lexical check for this specific brief, not a lyric-quality metric.
    call_reference = any(
        re.search(r"\b(call|phone|dial|number|ring)\w*\b", row["text"], re.IGNORECASE)
        for row in generation["candidates"]
    )
    assert call_reference, "Generated lines lost the brief's concrete action: wanting to call"
    gap = service.rank(
        ["emotionally unavailable", "alone", "restless"], "table",
        left="I feel", right=", tonight.",
        profile={"aliases": {"i": "IH", "e": "EH"},
                 "vowel_rules": [{"sounds": ["i", "e"], "weight": 0.98}]},
    )
    assert len(gap["candidates"]) == 3
    assert all(row["scored_tokens"] > 1 for row in gap["candidates"])
    result = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "tokenizer_vocabulary": len(service.model.tokenizer),
        "ranking": ranked,
        "generation": generation,
        "gap_with_right_context": gap,
        "call_brief_lexical_check": call_reference,
    }
    suffix = "" if args.model == "9b" else "-4b"
    path = Path(__file__).resolve().parents[1] / f"docs/model-smoke{suffix}.json"
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
