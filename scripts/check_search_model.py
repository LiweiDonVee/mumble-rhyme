"""Opt-in full-beam actual-model search and independent probability check."""

import argparse
import json
import time
from pathlib import Path

import torch

from mumble_rhyme.model import LocalModel
from mumble_rhyme.profile import Profile
from mumble_rhyme.service import LyricService


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["4b", "9b"], default="4b")
    parser.add_argument("--microbatch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--output", type=Path, help="Preserve a previous validation record")
    args = parser.parse_args()
    service = LyricService(Profile.load("examples/mumble.yaml"), LocalModel(model=args.model))
    start = time.perf_counter()
    result = service.search(
        "light", left="the city lights are", intent="深夜城市的孤独感",
        beam_width=20, branch_top_k=20, max_new_tokens=args.max_new_tokens,
        microbatch_size=args.microbatch_size, limit=10,
    )
    elapsed = time.perf_counter() - start
    assert result["candidates"]
    assert result["search"]["beam_width"] == result["search"]["branch_top_k"] == 20
    adapter = service.model
    prefix = result["search"]["conditioning_token_ids"]
    checks = []
    with adapter.lock, torch.inference_mode():
        for row in result["candidates"][:3]:
            tokens = prefix + row["token_ids"]
            inputs = torch.tensor([tokens], dtype=torch.long, device=adapter.model.device)
            positions = torch.arange(len(prefix) - 1, len(tokens) - 1, device=inputs.device)
            logits = adapter.model(input_ids=inputs, use_cache=False, logits_to_keep=positions).logits[0].float()
            wanted = inputs[0, len(prefix):]
            independent = logits.log_softmax(-1).gather(-1, wanted[:, None]).sum().item()
            error = abs(independent - row["sequence_logprob"])
            assert error < .08, (independent, row["sequence_logprob"])
            checks.append({"token_ids": row["token_ids"], "independent_logprob": independent,
                           "search_logprob": row["sequence_logprob"], "absolute_error": error})
        # All paths at a search depth have the same token count, so no padding is
        # needed semantically. Verify configurable microbatching stays equivalent.
        paths = [tuple(r["token_ids"][:2]) for r in result["candidates"][:2]]
        individual = adapter._next_token_options(prefix, paths, 20, 1)
        batched = adapter._next_token_options(prefix, paths, 20, 2)
        differences = [{"token_id": token, "single": dict(a).get(token), "batch": value}
                       for a, b in zip(individual, batched) for token, value in b[:5]]
        # NF4/BF16 GEMM kernels can round a logit by one 0.125 step when
        # the batch shape changes. Record this drift, not bitwise equivalence.
        assert all(d["single"] is not None and abs(d["single"] - d["batch"]) < .15
                   for d in differences), differences
    record = {"elapsed_seconds_including_load": elapsed,
              "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
              "independent_probability_checks": checks, "microbatch_comparison_passed": True,
              "microbatch_logprob_comparison": differences,
              "microbatch_logprob_tolerance": .15,
              "result": result}
    path = args.output or Path(__file__).resolve().parents[1] / f"docs/search-model-{args.model}.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(path), "seconds": elapsed,
                      "peak_allocated_gib": record["peak_allocated_gib"],
                      "options": [r["text"] for r in result["candidates"]]}, ensure_ascii=False), flush=True)
    service.model.unload(release_runtime_cache=True)


if __name__ == "__main__":
    main()
