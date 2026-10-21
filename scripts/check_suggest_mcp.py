"""Opt-in cold suggest-first MCP check using the reported city-lights request.

--inject-failure is test-only: a child bootstrap wraps one actual model's
generate method, allocating a CUDA tensor then raising the reported error.
No fault-injection setting exists in the production server.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from mumble_rhyme.model import MODEL_SPECS

REQUEST = {
    "target": "light", "left": "the city lights are", "intent": "深夜城市的孤独感", "count": 3,
}

BOOTSTRAP = '''
import numpy
import torch
from transformers import Qwen3_5ForConditionalGeneration
from mumble_rhyme.cli import main
from mumble_rhyme.model import LocalModel
original = Qwen3_5ForConditionalGeneration.generate
original_status = LocalModel.status
def status(self):
    result = original_status(self)
    result["test_cuda_allocated_bytes"] = torch.cuda.memory_allocated()
    result["test_cuda_reserved_bytes"] = torch.cuda.memory_reserved()
    return result
LocalModel.status = status
failed = False
def generate(self, *args, **kwargs):
    global failed
    if not failed:
        failed = True
        temporary = torch.empty(16 * 1024**2, dtype=torch.uint8, device="cuda:0")
        raise RuntimeError("CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate(handle) [injected]")
    return original(self, *args, **kwargs)
Qwen3_5ForConditionalGeneration.generate = generate
main()
'''


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(MODEL_SPECS), default="9b")
    parser.add_argument("--inject-failure", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    entry = ["-c", BOOTSTRAP] if args.inject_failure else ["-m", "mumble_rhyme"]
    params = StdioServerParameters(
        command=sys.executable,
        args=entry + ["--model", args.model, "--profile", str(root / "examples/mumble.yaml"), "serve"],
        env=dict(os.environ),
    )
    result = {"request": REQUEST, "model": args.model, "injected_failure": args.inject_failure, "calls": []}
    async with stdio_client(params) as (read, write), ClientSession(
        read, write, read_timeout_seconds=timedelta(minutes=5)
    ) as session:
        await session.initialize()
        if args.inject_failure:
            failed = await session.call_tool("suggest_lyrics", REQUEST)
            assert failed.isError
            error = " ".join(c.text for c in failed.content if c.type == "text")
            assert "CUBLAS_STATUS_ALLOC_FAILED" in error and "released" in error
            status = await session.call_tool("model_status", {})
            assert status.structuredContent["loaded"] is False
            assert "CUBLAS_STATUS_ALLOC_FAILED" in status.structuredContent["last_resource_error"]
            assert status.structuredContent["test_cuda_allocated_bytes"] < 128 * 1024**2
            result["failure_response"] = error
            result["after_failure_status"] = status.structuredContent
        # First model operation on the production entrypoint is suggestion,
        # not ranking or a tiny random-model check that warms cuBLAS first.
        for _ in range(2):
            started = time.perf_counter()
            response = await session.call_tool("suggest_lyrics", REQUEST)
            assert not response.isError, response.content
            content = response.structuredContent
            assert content["model"]["model_id"] == MODEL_SPECS[args.model][0]
            assert content["model"]["revision"] == MODEL_SPECS[args.model][1]
            # count is the finite proposal budget; duplicate proposals may be
            # deduplicated, with requested/received counts disclosed by the API.
            assert 1 <= len(content["candidates"]) <= REQUEST["count"]
            assert all(c["scored_tokens"] > 0 for c in content["candidates"])
            result["calls"].append({"elapsed_seconds": time.perf_counter() - started, "response": content})
        status = await session.call_tool("model_status", {})
        assert status.structuredContent["loaded"] is True
        assert status.structuredContent["last_resource_error"] is None
        result["final_status"] = status.structuredContent
    result["checks_passed"] = True
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "calls_seconds": [c["elapsed_seconds"] for c in result["calls"]]}))


if __name__ == "__main__":
    asyncio.run(main())
