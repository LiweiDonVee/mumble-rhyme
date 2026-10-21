"""Run the actual quantized model through a real MCP stdio client."""

import asyncio
import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    started = time.perf_counter()
    root = Path(__file__).resolve().parents[1]
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mumble_rhyme", "--profile", str(root / "examples/mumble.yaml"), "serve"],
        env=dict(os.environ),
    )
    async with stdio_client(params) as (read, write), ClientSession(
        read, write, read_timeout_seconds=timedelta(minutes=5)
    ) as session:
        print("MCP initialize", flush=True)
        await session.initialize()
        print("MCP compare", flush=True)
        comparison = await session.call_tool("compare_rhyme", {"candidate": "sit", "target": "set"})
        assert not comparison.isError
        assert comparison.structuredContent["alignment"][0]["source"] == "custom_rule"
        print("MCP rank with model", flush=True)
        ranked = await session.call_tool("rank_candidates", {
            "candidates": ["head", "chest", "dreams"], "target": "bed",
            "left": "I keep your voice inside my",
        })
        assert not ranked.isError, ranked.content
        assert ranked.structuredContent["scoring_mode"] == "local_model"
        print("MCP suggest", flush=True)
        suggested = await session.call_tool("suggest_lyrics", {
            "target": "night", "intent": "深夜失眠，想打电话却没勇气。短句英文歌词，口语化。",
            "count": 4,
        })
        assert not suggested.isError, suggested.content
        assert suggested.structuredContent["candidates"]
        assert all(c["scored_tokens"] > 0 for c in suggested.structuredContent["candidates"])
        status = await session.call_tool("model_status", {})
        assert status.structuredContent["loaded"] is True
        result = {
            "transport": "stdio", "elapsed_seconds": time.perf_counter() - started,
            "comparison": comparison.structuredContent, "ranking": ranked.structuredContent,
            "generation": suggested.structuredContent, "model_status": status.structuredContent,
        }
        path = root / "docs/mcp-model-smoke.json"
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"result_path": str(path), "elapsed_seconds": result["elapsed_seconds"]}))


if __name__ == "__main__":
    asyncio.run(main())
