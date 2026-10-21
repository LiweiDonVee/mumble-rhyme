"""Cold production MCP call for actual token-path search (no preceding score)."""

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


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["4b", "9b"], default="4b")
    args = parser.parse_args()
    params = StdioServerParameters(command=sys.executable,
        args=["-m", "mumble_rhyme", "--model", args.model, "serve"], env=dict(os.environ))
    async with stdio_client(params) as (read, write), ClientSession(
        read, write, read_timeout_seconds=timedelta(minutes=5)
    ) as session:
        await session.initialize()
        listed = await session.list_tools()
        assert "search_lyrics" in {t.name for t in listed.tools}
        started = time.perf_counter()
        request = {"target":"light", "left":"the city lights are", "intent":"深夜城市的孤独感",
                   "beam_width":20, "branch_top_k":20, "max_new_tokens":12, "limit":5}
        response = await session.call_tool("search_lyrics", request)
        assert not response.isError, response.content
        result = response.structuredContent
        assert result["candidates"] and result["scoring_mode"] == "token_path_search"
        assert result["model"]["model_id"] == f"Qwen/Qwen3.5-{args.model.upper()}"
        assert result["search"]["beam_width"] == 20
        assert all(r["sequence_logprob"] < 0 and r["rhyme_weight"] > 0 for r in result["candidates"])
        record = {"transport":"stdio", "elapsed_seconds_including_load":time.perf_counter()-started,
                  "request":request, "response":result, "checks_passed":True}
    path = Path(__file__).resolve().parents[1] / f"docs/search-mcp-{args.model}.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"output":str(path),"seconds":record["elapsed_seconds_including_load"]}),flush=True)


if __name__ == "__main__":
    asyncio.run(main())
