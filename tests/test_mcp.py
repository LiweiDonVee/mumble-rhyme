import os
import sys
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@pytest.mark.asyncio
async def test_real_stdio_roundtrip_and_tool_errors():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    params = StdioServerParameters(command=sys.executable, args=["-m", "mumble_rhyme", "serve"], env=env)
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        tools = await session.list_tools()
        assert {t.name for t in tools.tools} == {
            "compare_rhyme",
            "find_rhymes",
            "rank_candidates",
            "suggest_lyrics",
            "model_status",
        }
        comparison = await session.call_tool("compare_rhyme", {"candidate": "night", "target": "light"})
        assert not comparison.isError
        assert comparison.structuredContent["weight"] == 1
        custom = await session.call_tool(
            "compare_rhyme",
            {
                "candidate": "sit",
                "target": "set",
                "profile": {
                    "aliases": {"i": "IH", "e": "EH"},
                    "vowel_rules": [{"sounds": ["i", "e"], "weight": 0.98}],
                },
            },
        )
        assert not custom.isError
        assert custom.structuredContent["alignment"][0]["source"] == "custom_rule"
        assert custom.structuredContent["alignment"][0]["vowel_weight"] == 0.98
        ranking = await session.call_tool(
            "rank_candidates", {"candidates": ["night", "zzxyq"], "target": "light", "use_model": False}
        )
        assert not ranking.isError
        assert len(ranking.structuredContent["candidates"]) == 2
        status = await session.call_tool("model_status", {})
        assert status.structuredContent["model_id"] == "Qwen/Qwen3.5-9B"
        assert status.structuredContent["loaded"] is False
        bad = await session.call_tool("compare_rhyme", {"candidate": "", "target": "light"})
        assert bad.isError
