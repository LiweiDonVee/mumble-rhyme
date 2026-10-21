import os
import sys
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["9b", "4b"])
async def test_real_stdio_roundtrip_and_tool_errors(model):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "mumble_rhyme", "--model", model, "serve"], env=env
    )
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        tools = await session.list_tools()
        assert {t.name for t in tools.tools} == {
            "compare_rhyme",
            "find_rhymes",
            "rank_candidates",
            "suggest_lyrics",
            "model_status",
            "search_lyrics",
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
        assert status.structuredContent["model_id"] == f"Qwen/Qwen3.5-{model.upper()}"
        assert status.structuredContent["loaded"] is False
        lookup = await session.call_tool("find_rhymes", {"target":"mine", "limit":5})
        assert not lookup.isError
        assert lookup.structuredContent["mode"] == "balanced"
        assert "line" in [r["text"] for r in lookup.structuredContent["candidates"]]
        invalid_search = await session.call_tool("search_lyrics", {"target":"mine", "beam_width":0})
        assert invalid_search.isError
        bad = await session.call_tool("compare_rhyme", {"candidate": "", "target": "light"})
        assert bad.isError
