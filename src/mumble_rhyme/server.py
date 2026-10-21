from typing import Any

from mcp.server.fastmcp import FastMCP

from .service import LyricService


def create_server(service: LyricService | None = None):
    service = service or LyricService()
    server = FastMCP(
        "mumble-rhyme",
        log_level="WARNING",
        instructions=(
            "Local English songwriting assistant. Custom profile rules apply soft phonetic weights; "
            "all supplied candidates remain. Accept Chinese creative intent, produce English lyrics. "
            "Use compare_rhyme/find_rhymes for cheap dictionary operations; rank_candidates and suggest_lyrics "
            f"load pinned local {service.model.status()['model_id']}. strength controls rhyme preference. "
            "Do not equate rank_weight with a calibrated probability or singing quality. "
            "Use profile aliases/vowel_rules/pronunciations for the user's own mumble allowances."
        ),
    )

    @server.tool(structured_output=True)
    def compare_rhyme(candidate: str, target: str, profile: dict | None = None) -> dict[str, Any]:
        """Explain English rhyme using positive soft weights; no model required.

        profile example: {"aliases":{"i":"IH","e":"EH"},
        "vowel_rules":[{"sounds":["i","e"],"weight":0.95,"symmetric":true}]}
        Position 1 means last syllable; rules are pairwise and never transitive.
        """
        return service.compare(candidate, target, profile)

    @server.tool(structured_output=True)
    def find_rhymes(target: str, limit: int = 20, profile: dict | None = None) -> dict[str, Any]:
        """Rank CMUdict words by soft phonetics. No semantic model; disclose dictionary search coverage."""
        return service.find(target, limit, profile)

    @server.tool(structured_output=True)
    def rank_candidates(
        candidates: list[str],
        target: str,
        left: str = "",
        right: str = "",
        intent: str = "",
        strength: float = 2.0,
        target_syllables: int | None = None,
        use_model: bool = True,
        profile: dict | None = None,
    ) -> dict[str, Any]:
        """Rank 1-32 gap-fill alternatives using local Qwen likelihood plus soft rhyme weights.

        Provide context on both sides and optionally a Chinese meaning/mood. For entire-line
        alternatives, leave left/right empty. strength=0 removes the rhyme preference.
        target_syllables is a soft preference for the inserted text, not the entire line.
        use_model=false explicitly returns phonetic-only scores without downloading/loading Qwen.
        Every supplied alternative stays in the response, including unknown words.
        """
        return service.rank(
            candidates, target, left, right, intent, strength, target_syllables, use_model, profile
        )

    @server.tool(structured_output=True)
    def suggest_lyrics(
        target: str,
        left: str = "",
        right: str = "",
        intent: str = "",
        count: int = 8,
        strength: float = 2.0,
        target_syllables: int | None = None,
        profile: dict | None = None,
    ) -> dict[str, Any]:
        """Generate then rerank English gap fills or complete lyric lines with the selected pinned Qwen3.5.

        intent may be Chinese. count=1..12 is a finite search budget, not a rhyme gate.
        Unknown and uncertain rhymes are retained. First call loads the GPU model and can be slow.
        """
        return service.suggest(target, left, right, intent, count, strength, target_syllables, profile)

    @server.tool(structured_output=True)
    def model_status() -> dict[str, Any]:
        """Inspect fixed model/tokenizer revision and lazy loading state without loading weights."""
        return service.model.status()

    return server
