import pytest

from mumble_rhyme.service import LyricService


@pytest.mark.parametrize("target,common", [
    ("mine", {"line", "fine", "sign", "wine", "time", "shine", "design"}),
    ("fall", {"all", "call", "small", "wall", "ball", "tall", "hall"}),
])
def test_balanced_search_surfaces_common_useful_vocabulary(target, common):
    rows = LyricService().find(target, limit=10)["candidates"]
    assert len(common.intersection(r["text"] for r in rows)) >= 4
    assert not {"ein", "gyn", "fein", "kyne", "pol", "tol", "wal"}.intersection(
        r["text"] for r in rows
    )
    assert all("frequency_zipf" in r and "lookup_score" in r for r in rows)


def test_frequency_adjustment_does_not_modify_custom_rhyme_weight():
    service = LyricService()
    profile = {"vowel_rules": [{"sounds": ["IH", "EH"], "weight": 0.98}]}
    rows = service.find("set", limit=100, profile=profile)["candidates"]
    row = next(r for r in rows if r["text"] == "sit")
    assert row["rhyme"] == service.compare("sit", "set", profile)


def test_phonetic_mode_keeps_all_spellings_and_ignores_frequency():
    rows = LyricService().find("mine", 30, mode="phonetic")["candidates"]
    assert all(r["lookup_score"] == pytest.approx(r["rhyme"]["weight"]) for r in rows)
    assert "ein" in [r["text"] for r in rows]


def test_homophones_are_exposed_as_expandable_alternatives():
    result = LyricService().find("mine", limit=100)
    assert result["homophones_collapsed"] is True
    assert any(r["alternatives"] for r in result["candidates"])
    assert result["vocabulary_evaluated"] > 10000
