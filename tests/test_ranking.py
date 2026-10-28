import pytest

from mumble_rhyme.service import LyricService


def test_strength_zero_preserves_language_model_order_and_all_candidates():
    service = LyricService()
    result = service.rank(
        ["night", "orange", "zzxyq"], "light", strength=0, model_scores=[-4, -1, -2], use_model=False
    )
    assert [r["text"] for r in result["candidates"]] == ["orange", "zzxyq", "night"]
    assert sum(r["rank_weight"] for r in result["candidates"]) == pytest.approx(1)
    assert all(r["rank_weight"] > 0 for r in result["candidates"])
    assert result["scoring_mode"] == "supplied_scores"


def test_strong_semantic_preference_can_outweigh_exact_rhyme():
    result = LyricService().rank(["night", "heart"], "light", model_scores=[-9, -1], use_model=False)
    assert result["candidates"][0]["text"] == "heart"
    assert len(result["candidates"]) == 2


def test_unknown_candidate_retained_and_phonetic_only_label_is_explicit():
    result = LyricService().rank(["zzxyq", "night"], "light", use_model=False)
    assert result["scoring_mode"] == "phonetic_only"
    assert {r["text"] for r in result["candidates"]} == {"zzxyq", "night"}
    assert all(r["lm_mean_logprob"] is None for r in result["candidates"])


def test_soft_syllable_penalty_and_per_request_profile():
    result = LyricService().rank(["night", "in the night"], "light", use_model=False, target_syllables=1)
    assert result["candidates"][0]["text"] == "night"
    assert len(result["candidates"]) == 2
    e = LyricService()
    result = e.rank(
        ["sit"], "set", use_model=False, profile={"vowel_rules": [{"sounds": ["IH", "EH"], "weight": 0.99}]}
    )
    assert result["candidates"][0]["rhyme"]["weight"] > e.compare("sit", "set")["weight"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model_scores": [-1]},
        {"model_scores": [float("nan"), -1]},
        {"strength": -1},
    ],
)
def test_invalid_ranking_requests_rejected(kwargs):
    with pytest.raises(ValueError):
        LyricService().rank(["night", "light"], "fight", use_model=False, **kwargs)


@pytest.mark.parametrize("candidates", ["night", {"text": "night"}])
def test_rank_rejects_non_sequence_candidate_containers(candidates):
    with pytest.raises(TypeError, match="candidates"):
        LyricService().rank(candidates, "light", use_model=False)


def test_dictionary_search_discloses_budget_and_scores_near_rhymes():
    result = LyricService().find("night", limit=8)
    assert result["vocabulary_evaluated"] > 10000
    assert len(result["candidates"]) == 8
    assert all(r["rhyme"]["weight"] > 0 for r in result["candidates"])


@pytest.mark.parametrize(
    "method,kwargs",
    [
        ("rank", {"candidates": ["night"], "target": ""}),
        ("rank", {"candidates": ["night"], "target": "light", "profile": {"unknown_key": 1}}),
        ("suggest", {"target": ""}),
        ("suggest", {"target": "night", "target_syllables": 0}),
    ],
)
def test_invalid_inputs_fail_before_any_model_load(method, kwargs):
    service = LyricService()
    with pytest.raises(ValueError):
        getattr(service, method)(**kwargs)
    assert not service.model.status()["loaded"]


def test_meter_uses_the_selected_rhyme_pronunciation():
    row = LyricService().rank(["fire"], "night", use_model=False, target_syllables=1)["candidates"][0]
    assert row["rhyme"]["candidate_phones"] == ["F", "AY1", "R"]
    assert row["rhyme"]["syllable_count"] == 1
    assert row["syllable_weight"] == 1


def test_selected_pronunciation_count_does_not_depend_on_override_order():
    scores = []
    for readings in [["F AY1 ER0", "F AY1 R"], ["F AY1 R", "F AY1 ER0"]]:
        service = LyricService()
        row = service.rank(
            ["fyre"],
            "night",
            use_model=False,
            target_syllables=1,
            profile={"pronunciations": {"fyre": readings}},
        )["candidates"][0]
        assert row["rhyme"]["syllable_count"] == 1
        scores.append(row["adjusted_score"])
    assert scores[0] == pytest.approx(scores[1])


def test_unknown_prefix_does_not_claim_a_complete_syllable_count():
    rhyme = LyricService().compare("zzxyq fire", "night")
    assert rhyme["candidate_phones"] == ["F", "AY1", "R"]
    assert rhyme["syllable_count"] is None
