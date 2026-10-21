import pytest

from mumble_rhyme.phonetics import RhymeEngine
from mumble_rhyme.profile import Profile


def engine(**kwargs):
    return RhymeEngine(Profile(**kwargs))


def test_every_comparison_is_soft_including_unknown_words():
    e = engine()
    assert e.compare("light", "night")["weight"] == pytest.approx(1)
    for word in ["orange", "cat", "zzxyq"]:
        result = e.compare(word, "night")
        assert 0 < result["weight"] <= 1
    assert e.compare("zzxyq", "night")["unknown_words"] == ["zzxyq"]


def test_symbolic_pair_override_is_symmetric_and_not_transitive():
    e = engine(
        aliases={"i": "IH", "e": "EH", "a": "AE"},
        vowel_rules=[
            {"sounds": ["i", "e"], "weight": 0.97},
            {"sounds": ["e", "a"], "weight": 0.97},
        ],
    )
    assert e.vowel_weight("IH", "EH", 1)[0] == 0.97
    assert e.vowel_weight("EH", "IH", 1)[0] == 0.97
    assert e.vowel_weight("IH", "AE", 1)[0] < 0.97
    assert e.compare("sit", "set")["weight"] > engine().compare("sit", "set")["weight"]


def test_position_specific_rule_overrides_generic_rule():
    e = engine(
        vowel_rules=[
            {"sounds": ["IH", "EH"], "weight": 0.9},
            {"sounds": ["IH", "EH"], "weight": 0.3, "position": 2},
        ]
    )
    assert e.vowel_weight("IH", "EH", 1)[0] == 0.9
    assert e.vowel_weight("IH", "EH", 2)[0] == 0.3


def test_directional_rule_does_not_apply_backwards():
    e = engine(vowel_rules=[{"sounds": ["IH", "EH"], "weight": 0.99, "symmetric": False}])
    assert e.vowel_weight("IH", "EH", 1)[0] == 0.99
    assert e.vowel_weight("EH", "IH", 1)[0] < 0.99


def test_alias_by_example_and_mumble_pronunciation_override():
    e = engine(
        aliases={"i": {"vowel_of": "sit"}},
        pronunciations={"luv": ["L AH1 V"]},
        vowel_rules=[{"sounds": ["i", "EH"], "weight": 0.96}],
    )
    assert e.vowel_weight("IH", "EH", 1)[0] == 0.96
    result = e.compare("luv", "love")
    assert result["weight"] == pytest.approx(1)
    assert result["unknown_words"] == []


def test_multiple_syllables_and_pronunciations_are_considered():
    e = engine(syllables=2)
    result = e.compare("in motion", "emotion")
    assert len(result["alignment"]) == 2
    assert result["weight"] == pytest.approx(1)
    result = engine().compare("read", "red")
    assert result["weight"] == pytest.approx(1)
    assert result["pronunciation_pairs_considered"] >= 2


def test_unknown_word_inside_phrase_is_reported_without_discarding_known_ending():
    result = engine().compare("zzxyq night", "light")
    assert result["unknown_words"] == ["zzxyq"]
    assert result["weight"] == pytest.approx(1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"vowel_rules": [{"sounds": ["IH", "EH"], "weight": 0}]},
        {"vowel_rules": [{"sounds": ["x", "EH"], "weight": 0.8}]},
        {"pronunciations": {"foo": ["BOGUS1"]}},
        {"syllables": 0},
        {"unknown_weight": 0},
        {"typo_setting": 0.9},
    ],
)
def test_invalid_profiles_fail_clearly(kwargs):
    with pytest.raises(ValueError):
        engine(**kwargs)
