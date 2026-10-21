import pytest

from mumble_rhyme.service import LyricService


class NoModel:
    def search(self, **kwargs):
        raise AssertionError("Invalid request should fail before model load")


@pytest.mark.parametrize("kwargs", [
    {"target":""}, {"target":"night","beam_width":0},
    {"target":"night","profile":{"unknown":1}},
    {"target":"night","microbatch_size":0}, {"target":"night","right":"ignored"},
])
def test_search_rejects_invalid_inputs_before_model_load(kwargs):
    with pytest.raises((ValueError, TypeError)):
        LyricService(model=NoModel()).search(**kwargs)


def test_search_shares_custom_rhyme_rules_and_discloses_language_mode():
    class Fake:
        def search(self, **kwargs):
            assert kwargs["rhyme_weight"]("sit") == pytest.approx(service.compare("sit","set",profile)["weight"])
            assert kwargs["beam_width"] == 20 and kwargs["branch_top_k"] == 20
            return {"candidates":[{"text":"sit"}],"search":{"exhaustive":False}}

        def status(self):
            return {"model_id":"test"}

    profile={"vowel_rules":[{"sounds":["IH","EH"],"weight":.98}]}
    service=LyricService(model=Fake())
    result=service.search("set",left="I just",profile=profile)
    assert result["scoring_mode"] == "token_path_search"
    assert result["candidates"][0]["rhyme"]["alignment"][0]["source"] == "custom_rule"
