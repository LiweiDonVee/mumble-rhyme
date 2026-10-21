import math

import pytest


def test_search_keeps_real_path_probability_and_soft_rhyme_separate():
    from mumble_rhyme.search import beam_search

    table = {(): [(0, math.log(0.6)), (1, math.log(0.4))],
             (0,): [(2, math.log(0.5))], (1,): [(2, math.log(0.8))]}
    decode = lambda ids: "".join({0: " light", 1: " heart", 2: "."}[i] for i in ids)
    result = beam_search(lambda paths: [table[p] for p in paths], decode,
                         lambda text: 1 if "light" in text else .5,
                         beam_width=20, branch_top_k=20, max_new_tokens=2, min_new_tokens=1)
    rows = result["candidates"]
    assert rows[0]["text"] == "light."
    assert rows[0]["sequence_probability"] == pytest.approx(.3)
    heart = next(r for r in rows if r["text"] == "heart.")
    assert heart["sequence_probability"] == pytest.approx(.32)
    assert heart["adjusted_log_weight"] == pytest.approx(math.log(.32) + 2 * math.log(.5))
    assert sum(r["rank_weight"] for r in rows) == pytest.approx(1)


def test_search_does_not_treat_token_fragments_as_complete_words():
    from mumble_rhyme.search import beam_search

    seen = []
    parts = {0: " un", 1: "avail", 2: "able", 3: "."}
    def next_tokens(paths):
        return [[(len(p), -.1)] for p in paths]
    def rhyme(text):
        seen.append(text)
        return .8
    result = beam_search(next_tokens, lambda ids: "".join(parts[i] for i in ids), rhyme,
                         max_new_tokens=4, min_new_tokens=1)
    assert result["candidates"][0]["text"] == "unavailable."
    assert seen and all("unavail" not in text or "unavailable" in text for text in seen)
    assert "un" not in seen


def test_search_discloses_length_cutoff_and_preserves_unknown_ending():
    from mumble_rhyme.search import beam_search

    result = beam_search(lambda paths: [[(0, -.2)] for _ in paths], lambda ids: " zzxyq" * len(ids),
                         lambda text: .65, max_new_tokens=2, min_new_tokens=1)
    row = result["candidates"][0]
    assert row["finished_by"] == "max_tokens"
    assert row["word_boundary_confirmed"] is False
    assert result["search"]["length_limit_reached"] is True
    assert row["rhyme_weight"] > 0


def test_beam_width_bounds_frontier_without_reducing_branch_expansion():
    from mumble_rhyme.search import beam_search

    sizes = []
    def predict(paths):
        sizes.append(len(paths))
        return [[(i, -math.log(20)) for i in range(20)] for _ in paths]
    result = beam_search(predict, lambda ids: "".join(f" w{i}" for i in ids), lambda text: 1,
                         max_new_tokens=3, beam_width=20, branch_top_k=20)
    assert sizes == [1, 20, 20]
    assert result["search"]["expanded_edges"] == 820
    assert result["search"]["pruned_paths"] > 0


def test_strength_zero_and_explicit_length_normalization():
    from mumble_rhyme.search import beam_search

    table = {(): [(0, -.1), (1, -.2)], (0,): [(2, -.1)], (1,): [(2, -.1)]}
    decode = lambda ids: "".join({0:" cat", 1:" light", 2:"."}[i] for i in ids)
    r = beam_search(lambda paths: [table[p] for p in paths], decode,
                    lambda text: .01 if "cat" in text else 1,
                    max_new_tokens=2, min_new_tokens=1, strength=0, length_normalization=1)
    assert r["candidates"][0]["text"] == "cat."
    assert r["candidates"][0]["sequence_logprob"] == pytest.approx(-.2)
    assert r["candidates"][0]["adjusted_log_weight"] == pytest.approx(-.1)


def test_eos_probability_is_included_and_early_stop_disclosed():
    from mumble_rhyme.search import beam_search

    table = {(): [(0, math.log(.2))], (0,): [(9, math.log(.3))]}
    result = beam_search(lambda paths: [table[p] for p in paths], lambda ids: " wine",
                         lambda text: 1, eos_ids=[9], max_new_tokens=5)
    row = result["candidates"][0]
    assert row["token_ids"] == [0, 9]
    assert row["sequence_probability"] == pytest.approx(.06)
    assert row["finished_by"] == "eos"
    assert result["search"]["length_limit_reached"] is False


@pytest.mark.parametrize("kwargs", [{"beam_width":0}, {"branch_top_k":0}, {"max_new_tokens":0},
    {"strength":-1}, {"length_normalization":float("nan")}, {"min_new_tokens":5,"max_new_tokens":3}])
def test_invalid_search_preferences(kwargs):
    from mumble_rhyme.search import validate_search

    with pytest.raises(ValueError):
        validate_search(**kwargs)
