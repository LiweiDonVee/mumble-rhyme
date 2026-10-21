from mumble_rhyme.model import assemble_completion, scored_token_positions


def test_token_crossing_left_boundary_is_scored_and_suffix_is_included():
    # A BPE token can span left-context characters and candidate characters.
    assert scored_token_positions([(0, 3), (3, 7), (7, 10)], 5) == [1, 2]


def test_prompt_only_and_empty_offsets_are_not_scored():
    assert scored_token_positions([(0, 3), (0, 0), (3, 6), (6, 7)], 6) == [3]


def test_completion_inserts_word_spaces_but_preserves_punctuation():
    assert assemble_completion("I feel", "alone", "tonight") == ("I feel alone tonight", 7)
    assert assemble_completion("I feel ", "alone", ", again") == ("I feel alone, again", 7)
