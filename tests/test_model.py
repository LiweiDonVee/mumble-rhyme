from mumble_rhyme.model import assemble_completion, scored_token_positions


def test_each_model_has_its_own_pinned_weights_and_tokenizer():
    from mumble_rhyme.model import LocalModel

    default = LocalModel().status()
    small = LocalModel(model="4b").status()
    assert default["model_id"] == "Qwen/Qwen3.5-9B"
    assert default["revision"] == "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    assert small["model_id"] == "Qwen/Qwen3.5-4B"
    assert small["revision"] == "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    for status in (default, small):
        assert status["revision"] == status["tokenizer_revision"]
        assert status["loaded"] is False
        assert status["automatic_model_substitution"] is False


def test_unknown_model_is_rejected_without_importing_model_runtime():
    import pytest

    from mumble_rhyme.model import LocalModel

    with pytest.raises(ValueError, match="model must be"):
        LocalModel(model="not-a-supported-model")


def test_four_b_download_uses_its_exact_snapshot(monkeypatch):
    import sys
    from types import SimpleNamespace

    from mumble_rhyme.model import LocalModel

    calls = []

    def download(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return "snapshot"

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    assert LocalModel(model="4b").download() == "snapshot"
    assert calls[0][0] == "Qwen/Qwen3.5-4B"
    assert calls[0][1]["revision"] == "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def test_four_b_loader_uses_matching_model_and_tokenizer(monkeypatch):
    import sys
    from types import SimpleNamespace

    from mumble_rhyme.model import LocalModel

    calls = []

    class Tokenizer:
        def __len__(self):
            return 12

    class Model:
        def eval(self):
            return self

        def get_input_embeddings(self):
            return SimpleNamespace(num_embeddings=12)

    def load_tokenizer(repo, **kwargs):
        calls.append(("tokenizer", repo, kwargs))
        return Tokenizer()

    def load_model(repo, **kwargs):
        calls.append(("model", repo, kwargs))
        return Model()

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(bfloat16="bf16"))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer),
        BitsAndBytesConfig=None,
        Qwen3_5ForConditionalGeneration=SimpleNamespace(from_pretrained=load_model),
    ))
    monkeypatch.setattr(sys, "platform", "linux")
    adapter = LocalModel(quantization="none", model="4b")
    adapter.load()
    adapter.load()  # Loaded instances reuse the same model.
    assert len(calls) == 2
    for _, repo, kwargs in calls:
        assert repo == "Qwen/Qwen3.5-4B"
        assert kwargs["revision"] == "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
        assert kwargs["trust_remote_code"] is False
    assert adapter.status()["loaded"] is True


def test_token_crossing_left_boundary_is_scored_and_suffix_is_included():
    # A BPE token can span left-context characters and candidate characters.
    assert scored_token_positions([(0, 3), (3, 7), (7, 10)], 5) == [1, 2]


def test_prompt_only_and_empty_offsets_are_not_scored():
    assert scored_token_positions([(0, 3), (0, 0), (3, 6), (6, 7)], 6) == [3]


def test_completion_inserts_word_spaces_but_preserves_punctuation():
    assert assemble_completion("I feel", "alone", "tonight") == ("I feel alone tonight", 7)
    assert assemble_completion("I feel ", "alone", ", again") == ("I feel alone, again", 7)


def test_unload_releases_owned_references_and_unused_cuda_cache(monkeypatch):
    import sys
    import weakref
    from types import SimpleNamespace

    from mumble_rhyme.model import LocalModel

    events = []
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        is_initialized=lambda: True, empty_cache=lambda: events.append("empty_cache"),
        synchronize=lambda: events.append("synchronize"),
    ), _C=SimpleNamespace(_cuda_clearCublasWorkspaces=lambda: events.append("workspaces"))))

    class Owned:
        pass

    adapter = LocalModel()
    adapter.model, adapter.tokenizer = Owned(), Owned()
    owned = weakref.ref(adapter.model)
    adapter.unload(release_runtime_cache=True)
    assert owned() is None
    assert adapter.tokenizer is None
    assert adapter.status()["loaded"] is False
    assert events == ["synchronize", "workspaces", "empty_cache"]


def test_default_unload_does_not_purge_process_wide_workspaces(monkeypatch):
    import sys
    from types import SimpleNamespace

    from mumble_rhyme.model import LocalModel

    events = []
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        is_initialized=lambda: True, empty_cache=lambda: events.append("empty_cache"),
        synchronize=lambda: events.append("synchronize"),
    ), _C=SimpleNamespace(_cuda_clearCublasWorkspaces=lambda: events.append("workspaces"))))
    LocalModel().unload()
    assert events == ["empty_cache"]


def test_adapters_share_runtime_lock_for_process_wide_cuda_cleanup():
    from mumble_rhyme.model import LocalModel

    assert LocalModel().lock is LocalModel(model="4b").lock


def test_failed_load_drops_partial_model_even_if_error_is_kept(monkeypatch):
    import sys
    import weakref
    from types import SimpleNamespace

    import pytest

    from mumble_rhyme.model import LocalModel

    partials, events = [], []

    class Partial:
        pass

    def fail(*args, **kwargs):
        def inner():
            partial = Partial()
            partials.append(weakref.ref(partial))
            raise MemoryError("injected while model owns memory")

        try:
            inner()
        except MemoryError as cause:
            raise RuntimeError("wrapped load failure") from cause

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(bfloat16="bf16", cuda=SimpleNamespace(
        is_initialized=lambda: True, empty_cache=lambda: events.append("empty_cache"),
    )))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: object()),
        BitsAndBytesConfig=None,
        Qwen3_5ForConditionalGeneration=SimpleNamespace(from_pretrained=fail),
    ))
    monkeypatch.setattr(sys, "platform", "linux")
    adapter = LocalModel(quantization="none")
    with pytest.raises(RuntimeError) as captured:
        adapter.load()
    assert captured.value is not None  # Holding the error must not retain the partial model.
    assert partials[0]() is None
    assert adapter.model is None and adapter.tokenizer is None
    assert events == ["empty_cache"]
