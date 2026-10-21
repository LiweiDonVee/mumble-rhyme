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


def test_cuda_runtime_is_prepared_before_weights_are_loaded(monkeypatch):
    import sys
    from types import SimpleNamespace

    from mumble_rhyme.model import LocalModel

    events = []

    class Model:
        def eval(self):
            return self

        def get_input_embeddings(self):
            return SimpleNamespace(num_embeddings=12)

    class Tokenizer:
        def __len__(self):
            return 12

    def load_weights(*args, **kwargs):
        events.append("weights")
        return Model()

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        bfloat16="bf16", cuda=SimpleNamespace(
            is_available=lambda: True, empty_cache=lambda: events.append("unused_loading_cache"),
        ),
    ))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer()),
        BitsAndBytesConfig=lambda **kwargs: kwargs,
        Qwen3_5ForConditionalGeneration=SimpleNamespace(from_pretrained=load_weights),
    ))
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(LocalModel, "_prepare_cuda", lambda self: events.append("runtime"), raising=False)
    adapter = LocalModel()
    adapter.load()
    assert events == ["runtime", "weights", "unused_loading_cache"]


def test_first_generate_resource_failure_releases_weights_and_can_retry(monkeypatch):
    import sys
    import weakref
    from contextlib import nullcontext
    from types import SimpleNamespace

    import pytest

    from mumble_rhyme.model import LocalModel

    events, partials = [], []

    class Inputs(dict):
        input_ids = SimpleNamespace(shape=(1, 10))

        def to(self, device):
            return self

    class Tokenizer:
        def apply_chat_template(self, *a, **k):
            return "prompt"

        def __call__(self, *a, **k):
            return Inputs()

        def decode(self, *a, **k):
            return '["fading into the night"]'

    class Temporary:
        pass

    class Output:
        def __getitem__(self, index):
            return []

    class Model:
        device = "cuda:0"

        def __init__(self, fail):
            self.fail = fail

        def generate(self, **kwargs):
            if self.fail:
                temporary = Temporary()
                partials.append(weakref.ref(temporary))
                raise RuntimeError("CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate(handle)")
            return Output()

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        inference_mode=nullcontext,
        cuda=SimpleNamespace(is_initialized=lambda: True, empty_cache=lambda: events.append("released")),
    ))
    adapter = LocalModel()
    adapter.model, adapter.tokenizer = Model(True), Tokenizer()
    with pytest.raises(RuntimeError, match="CUBLAS_STATUS_ALLOC_FAILED") as captured:
        adapter.generate("the city lights are", "", "深夜城市的孤独感", "light", 3)
    assert adapter.status()["loaded"] is False
    assert adapter.tokenizer is None
    assert partials[0]() is None  # Even if the caller keeps the error.
    assert "released" in str(captured.value).lower()
    assert events == ["released"]
    adapter.model, adapter.tokenizer = Model(False), Tokenizer()
    assert adapter.generate("the city lights are", "", "深夜城市的孤独感", "light", 3)[0]


def test_validation_error_does_not_unload_model(monkeypatch):
    import sys
    from contextlib import nullcontext
    from types import SimpleNamespace

    import pytest

    from mumble_rhyme.model import LocalModel

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    adapter = LocalModel()
    adapter.model = object()
    with pytest.raises(ValueError), adapter._inference():
        raise ValueError("input too long")
    assert adapter.status()["loaded"] is True
