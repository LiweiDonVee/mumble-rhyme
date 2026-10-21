import json
import struct


def test_lazy_full_tensor_reader_avoids_slice_materialization_and_closes_files(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    from mumble_rhyme.checkpoint import full_tensor_state_dict

    (tmp_path / "model.safetensors").touch()
    events = []

    class Slice:
        def get_shape(self):
            return [2]

        def get_dtype(self):
            return "BF16"

        def __getitem__(self, index):
            raise AssertionError("The duplicate-buffer slice path must not be used")

    class Reader:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            events.append("closed")

        def keys(self):
            return ["weight"]

        def get_slice(self, name):
            return Slice()

        def get_tensor(self, name):
            events.append("read")
            return [10, 20]

    monkeypatch.setitem(sys.modules, "safetensors", SimpleNamespace(safe_open=lambda *a, **k: Reader()))
    with full_tensor_state_dict(tmp_path) as state:
        view = state["weight"]
        assert view.get_shape() == [2]
        assert view.get_dtype() == "BF16"
        assert not events
        assert view[...] == [10, 20]
        assert events == ["read"]
    assert events == ["read", "closed"]
    assert state == {}


def test_lazy_reader_closes_on_load_failure(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    import pytest

    from mumble_rhyme.checkpoint import full_tensor_state_dict

    (tmp_path / "model.safetensors").touch()
    closed = []

    class Reader:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            closed.append(True)

        def keys(self):
            return []

    monkeypatch.setitem(sys.modules, "safetensors", SimpleNamespace(safe_open=lambda *a, **k: Reader()))
    with pytest.raises(MemoryError), full_tensor_state_dict(tmp_path):
        raise MemoryError("injected failure")
    assert closed == [True]


def write_fixture(path, tensors):
    header = {"__metadata__": {"format": "pt"}}
    payload = b""
    for name, data in tensors.items():
        header[name] = {"dtype": "U8", "shape": [len(data)],
                        "data_offsets": [len(payload), len(payload) + len(data)]}
        payload += data
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)


def read_fixture(path):
    raw = path.read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8+n])
    return {name: (row["dtype"], row["shape"], raw[8+n+row["data_offsets"][0]:8+n+row["data_offsets"][1]])
            for name, row in header.items() if name != "__metadata__"}


def test_reshard_preserves_every_tensor_and_bounds_temporary_file_mapping(tmp_path):
    from mumble_rhyme.checkpoint import reshard_checkpoint

    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text('{"model_type":"qwen3_5"}')
    write_fixture(source / "first.safetensors", {"a": b"0123", "big": b"0123456789"})
    write_fixture(source / "second.safetensors", {"b": b"abcd", "c": b"efgh"})
    dest = tmp_path / "small"
    result = reshard_checkpoint(source, dest, max_shard_bytes=8)
    assert result == dest
    index = json.loads((dest / "model.safetensors.index.json").read_text())
    merged = {}
    for name in set(index["weight_map"].values()):
        tensors = read_fixture(dest / name)
        merged.update(tensors)
        size = sum(len(value[2]) for value in tensors.values())
        assert size <= 8 or (len(tensors) == 1 and size == 10)
    expected = read_fixture(source / "first.safetensors") | read_fixture(source / "second.safetensors")
    assert merged == expected
    assert set(index["weight_map"]) == set(expected)
    assert index["metadata"]["total_size"] == 22
    assert (dest / "config.json").read_bytes() == (source / "config.json").read_bytes()
    assert len(list(source.glob("*.safetensors"))) == 2
    # Reuse a completed cache without rewriting 19 GB every time a tool starts.
    stamp = (dest / "model.safetensors.index.json").stat().st_mtime_ns
    assert reshard_checkpoint(source, dest, max_shard_bytes=8) == dest
    assert (dest / "model.safetensors.index.json").stat().st_mtime_ns == stamp


def test_incomplete_reshard_is_rebuilt_instead_of_loaded(tmp_path):
    from mumble_rhyme.checkpoint import reshard_checkpoint

    source = tmp_path / "source"
    source.mkdir()
    write_fixture(source / "weights.safetensors", {"a": b"abcd"})
    dest = tmp_path / "small"
    dest.mkdir()
    (dest / "model-00001-of-00001.safetensors").write_bytes(b"interrupted")
    reshard_checkpoint(source, dest, max_shard_bytes=8)
    assert read_fixture(dest / "model-00001-of-00001.safetensors")["a"][2] == b"abcd"


def test_interrupted_repair_invalidates_previously_published_index(tmp_path, monkeypatch):
    from pathlib import Path

    import pytest

    from mumble_rhyme.checkpoint import reshard_checkpoint

    source, dest = tmp_path / "source", tmp_path / "small"
    source.mkdir()
    write_fixture(source / "weights.safetensors", {"a": b"abcd"})
    reshard_checkpoint(source, dest, max_shard_bytes=8)
    shard = dest / "model-00001-of-00001.safetensors"
    shard.unlink()
    original_open = Path.open

    def interrupt_write(path, mode="r", *args, **kwargs):
        if path == shard and mode == "wb":
            original_open(path, mode).close()
            raise OSError("interrupted repair")
        return original_open(path, mode, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", interrupt_write)
        with pytest.raises(OSError, match="interrupted repair"):
            reshard_checkpoint(source, dest, max_shard_bytes=8)
    assert not (dest / "model.safetensors.index.json").exists()
    reshard_checkpoint(source, dest, max_shard_bytes=8)
    assert read_fixture(shard)["a"][2] == b"abcd"
