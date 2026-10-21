"""Optional real-tensor checks; core CI remains usable without torch."""

import json
import struct

import pytest

torch = pytest.importorskip("torch")
safetensors = pytest.importorskip("safetensors.torch")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float16])
@pytest.mark.parametrize("shape", [(5, 7), (), (0, 3)])
def test_chunked_tensor_read_preserves_exact_bytes(tmp_path, dtype, shape):
    from mumble_rhyme.checkpoint import _stream_tensor

    tensor = torch.arange(max(1, torch.tensor(shape).prod().item()), dtype=torch.float32)
    if 0 in shape:
        tensor = tensor[:0]
    tensor = tensor.to(dtype).reshape(shape)
    path = tmp_path / "weights.safetensors"
    safetensors.save_file({"weight": tensor}, path)
    with path.open("rb") as source:
        header_length = struct.unpack("<Q", source.read(8))[0]
        info = json.loads(source.read(header_length))["weight"]
    result = _stream_tensor(path, 8 + header_length + info["data_offsets"][0],
                            shape, dtype, "cpu", chunk_bytes=16)
    assert torch.equal(result.reshape(-1).view(torch.uint8), tensor.reshape(-1).view(torch.uint8))


def test_chunked_read_reports_truncation(tmp_path):
    from mumble_rhyme.checkpoint import _stream_tensor

    path = tmp_path / "short"
    path.write_bytes(b"12")
    with pytest.raises(ValueError, match="Truncated"):
        _stream_tensor(path, 0, (10,), torch.float32, "cpu", chunk_bytes=16)
