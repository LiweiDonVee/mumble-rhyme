"""Read validated tensors with bounded staging and prepare smaller safetensors shards.

Windows safetensors 0.8 temporarily maps each whole file even with pread.
Only the layout changes; tensors, configuration and the source snapshot remain intact.
"""

import json
import shutil
import struct
from contextlib import ExitStack, contextmanager
from pathlib import Path

SHARD_BYTES = 256 * 1024**2
TRANSFER_BYTES = 16 * 1024**2


def _stream_tensor(path, offset, shape, dtype, device, chunk_bytes=TRANSFER_BYTES):
    """Copy validated tensor bytes with a bounded CPU buffer, including large embeddings."""
    import torch

    tensor = torch.empty(shape, dtype=dtype, device=device)
    destination = tensor.reshape(-1).view(torch.uint8)
    remaining = destination.numel()
    buffer = torch.empty(min(chunk_bytes, remaining), dtype=torch.uint8, device="cpu")
    view = memoryview(buffer.numpy())
    with path.open("rb", buffering=0) as source:
        source.seek(offset)
        written = 0
        while remaining:
            count = min(chunk_bytes, remaining)
            received = 0
            while received < count:
                amount = source.readinto(view[received:count])
                if not amount:
                    raise ValueError(f"Truncated tensor data in {path}")
                received += amount
            destination[written:written + count].copy_(buffer[:count])
            written += count
            remaining -= count
    return tensor


class _FullTensor:
    """Lazy Transformers-compatible view using safetensors' single-buffer read."""

    def __init__(self, reader, name, path=None, info=None, header_size=0, device="cpu"):
        self.reader, self.name = reader, name
        self.path, self.info, self.header_size, self.device = path, info, header_size, device

    def get_shape(self):
        return self.reader.get_slice(self.name).get_shape()

    def get_dtype(self):
        return self.reader.get_slice(self.name).get_dtype()

    def __getitem__(self, index):
        if self.device != "cpu" and self.info is not None:
            import torch

            types = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
            start, end = self.info["data_offsets"]
            if end - start > TRANSFER_BYTES and self.info["dtype"] in types:
                tensor = _stream_tensor(
                    self.path, self.header_size + start, self.get_shape(),
                    types[self.info["dtype"]], self.device,
                )
                return tensor if index is Ellipsis else tensor[index]
        tensor = self.reader.get_tensor(self.name)
        return tensor if index is Ellipsis else tensor[index]


@contextmanager
def full_tensor_state_dict(directory: Path, device="cpu"):
    """Keep validated readers open only while loading; do not retain tensor data.

    safetensors 0.8 pread get_slice()[...] creates a Rust buffer and a Python
    bytearray. get_tensor() reads straight into the bytearray, avoiding one
    tensor-sized temporary allocation. Large CUDA tensors instead stream through
    a 16 MiB buffer using offsets from the validated header. Source files are read-only.
    """
    from safetensors import safe_open

    state = {}
    with ExitStack() as stack:
        try:
            for path in sorted(directory.glob("*.safetensors")):
                reader = stack.enter_context(safe_open(str(path), framework="pt", device="cpu", backend="pread"))
                header, header_size = {}, 0
                if device != "cpu":
                    # safe_open above validates the safetensors layout first.
                    with path.open("rb") as source:
                        length = struct.unpack("<Q", source.read(8))[0]
                        header = json.loads(source.read(length))
                        header_size = 8 + length
                for name in reader.keys():  # noqa: SIM118 -- safe_open is not an iterable dict
                    state[name] = _FullTensor(reader, name, path, header.get(name), header_size, device)
            yield state
        finally:
            state.clear()


def reshard_checkpoint(source: Path, destination: Path, max_shard_bytes=SHARD_BYTES):
    from filelock import FileLock

    destination.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(destination) + ".lock"):
        index_path = destination / "model.safetensors.index.json"
        if index_path.exists():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            if all((destination / name).is_file() for name in index["weight_map"].values()):
                return destination
            index_path.unlink()
        destination.mkdir(exist_ok=True)
        entries = []
        for path in sorted(source.glob("*.safetensors")):
            with path.open("rb") as stream:
                header_size = struct.unpack("<Q", stream.read(8))[0]
                header = json.loads(stream.read(header_size))
            for name, info in header.items():
                if name != "__metadata__":
                    start, end = info["data_offsets"]
                    entries.append((name, info, path, 8 + header_size + start, end - start))
        if not entries:
            raise ValueError(f"No safetensors weights in {source}")
        groups, group, size = [], [], 0
        for entry in entries:
            if group and size + entry[4] > max_shard_bytes:
                groups.append(group)
                group, size = [], 0
            group.append(entry)
            size += entry[4]
        if group:
            groups.append(group)
        weight_map = {}
        for number, group in enumerate(groups, 1):
            filename = f"model-{number:05d}-of-{len(groups):05d}.safetensors"
            header, offset = {"__metadata__": {"format": "pt"}}, 0
            for name, info, _, _, length in group:
                header[name] = {**info, "data_offsets": [offset, offset + length]}
                weight_map[name] = filename
                offset += length
            raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
            raw += b" " * (-len(raw) % 8)
            with (destination / filename).open("wb") as output:
                output.write(struct.pack("<Q", len(raw)))
                output.write(raw)
                for _, _, path, start, remaining in group:
                    with path.open("rb") as stream:
                        stream.seek(start)
                        while remaining:
                            block = stream.read(min(1024**2, remaining))
                            if not block:
                                raise ValueError(f"Truncated tensor data in {path}")
                            output.write(block)
                            remaining -= len(block)
        for path in source.iterdir():
            if (path.is_file() and path.suffix in {".json", ".jinja", ".txt"}
                    and path.name != "model.safetensors.index.json"):
                shutil.copyfile(path, destination / path.name)
        index = {"metadata": {"total_size": sum(e[4] for e in entries)}, "weight_map": weight_map}
        # Publish the index last so an interrupted conversion is never reused.
        temporary_index = destination / "index.json.partial"
        temporary_index.write_text(json.dumps(index, indent=2), encoding="utf-8")
        temporary_index.replace(index_path)
        return destination
