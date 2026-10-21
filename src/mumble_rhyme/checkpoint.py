"""Stream unchanged tensor bytes into smaller standard safetensors shards.

Windows safetensors 0.8 temporarily maps each whole file even with pread.
Only the layout changes; tensors, configuration and the source snapshot remain intact.
"""

import json
import shutil
import struct
from pathlib import Path

SHARD_BYTES = 256 * 1024**2


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
