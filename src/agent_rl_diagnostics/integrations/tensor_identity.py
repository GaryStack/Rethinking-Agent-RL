"""Streaming tensor-content identity independent of HF shard/header layout.

Read-only, stdlib-only subset of the official safetensors format. Supports ordinary
byte-aligned dtypes; packed/sub-byte quantization is deliberately rejected.
Format: https://github.com/safetensors/safetensors#format
"""

import hashlib
import json
import math
import struct
from pathlib import Path

from ..artifacts.io import canonical_hash, read_json
from ..contracts.schema import SchemaError

WIDTHS = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SchemaError("duplicate safetensors header key")
        result[key] = value
    return result


def tensor_identity(directory: Path) -> dict:
    index = directory / "model.safetensors.index.json"
    document = read_json(index) if index.is_file() else None
    if document is not None and not isinstance(document, dict):
        raise SchemaError("tensor index must be an object")
    mapping = document.get("weight_map", {}) if document is not None else None
    if mapping is not None and (not isinstance(mapping, dict) or not mapping):
        raise SchemaError("missing tensor index")
    if mapping is not None and any(not isinstance(v, str) for v in mapping.values()):
        raise SchemaError("invalid tensor shard name")
    files = sorted(set(mapping.values())) if mapping is not None else ["model.safetensors"]
    tensors = {}
    for name in files:
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not name.endswith(".safetensors")
        ):
            raise SchemaError("unsafe tensor shard name")
        path = directory / name
        size = path.stat().st_size
        with path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise SchemaError("truncated safetensors header")
            length = struct.unpack("<Q", prefix)[0]
            if not 1 <= length <= min(size - 8, 100_000_000):
                raise SchemaError("invalid safetensors header length")
            raw = stream.read(length)
            if not raw.startswith(b"{"):
                raise SchemaError("invalid safetensors JSON header")
            header = json.loads(raw, object_pairs_hook=unique_pairs)
            if not isinstance(header, dict):
                raise SchemaError("safetensors header must be an object")
            metadata = header.pop("__metadata__", {})
            if not isinstance(metadata, dict) or any(
                not isinstance(v, str) for v in metadata.values()
            ):
                raise SchemaError("invalid safetensors metadata")
            bounds = []
            for key, tensor in header.items():
                if not isinstance(tensor, dict) or set(tensor) != {
                    "dtype",
                    "shape",
                    "data_offsets",
                }:
                    raise SchemaError("invalid tensor header")
                dtype, shape, offsets = tensor["dtype"], tensor["shape"], tensor["data_offsets"]
                if not isinstance(dtype, str) or dtype not in WIDTHS:
                    raise SchemaError("unsupported tensor dtype (packed formats are not inferred)")
                if not isinstance(shape, list) or any(type(x) is not int or x < 0 for x in shape):
                    raise SchemaError("invalid tensor shape")
                if (
                    not isinstance(offsets, list)
                    or len(offsets) != 2
                    or any(type(x) is not int for x in offsets)
                ):
                    raise SchemaError("invalid tensor offsets")
                begin, end = offsets
                if (
                    not 0 <= begin <= end <= size - 8 - length
                    or end - begin != math.prod(shape) * WIDTHS[dtype]
                ):
                    raise SchemaError("tensor shape/byte range mismatch")
                if key in tensors or (mapping is not None and mapping.get(key) != name):
                    raise SchemaError("duplicate tensor or shard-index mismatch")
                bounds.append((begin, end, key, tensor))
            cursor = 0
            for begin, end, key, tensor in sorted(bounds):
                if end > begin:
                    if begin != cursor:
                        raise SchemaError("safetensors data overlap or gap")
                    cursor = end
                stream.seek(8 + length + begin)
                remaining, digest = end - begin, hashlib.sha256()
                while remaining:
                    chunk = stream.read(min(remaining, 1024 * 1024))
                    if not chunk:
                        raise SchemaError("truncated tensor data")
                    digest.update(chunk)
                    remaining -= len(chunk)
                tensors[key] = {
                    "dtype": tensor["dtype"],
                    "shape": tensor["shape"],
                    "sha256": digest.hexdigest(),
                }
            if cursor != size - 8 - length:
                raise SchemaError("unindexed safetensors data")
    if not tensors or (mapping is not None and set(tensors) != set(mapping)):
        raise SchemaError("missing tensor contents")
    return {
        "schema_version": "ard-tensor-identity-v1",
        "fingerprint": canonical_hash(tensors),
        "tensors": tensors,
    }
