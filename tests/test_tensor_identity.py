import json
import struct

import pytest

from agent_rl_diagnostics.contracts.schema import SchemaError
from agent_rl_diagnostics.integrations.tensor_identity import tensor_identity


def write_tensors(path, pairs, metadata="fixture"):
    header, data = {"__metadata__": {"source": metadata}}, b""
    for name, raw in pairs:
        header[name] = {
            "dtype": "U8",
            "shape": [len(raw)],
            "data_offsets": [len(data), len(data) + len(raw)],
        }
        data += raw
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)


def test_tensor_content_hash_ignores_metadata_order_and_shard_layout(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    write_tensors(a / "model.safetensors", [("left", b"123"), ("right", b"456")])
    write_tensors(b / "model-1.safetensors", [("right", b"456")], metadata="different-header")
    write_tensors(b / "model-2.safetensors", [("left", b"123")])
    (b / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"right": "model-1.safetensors", "left": "model-2.safetensors"}})
    )
    assert tensor_identity(a) == tensor_identity(b)
    write_tensors(b / "model-2.safetensors", [("left", b"124")])
    assert tensor_identity(a)["fingerprint"] != tensor_identity(b)["fingerprint"]


@pytest.mark.parametrize(
    "header,data",
    [
        ({"x": {"dtype": "U8", "shape": [1], "data_offsets": [1, 2]}}, b"12"),
        ({"x": {"dtype": "U8", "shape": [2], "data_offsets": [0, 1]}}, b"1"),
        ({"x": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 3]}}, b"123"),
        ({"x": {"dtype": "UNKNOWN", "shape": [1], "data_offsets": [0, 1]}}, b"1"),
    ],
)
def test_invalid_tensor_bounds_and_dtypes_fail_closed(tmp_path, header, data):
    raw = json.dumps(header).encode()
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + data)
    with pytest.raises(SchemaError):
        tensor_identity(tmp_path)
