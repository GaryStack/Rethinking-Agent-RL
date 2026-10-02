#!/usr/bin/env python3
"""Inspect a finished W&B 0.25 offline file locally; never sync it."""

import argparse
import json
import shutil
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore

    rows = []
    # SDK scanner opens r+b: scan a temporary copy, preserving original bytes.
    with tempfile.TemporaryDirectory(prefix="ard-wandb-scan-") as directory:
        copy = Path(directory) / "history.wandb"
        shutil.copyfile(args.input, copy)
        store = DataStore()
        store.open_for_scan(str(copy))
        try:
            while (data := store.scan_data()) is not None:
                record = wandb_internal_pb2.Record()
                record.ParseFromString(data)
                if record.HasField("history"):
                    rows.append(
                        {
                            item.key or "/".join(item.nested_key): json.loads(item.value_json)
                            for item in record.history.item
                        }
                    )
        finally:
            store.close()
    with args.output.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps({"history_rows": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
