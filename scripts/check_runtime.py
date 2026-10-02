#!/usr/bin/env python3
"""Probe selected Python/CUDA/namespace facilities without starting inference or training."""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from agent_rl_diagnostics import __version__
from agent_rl_diagnostics.integrations.health import run_probe

PYTHON = "import json,platform; print(json.dumps({'python':platform.python_version()}))"
PACKAGES = """
import importlib.metadata as m, json
versions = {}
for package in ('torch','vllm','transformers','ray','pyarrow','pandas','peft','wandb'):
    try: versions[package] = m.version(package)
    except m.PackageNotFoundError: versions[package] = None
print(json.dumps(versions))
"""
DRIVER = """
import ctypes,json
print(json.dumps({'stage':'calling_cuInit'}), flush=True)
code = ctypes.CDLL('libcuda.so.1').cuInit(0)
print(json.dumps({'cuInit_returncode':code}), flush=True)
raise SystemExit(0 if code == 0 else 1)
"""
CUDA = """
import torch,json
x = torch.ones(16,device='cuda:0')
torch.cuda.synchronize()
print(json.dumps({'torch':torch.__version__, 'device':torch.cuda.get_device_name(0),
                  'sum':x.sum().item()}), flush=True)
"""
NAMESPACE = """
import json
from agent_rl_diagnostics.environments.execution import NamespaceExecutor
result = NamespaceExecutor().run('print(6*7)')
print(json.dumps(result), flush=True)
raise SystemExit(0 if result['status']=='ok' and result['stdout']=='42\\n' else 1)
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--cuda-device", help="One explicitly selected physical GPU index")
    parser.add_argument("--namespace", action="store_true")
    parser.add_argument("--timeout", type=float, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("output exists; choose a new file")
    if not args.python.is_file() or not os.access(args.python, os.X_OK):
        parser.error("selected Python must be an existing executable")
    if args.cuda_device is not None and not args.cuda_device.isdecimal():
        parser.error("select one physical CUDA device by index")
    # Do not resolve a venv's Python symlink: it would lose venv discovery.
    executable = str(args.python.absolute())
    environment = dict(os.environ)
    if args.cuda_device is not None:
        environment["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    probes = []

    def probe(name, code):
        result = run_probe(
            name,
            [executable, "-c", code],
            timeout_seconds=args.timeout,
            environment=environment,
        )
        probes.append(result)
        return result["status"] == "passed"

    if probe("python", PYTHON):
        probe("package_metadata", PACKAGES)
        if args.cuda_device is not None:
            if probe("cuda_driver", DRIVER):
                probe("cuda_tensor", CUDA)
            else:
                probes.append({"name": "cuda_tensor", "status": "skipped_driver_failed"})
        if args.namespace:
            probe("namespace_execution", NAMESPACE)
    passed = all(p["status"] == "passed" for p in probes)
    report = {
        "schema_version": "ard-runtime-health-v1",
        "package_version": __version__,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "python_executable": executable,
        "physical_cuda_device": args.cuda_device,
        "requested_checks_passed": passed,
        "inference_executed": False,
        "training_executed": False,
        "probes": probes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
