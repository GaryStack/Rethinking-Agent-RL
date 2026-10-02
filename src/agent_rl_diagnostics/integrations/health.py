"""Bounded subprocess probes; no optional imports or inference in the parent."""

import math
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path

from ..contracts.schema import SchemaError


def run_probe(name, command, *, timeout_seconds=20, environment=None):
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or not 0 < timeout_seconds <= 300
    ):
        raise SchemaError("probe timeout must be finite and in (0,300]")
    started = time.monotonic()
    result = {"name": name, "status": "launch_error", "returncode": None}
    with tempfile.TemporaryDirectory(prefix="ard-probe-") as temporary:
        directory = Path(temporary)
        with (
            (directory / "stdout").open("wb") as stdout,
            (directory / "stderr").open("wb") as stderr,
        ):
            try:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    env=environment,
                    start_new_session=True,
                )
            except OSError as exc:
                result["launch_error"] = str(exc)
            else:
                result["pid"] = process.pid
                try:
                    process.wait(timeout=timeout_seconds)
                    result["status"] = "passed" if process.returncode == 0 else "failed"
                except subprocess.TimeoutExpired:
                    result["status"] = "timeout"
                    # Terminate only our probe's process group. Never reset a GPU,
                    # restart a driver or touch the caller's training processes.
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        pass  # A blocked kernel call must not block the doctor.
                result["returncode"] = process.poll()
                result["process_reaped"] = process.returncode is not None
        for name in ("stdout", "stderr"):
            with (directory / name).open("rb") as stream:
                output = stream.read(65537)
            result[name] = output[:65536].decode("utf-8", errors="replace")
            result[name + "_truncated"] = len(output) > 65536
    result["elapsed_seconds"] = time.monotonic() - started
    return result
