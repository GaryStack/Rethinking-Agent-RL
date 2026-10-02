"""Private short Ray runtime paths; logs are retained, not silently deleted."""

import tempfile
from pathlib import Path

from ..artifacts.io import write_json


def create_ray_runtime_directory(output: Path) -> Path:
    # Ray appends a timestamp/PID session plus sockets/plasma_store. Deep result
    # directories can exceed Linux AF_UNIX's 107-byte path limit before startup.
    root = Path(tempfile.mkdtemp(prefix="ard-r-", dir="/tmp"))
    write_json(
        output / "ray-runtime.json",
        {
            "temp_dir": str(root),
            "scope": "private_this_run_only",
            "retained_after_shutdown": True,
            "reason": "short_AF_UNIX_socket_paths",
        },
    )
    return root
