"""Bounded Linux namespace execution. No unisolated fallback."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from ..contracts.schema import SchemaError, finite, integer


class ExecutorError(RuntimeError):
    """The executor could not run a candidate; not a candidate test failure."""


class NamespaceExecutor:
    version = "ard-bwrap-stdin-v1"

    def __init__(
        self, *, timeout_seconds: float = 3, memory_mb: int = 256, output_bytes: int = 65536
    ) -> None:
        finite(timeout_seconds, "timeout_seconds")
        if timeout_seconds <= 0:
            raise SchemaError("timeout_seconds must be positive")
        integer(memory_mb, "memory_mb", 32)
        integer(output_bytes, "output_bytes", 1024)
        self.timeout, self.memory_mb, self.output_bytes = timeout_seconds, memory_mb, output_bytes
        self.bwrap, self.prlimit = shutil.which("bwrap"), shutil.which("prlimit")
        if not self.bwrap or not self.prlimit or not Path("/usr/bin/python3").is_file():
            raise ExecutorError("executor requires Linux bwrap, prlimit and /usr/bin/python3")
        self.checked = False

    def _command(self, directory: Path, program: list[str]) -> list[str]:
        cpu = max(1, int(self.timeout) + 1)
        command = [
            self.prlimit,
            f"--as={self.memory_mb * 1024 * 1024}",
            f"--cpu={cpu}",
            f"--fsize={self.output_bytes}",
            "--nofile=64",
            "--",
            self.bwrap,
            "--die-with-parent",
            "--new-session",
            "--unshare-all",
            "--cap-drop",
            "ALL",
            "--clearenv",
            "--setenv",
            "PATH",
            "/usr/bin:/bin",
            "--setenv",
            "LANG",
            "C.UTF-8",
            "--ro-bind",
            "/usr",
            "/usr",
        ]
        for path in ("/lib", "/lib64"):
            if Path(path).exists():
                command += ["--ro-bind", path, path]
        command += [
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            "--bind",
            str(directory),
            "/work",
            "--chdir",
            "/work",
            # Apply NPROC after namespace setup: on a shared server a low
            # outer limit can prevent bwrap itself from creating namespaces.
            "/usr/bin/prlimit",
            "--nproc=32",
            "--",
            "/usr/bin/python3",
            "-I",
            "-S",
            "-B",
            *program,
        ]
        return command

    def preflight(self) -> None:
        if self.checked:
            return
        with tempfile.TemporaryDirectory(prefix="ard-executor-probe-") as temporary:
            try:
                completed = subprocess.run(
                    self._command(Path(temporary), ["-c", "print('ard-sandbox-ready')"]),
                    capture_output=True,
                    timeout=10,
                    env={"PATH": "/usr/bin:/bin"},
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise ExecutorError("namespace executor preflight could not start") from exc
        if completed.returncode != 0 or completed.stdout.strip() != b"ard-sandbox-ready":
            raise ExecutorError("namespace isolation unavailable; no unisolated execution fallback")
        self.checked = True

    def run(self, code: str, stdin: str = "") -> dict[str, Any]:
        self.preflight()
        if not isinstance(code, str) or len(code.encode()) > 65536:
            return {"status": "code_size_limit", "returncode": None, "stdout": "", "stderr": ""}
        if not isinstance(stdin, str):
            raise SchemaError("stdin must be text")
        with tempfile.TemporaryDirectory(prefix="ard-program-") as temporary:
            directory = Path(temporary)
            (directory / "candidate.py").write_text(code, encoding="utf-8")
            (directory / "runner.py").write_text(
                "from pathlib import Path\nPath('/work/started').write_text('started')\n"
                "exec(compile(Path('/work/candidate.py').read_text(), '<candidate>', 'exec'), "
                "{'__name__': '__main__', '__file__': '/work/candidate.py'})\n",
                encoding="utf-8",
            )
            with (
                (directory / "stdout").open("wb") as stdout,
                (directory / "stderr").open("wb") as stderr,
            ):
                try:
                    process = subprocess.Popen(
                        self._command(directory, ["/work/runner.py"]),
                        stdin=subprocess.PIPE,
                        stdout=stdout,
                        stderr=stderr,
                        start_new_session=True,
                        env={"PATH": "/usr/bin:/bin"},
                    )
                except OSError as exc:
                    raise ExecutorError("namespace executor launch failed") from exc
                timed_out = False
                try:
                    process.communicate(stdin.encode(), timeout=self.timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.communicate()
            if not (directory / "started").is_file():
                raise ExecutorError("executor did not enter the isolated program runner")
            output = (
                (directory / "stdout").read_bytes()[: self.output_bytes].decode(errors="replace")
            )
            error = (
                (directory / "stderr").read_bytes()[: self.output_bytes].decode(errors="replace")
            )
            status = (
                "timeout" if timed_out else "ok" if process.returncode == 0 else "program_error"
            )
            return {
                "status": status,
                "returncode": process.returncode,
                "stdout": output,
                "stderr": error,
            }
