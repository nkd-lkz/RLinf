# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Client and process lifecycle for the RoboTwin 2.0 simulator bridge."""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
import uuid
from multiprocessing.connection import Client, Connection
from pathlib import Path
from typing import Any

from rlinf.envs.robotwin2.protocol import AUTHKEY, make_request, validate_response


class RoboTwin2BridgeClient:
    """Launch and communicate with one isolated RoboTwin 2.0 simulator process."""

    def __init__(
        self,
        *,
        python_executable: str,
        robotwin_root: str,
        task_name: str,
        task_config: str = "demo_clean",
        action_type: str = "qpos",
        instruction: str | None = None,
        startup_timeout_s: float = 180.0,
        request_timeout_s: float = 120.0,
        log_path: str | None = None,
    ) -> None:
        # Keep a virtualenv's ``bin/python`` symlink intact. Resolving it to the
        # base interpreter bypasses the virtualenv and loses its site-packages.
        self.python_executable = str(Path(python_executable).expanduser().absolute())
        self.robotwin_root = str(Path(robotwin_root).expanduser().resolve())
        self.task_name = task_name
        self.task_config = task_config
        self.action_type = action_type
        self.instruction = instruction or task_name.replace("_", " ")
        self.startup_timeout_s = float(startup_timeout_s)
        self.request_timeout_s = float(request_timeout_s)
        self.log_path = log_path

        self._request_id = 0
        self._connection: Connection | None = None
        self._process: subprocess.Popen | None = None
        self._log_file = None
        self._socket_path = str(
            Path(tempfile.gettempdir()) / f"rlinf-robotwin2-{uuid.uuid4().hex}.sock"
        )
        self._start()

    def _start(self) -> None:
        server_path = Path(__file__).with_name("bridge_server.py")
        command = [
            self.python_executable,
            str(server_path),
            "--socket-path",
            self._socket_path,
            "--robotwin-root",
            self.robotwin_root,
            "--task-name",
            self.task_name,
            "--task-config",
            self.task_config,
            "--action-type",
            self.action_type,
            "--instruction",
            self.instruction,
        ]

        stdout = subprocess.DEVNULL
        if self.log_path:
            path = Path(self.log_path).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = path.open("a", encoding="utf-8")
            stdout = self._log_file

        child_env = os.environ.copy()
        child_env.setdefault("PYTHONUNBUFFERED", "1")
        self._process = subprocess.Popen(
            command,
            cwd=self.robotwin_root,
            env=child_env,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        deadline = time.monotonic() + self.startup_timeout_s
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise RuntimeError(
                    "RoboTwin 2.0 bridge exited during startup with code "
                    f"{self._process.returncode}. See {self.log_path or 'captured logs'}."
                )
            try:
                self._connection = Client(self._socket_path, authkey=AUTHKEY)
                self.request("ping")
                return
            except (FileNotFoundError, ConnectionRefusedError, OSError) as exc:
                last_error = exc
                time.sleep(0.1)
        self.close(force=True)
        raise TimeoutError(
            "Timed out waiting for the RoboTwin 2.0 bridge socket "
            f"after {self.startup_timeout_s:.1f}s: {last_error}"
        )

    def request(self, command: str, **payload: Any) -> Any:
        """Send one command and wait for its correlated response."""
        if self._connection is None:
            raise RuntimeError("RoboTwin 2.0 bridge is not connected.")
        self._request_id += 1
        request_id = self._request_id
        self._connection.send(make_request(request_id, command, **payload))
        if not self._connection.poll(self.request_timeout_s):
            raise TimeoutError(
                f"RoboTwin 2.0 bridge command {command!r} exceeded "
                f"{self.request_timeout_s:.1f}s."
            )
        return validate_response(self._connection.recv(), request_id)

    def reset(self, seed: int, episode_id: int) -> dict[str, Any]:
        """Reset the simulator and return its initial observation packet."""
        return self.request("reset", seed=int(seed), episode_id=int(episode_id))

    def step(self, action: Any) -> dict[str, Any]:
        """Execute one action and return the transition endpoint packet."""
        return self.request("step", action=action)

    def close(self, *, force: bool = False) -> None:
        """Close the connection and reap the bridge subprocess."""
        if self._connection is not None:
            if not force:
                try:
                    self.request("close")
                except Exception:
                    pass
            try:
                self._connection.close()
            except Exception:
                pass
            self._connection = None

        if self._process is not None:
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=5)
            self._process = None

        try:
            Path(self._socket_path).unlink(missing_ok=True)
        except OSError:
            pass
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def __enter__(self) -> "RoboTwin2BridgeClient":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()
