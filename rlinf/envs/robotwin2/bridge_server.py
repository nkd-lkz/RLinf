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

"""RoboTwin 2.0 simulator process owned by one RLinf environment."""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from multiprocessing.connection import Listener
from pathlib import Path
from typing import Any

import numpy as np
from protocol import AUTHKEY, PROTOCOL_VERSION, make_response


class RoboTwin2Simulator:
    """Own one latest-main RoboTwin task and expose reset/step packets."""

    def __init__(
        self,
        *,
        robotwin_root: str,
        task_name: str,
        task_config: str,
        action_type: str,
        instruction: str,
    ) -> None:
        self.robotwin_root = Path(robotwin_root).resolve()
        self.task_name = task_name
        self.task_config = task_config
        self.action_type = self._normalize_action_type(action_type)
        self.instruction = instruction
        self.task_env = None
        self.episode_id = -1
        self.seed = -1
        self.step_index = 0
        self.transition_id = 0
        self.success_once = False

        os.chdir(self.robotwin_root)
        for path in (
            self.robotwin_root,
            self.robotwin_root / "scripts",
            self.robotwin_root / "description" / "utils",
        ):
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)

        from scripts.eval_policy_xpolicylab import class_decorator, load_task_args

        user_args = {
            "task_name": self.task_name,
            "task_config": self.task_config,
            "policy_name": "rlinf_bridge",
        }
        self.task_args, _ = load_task_args(user_args)
        self.task_args["render_freq"] = 0
        self.task_args["eval_mode"] = True
        self.task_args["eval_video_log"] = False
        self.task_args["eval_video_save_dir"] = None
        self.task_args.setdefault("data_type", {})["rgb"] = True
        self.task_args["data_type"]["qpos"] = True
        self.task_env = class_decorator(self.task_name)

    @staticmethod
    def _normalize_action_type(action_type: str) -> str:
        normalized = action_type.lower()
        if normalized in {"joint", "qpos"}:
            return "qpos"
        if normalized in {"ee", "endpose"}:
            return "ee"
        raise ValueError(f"Unsupported RoboTwin 2.0 action type: {action_type!r}.")

    def reset(self, *, seed: int, episode_id: int) -> dict[str, Any]:
        """Create a fresh deterministic task scene."""
        self._close_scene(clear_cache=False)
        self.episode_id = int(episode_id)
        self.seed = int(seed)
        self.step_index = 0
        self.success_once = False
        self.task_env.setup_demo(
            now_ep_num=self.episode_id,
            seed=self.seed,
            is_test=True,
            **self.task_args,
        )
        self.task_env.set_instruction(instruction=self.instruction)
        raw_obs = self.task_env.get_obs()
        return self._packet(raw_obs=raw_obs, reward=0.0)

    def step(self, *, action: Any) -> dict[str, Any]:
        """Execute exactly one policy action and return its endpoint."""
        if self.task_env is None or self.episode_id < 0:
            raise RuntimeError("reset must be called before step.")
        action_array = np.asarray(action, dtype=np.float32).reshape(-1)
        expected_dim = 14 if self.action_type == "qpos" else 16
        if action_array.size != expected_dim:
            raise ValueError(
                f"RoboTwin 2.0 {self.action_type} action must have {expected_dim} "
                f"values, got shape {np.asarray(action).shape}."
            )

        started = time.perf_counter()
        self.task_env.take_action(action_array, action_type=self.action_type)
        action_elapsed_s = time.perf_counter() - started
        self.step_index += 1
        self.transition_id += 1
        observation_started = time.perf_counter()
        raw_obs = self.task_env.get_obs()
        observation_elapsed_s = time.perf_counter() - observation_started
        success = bool(self.task_env.eval_success or self.task_env.check_success())
        reward = float(success and not self.success_once)
        self.success_once = self.success_once or success
        packet = self._packet(raw_obs=raw_obs, reward=reward, action=action_array)
        packet["info"]["server_action_s"] = action_elapsed_s
        packet["info"]["server_observation_s"] = observation_elapsed_s
        packet["info"]["server_compute_s"] = time.perf_counter() - started
        return packet

    def _packet(
        self,
        *,
        raw_obs: dict[str, Any],
        reward: float,
        action: np.ndarray | None = None,
    ) -> dict[str, Any]:
        success = bool(self.task_env.eval_success or self.task_env.check_success())
        step_limit = int(self.task_env.step_lim)
        truncated = self.step_index >= step_limit and not success
        observation = raw_obs.get("observation", {})

        def camera(name: str) -> np.ndarray | None:
            payload = observation.get(name, {})
            rgb = payload.get("rgb")
            return None if rgb is None else np.asarray(rgb, dtype=np.uint8)

        state = np.asarray(raw_obs["joint_action"]["vector"], dtype=np.float32)
        return {
            "observation": {
                "main_image": camera("head_camera"),
                "left_wrist_image": camera("left_camera"),
                "right_wrist_image": camera("right_camera"),
                "state": state,
                "instruction": self.task_env.get_instruction() or self.instruction,
            },
            "reward": float(reward),
            "terminated": success,
            "truncated": truncated,
            "info": {
                "episode_id": self.episode_id,
                "seed": self.seed,
                "step_index": self.step_index,
                "transition_id": self.transition_id,
                "success": success,
                "step_limit": step_limit,
                "requested_action": action,
                "post_action_state": state.copy(),
            },
        }

    def _close_scene(self, *, clear_cache: bool = False) -> None:
        if self.task_env is None:
            return
        try:
            self.task_env.close_env(clear_cache=clear_cache)
        except Exception:
            pass

    def close(self) -> None:
        """Release SAPIEN scene resources."""
        self._close_scene(clear_cache=True)


def serve(args: argparse.Namespace) -> None:
    """Serve requests until the client sends ``close`` or disconnects."""
    socket_path = Path(args.socket_path)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path.unlink(missing_ok=True)
    simulator = None
    listener = None
    connection = None
    try:
        # Initialize the heavy renderer before exposing the socket. Otherwise a
        # client can connect and block inside the authentication handshake while
        # SAPIEN is still importing, bypassing the startup timeout.
        simulator = RoboTwin2Simulator(
            robotwin_root=args.robotwin_root,
            task_name=args.task_name,
            task_config=args.task_config,
            action_type=args.action_type,
            instruction=args.instruction,
        )
        listener = Listener(str(socket_path), family="AF_UNIX", authkey=AUTHKEY)
        connection = listener.accept()
        while True:
            try:
                request = connection.recv()
            except EOFError:
                break
            try:
                if request.get("protocol_version") != PROTOCOL_VERSION:
                    raise ValueError("Unsupported bridge protocol version.")
                command = request.get("command")
                payload = request.get("payload") or {}
                if command == "ping":
                    result = {"pid": os.getpid(), "protocol_version": PROTOCOL_VERSION}
                elif command == "reset":
                    result = simulator.reset(**payload)
                elif command == "step":
                    result = simulator.step(**payload)
                elif command == "close":
                    connection.send(make_response(request, result={"closed": True}))
                    break
                else:
                    raise ValueError(f"Unknown bridge command: {command!r}")
                connection.send(make_response(request, result=result))
            except Exception:
                connection.send(make_response(request, error=traceback.format_exc()))
    finally:
        if simulator is not None:
            simulator.close()
        if connection is not None:
            connection.close()
        if listener is not None:
            listener.close()
        socket_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    """Parse bridge server arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket-path", required=True)
    parser.add_argument("--robotwin-root", required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--action-type", default="qpos")
    parser.add_argument("--instruction", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    serve(parse_args())
