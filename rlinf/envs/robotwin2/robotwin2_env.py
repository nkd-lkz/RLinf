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

"""RLinf environment adapter for the latest RoboTwin 2.0 ``main`` branch."""

from __future__ import annotations

import copy
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from PIL import Image

from rlinf.envs.robotwin2.bridge_client import RoboTwin2BridgeClient
from rlinf.envs.utils import center_crop_image

__all__ = ["RoboTwin2Env"]


class RoboTwin2Env(gym.Env):
    """Run one latest-main RoboTwin task through an isolated Python process.

    The first integration milestone deliberately accepts one environment only.
    RoboTwin and RLinf currently require incompatible Python environments; the
    subprocess boundary keeps their CUDA, SAPIEN, and package dependencies
    isolated while preserving RLinf's normal ``reset``/``chunk_step`` contract.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg: Any,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info: Any,
        record_metrics: bool = True,
        _bridge_client: Any | None = None,
    ) -> None:
        if int(num_envs) != 1:
            raise ValueError(
                "The latest-main RoboTwin 2.0 bridge currently requires "
                f"num_envs=1 per RLinf env worker, got {num_envs}."
            )

        self.cfg = cfg
        self.num_envs = 1
        self.seed_offset = int(seed_offset)
        self.total_num_processes = int(total_num_processes)
        self.worker_info = worker_info
        self.record_metrics = bool(record_metrics)
        self.auto_reset = bool(cfg.get("auto_reset", False))
        self.ignore_terminations = bool(cfg.get("ignore_terminations", False))
        self.use_rel_reward = bool(cfg.get("use_rel_reward", True))
        self.use_custom_reward = bool(cfg.get("use_custom_reward", True))
        self.reward_coef = float(cfg.get("reward_coef", 1.0))
        self.max_episode_steps = int(cfg.get("max_episode_steps", 200))
        self.use_fixed_reset_state_ids = bool(
            cfg.get("use_fixed_reset_state_ids", False)
        )
        self.strict_k1 = bool(cfg.get("strict_k1", True))
        self.center_crop = bool(cfg.get("center_crop", False))
        self.action_dim = (
            14
            if cfg.task_config.get("action_type", "qpos")
            in {
                "joint",
                "qpos",
            }
            else 16
        )

        self.base_seed = int(cfg.get("seed", 0))
        self.seed = self.base_seed + self.seed_offset
        self.reset_state_ids = torch.tensor([self.seed], dtype=torch.long)
        self._next_seed = self.seed + self.total_num_processes
        self._episode_id = -1
        self._episode_done = False
        self._is_start = True
        self._last_obs: dict[str, Any] | None = None
        self._bridge_client = _bridge_client
        self._owns_bridge = _bridge_client is None
        self._bridge_kwargs = self._build_bridge_kwargs() if self._owns_bridge else None

        self._elapsed_steps = torch.zeros(1, dtype=torch.long)
        self.prev_step_reward = torch.zeros(1, dtype=torch.float32)
        self.success_once = torch.zeros(1, dtype=torch.bool)
        self.fail_once = torch.zeros(1, dtype=torch.bool)
        self.returns = torch.zeros(1, dtype=torch.float32)
        self.action_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.action_dim,),
            dtype=np.float32,
        )

        switch_cfg = cfg.get("rlt_policy_switch", {}) or {}
        self._rlt_switch_enabled = bool(switch_cfg.get("enable", False))
        self._rlt_actor_start_step = int(switch_cfg.get("actor_start_step", 0))

        self._ensure_bridge()

    @property
    def device(self) -> torch.device:
        """Device used for environment outputs."""
        return torch.device("cpu")

    @property
    def elapsed_steps(self) -> torch.Tensor:
        """Number of executed policy actions in the active episode."""
        return self._elapsed_steps

    @property
    def is_start(self) -> bool:
        return self._is_start

    @is_start.setter
    def is_start(self, value: bool) -> None:
        self._is_start = bool(value)

    def _build_bridge_kwargs(self) -> dict[str, Any]:
        bridge_cfg = self.cfg.get("bridge", {}) or {}
        python_executable = bridge_cfg.get("python_executable")
        robotwin_root = bridge_cfg.get("robotwin_root")
        if not python_executable or not robotwin_root:
            raise ValueError(
                "robotwin2 requires env.bridge.python_executable and "
                "env.bridge.robotwin_root."
            )
        task_cfg = self.cfg.task_config
        return {
            "python_executable": str(python_executable),
            "robotwin_root": str(robotwin_root),
            "task_name": str(task_cfg.task_name),
            "task_config": str(task_cfg.get("setting", "demo_clean")),
            "action_type": str(task_cfg.get("action_type", "qpos")),
            "instruction": str(
                task_cfg.get("instruction", task_cfg.task_name.replace("_", " "))
            ),
            "startup_timeout_s": float(bridge_cfg.get("startup_timeout_s", 180.0)),
            "request_timeout_s": float(bridge_cfg.get("request_timeout_s", 120.0)),
            "log_path": bridge_cfg.get("log_path"),
        }

    def _ensure_bridge(self) -> None:
        if self._bridge_client is None:
            if self._bridge_kwargs is None:
                raise RuntimeError("RoboTwin 2.0 bridge cannot be recreated.")
            self._bridge_client = RoboTwin2BridgeClient(**self._bridge_kwargs)

    def _reset_metrics(self) -> None:
        self._elapsed_steps.zero_()
        self.prev_step_reward.zero_()
        self.success_once.zero_()
        self.fail_once.zero_()
        self.returns.zero_()

    def _process_image(self, image: Any) -> np.ndarray:
        array = np.asarray(image, dtype=np.uint8)
        if self.center_crop:
            array = np.asarray(center_crop_image(Image.fromarray(array).convert("RGB")))
        return array

    def _extract_obs(self, packet: dict[str, Any]) -> dict[str, Any]:
        raw = packet["observation"]
        main_image = self._process_image(raw["main_image"])
        wrist_images = []
        for key in ("left_wrist_image", "right_wrist_image"):
            if raw.get(key) is not None:
                wrist_images.append(torch.from_numpy(self._process_image(raw[key])))
        wrist_tensor = torch.stack(wrist_images)[None] if wrist_images else None
        return {
            "main_images": torch.from_numpy(main_image)[None],
            "wrist_images": wrist_tensor,
            "states": torch.as_tensor(
                np.asarray(raw["state"], dtype=np.float32).copy()
            )[None],
            "task_descriptions": [str(raw["instruction"])],
        }

    def _switch_flag(self) -> torch.Tensor:
        enabled = self._rlt_switch_enabled and (
            int(self._elapsed_steps[0]) >= self._rlt_actor_start_step
        )
        return torch.tensor([enabled], dtype=torch.bool)

    def _record_step_metrics(
        self, reward: torch.Tensor, success: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        self.returns += reward
        self.success_once |= success
        return {
            "success_once": self.success_once.clone(),
            "return": self.returns.clone(),
            "episode_len": self._elapsed_steps.clone(),
            "reward": self.returns / self._elapsed_steps.clamp_min(1),
        }

    def _extract_info(
        self,
        packet: dict[str, Any],
        *,
        reward: torch.Tensor,
        success: torch.Tensor,
        action_valid: bool = True,
    ) -> dict[str, Any]:
        raw = packet.get("info", {})
        info: dict[str, Any] = {
            "success": success,
            "episode_id": torch.tensor([int(raw.get("episode_id", -1))]),
            "seed": torch.tensor([int(raw.get("seed", -1))]),
            "step_index": torch.tensor([int(raw.get("step_index", 0))]),
            "transition_id": torch.tensor([int(raw.get("transition_id", 0))]),
            "action_valid_mask": torch.tensor([action_valid], dtype=torch.bool),
            "rlt_switch_flags": self._switch_flag(),
        }
        for key in ("requested_action", "post_action_state"):
            value = raw.get(key)
            if value is not None:
                info[key] = torch.as_tensor(np.asarray(value).copy())[None]
        if self.record_metrics:
            info["episode"] = self._record_step_metrics(reward, success)
        return info

    def _select_seed(self, env_seeds: Any | None, seed: int | None) -> int:
        if seed is not None:
            return int(seed)
        if env_seeds is None:
            return int(self.reset_state_ids[0])
        values = torch.as_tensor(env_seeds).reshape(-1)
        if values.numel() != 1:
            raise ValueError("robotwin2 reset expects exactly one seed.")
        return int(values[0])

    def reset(
        self,
        env_idx: int | list[int] | None = None,
        env_seeds: Any | None = None,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset the single simulator and return RLinf-formatted observations."""
        del options
        if env_idx not in (None, 0, [0]):
            raise ValueError("robotwin2 currently supports resetting env index 0 only.")
        self._ensure_bridge()
        selected_seed = self._select_seed(env_seeds, seed)
        self._episode_id += 1
        packet = self._bridge_client.reset(
            seed=selected_seed,
            episode_id=self._episode_id,
        )
        self._reset_metrics()
        self._episode_done = False
        self._is_start = False
        self._last_obs = self._extract_obs(packet)
        info = {
            "episode_id": torch.tensor([self._episode_id]),
            "seed": torch.tensor([selected_seed]),
            "step_index": torch.zeros(1, dtype=torch.long),
            "transition_id": torch.tensor(
                [int(packet.get("info", {}).get("transition_id", 0))]
            ),
            "rlt_switch_flags": self._switch_flag(),
        }
        return self._last_obs, info

    def _reward_from_packet(
        self, packet: dict[str, Any], success: torch.Tensor
    ) -> torch.Tensor:
        if not self.use_custom_reward:
            return torch.tensor(
                [float(packet.get("reward", 0.0)) * self.reward_coef],
                dtype=torch.float32,
            )
        absolute = success.to(torch.float32) * self.reward_coef
        if self.use_rel_reward:
            reward = absolute - self.prev_step_reward
            self.prev_step_reward = absolute
            return reward
        return absolute

    def step(
        self,
        actions: torch.Tensor | np.ndarray | dict[str, Any],
        auto_reset: bool = True,
    ) -> tuple[
        dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]
    ]:
        """Execute one RoboTwin policy action (one macro transition)."""
        if self._episode_id < 0:
            raise RuntimeError("reset must be called before step.")
        if self._episode_done:
            raise RuntimeError("The episode is done; call reset before another step.")
        if isinstance(actions, dict):
            actions = actions.get("actions")
        action = np.asarray(
            actions.detach().cpu().numpy()
            if isinstance(actions, torch.Tensor)
            else actions,
            dtype=np.float32,
        )
        if action.ndim == 2 and action.shape[0] == 1:
            action = action[0]
        if action.shape != (self.action_dim,):
            raise ValueError(
                f"robotwin2 action must have shape ({self.action_dim},) or "
                f"(1, {self.action_dim}), got {action.shape}."
            )

        packet = self._bridge_client.step(action)
        self._elapsed_steps += 1
        success = torch.tensor(
            [bool(packet.get("info", {}).get("success", packet["terminated"]))],
            dtype=torch.bool,
        )
        reward = self._reward_from_packet(packet, success)
        termination = torch.tensor([bool(packet["terminated"])], dtype=torch.bool)
        truncation = torch.tensor([bool(packet["truncated"])], dtype=torch.bool)
        truncation |= self._elapsed_steps >= self.max_episode_steps
        if self.ignore_terminations:
            termination.zero_()
        done = termination | truncation

        obs = self._extract_obs(packet)
        info = self._extract_info(packet, reward=reward, success=success)
        self._last_obs = obs
        self._episode_done = bool(done[0])

        if self._episode_done and auto_reset and self.auto_reset:
            final_obs = copy.deepcopy(obs)
            final_info = copy.deepcopy(info)
            self.update_reset_state_ids()
            obs, reset_info = self.reset()
            reset_info.update(
                {
                    "final_observation": final_obs,
                    "final_info": final_info,
                    "_final_info": done.clone(),
                    "_final_observation": done.clone(),
                    "_elapsed_steps": done.clone(),
                    "action_valid_mask": torch.ones(1, dtype=torch.bool),
                }
            )
            info = reset_info
        return obs, reward, termination, truncation, info

    def chunk_step(
        self, chunk_actions: torch.Tensor | np.ndarray | dict[str, Any]
    ) -> tuple[
        list[dict[str, Any]],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list[dict[str, Any]],
    ]:
        """Execute a chunk sequentially without crossing an episode boundary."""
        if isinstance(chunk_actions, dict):
            chunk_actions = chunk_actions.get("actions")
        actions = (
            chunk_actions.detach().cpu().numpy()
            if isinstance(chunk_actions, torch.Tensor)
            else np.asarray(chunk_actions)
        )
        if actions.ndim == 2:
            actions = actions[:, None, :]
        if actions.ndim != 3 or actions.shape[0] != 1:
            raise ValueError(
                "robotwin2 chunk actions must have shape [1, K, action_dim], "
                f"got {actions.shape}."
            )
        if actions.shape[2] != self.action_dim:
            raise ValueError(
                f"robotwin2 expected action_dim={self.action_dim}, got {actions.shape[2]}."
            )
        chunk_size = int(actions.shape[1])
        if self.strict_k1 and chunk_size != 1:
            raise ValueError(
                f"robotwin2 strict_k1 requires K=1, got action chunk length {chunk_size}."
            )

        obs_list: list[dict[str, Any]] = []
        infos_list: list[dict[str, Any]] = []
        rewards: list[torch.Tensor] = []
        terminations: list[torch.Tensor] = []
        truncations: list[torch.Tensor] = []
        boundary_info: dict[str, Any] | None = None

        for index in range(chunk_size):
            if boundary_info is not None:
                obs_list.append(copy.deepcopy(self._last_obs))
                padded_info = copy.deepcopy(boundary_info)
                padded_info["action_valid_mask"] = torch.zeros(1, dtype=torch.bool)
                infos_list.append(padded_info)
                rewards.append(torch.zeros(1, dtype=torch.float32))
                terminations.append(torch.zeros(1, dtype=torch.bool))
                truncations.append(torch.zeros(1, dtype=torch.bool))
                continue

            obs, reward, terminated, truncated, info = self.step(
                actions[:, index, :], auto_reset=True
            )
            obs_list.append(obs)
            infos_list.append(info)
            rewards.append(reward)
            terminations.append(terminated)
            truncations.append(truncated)
            if bool((terminated | truncated)[0]):
                boundary_info = info

        return (
            obs_list,
            torch.stack(rewards, dim=1),
            torch.stack(terminations, dim=1),
            torch.stack(truncations, dim=1),
            infos_list,
        )

    def update_reset_state_ids(self, env_idx: Any | None = None) -> None:
        """Advance the deterministic seed stream for the single environment."""
        del env_idx
        if self.use_fixed_reset_state_ids:
            return
        self.reset_state_ids[0] = self._next_seed
        self._next_seed += self.total_num_processes

    def sample_action_space(self) -> np.ndarray:
        """Sample one action chunk with the configured smoke-test horizon."""
        horizon = int(self.cfg.get("action_chunk_size", 1))
        return np.random.standard_normal((1, horizon, self.action_dim)).astype(
            np.float32
        )

    def offload(self, clear_cache: bool = True) -> None:
        """Release the external simulator process; the next reset recreates it."""
        del clear_cache
        if self._bridge_client is not None and self._owns_bridge:
            self._bridge_client.close()
            self._bridge_client = None
            self._episode_done = False

    def close(self) -> None:
        """Release bridge resources."""
        if self._bridge_client is not None and self._owns_bridge:
            self._bridge_client.close()
            self._bridge_client = None
