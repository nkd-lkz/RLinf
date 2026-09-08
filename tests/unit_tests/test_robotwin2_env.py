# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.algorithms.rlt.transition import use_simulator_transition_replay
from rlinf.envs.robotwin2.protocol import (
    BridgeProtocolError,
    make_request,
    make_response,
    validate_response,
)
from rlinf.envs.robotwin2.robotwin2_env import RoboTwin2Env


class FakeBridge:
    def __init__(self, *, success_step: int | None = None) -> None:
        self.success_step = success_step
        self.episode_id = -1
        self.seed = -1
        self.step_index = 0
        self.transition_id = 0
        self.actions: list[np.ndarray] = []

    def _packet(self, *, action=None) -> dict:
        success = self.success_step is not None and self.step_index >= self.success_step
        state = np.full(14, self.step_index, dtype=np.float32)
        image = np.full((8, 10, 3), self.step_index, dtype=np.uint8)
        return {
            "observation": {
                "main_image": image,
                "left_wrist_image": image,
                "right_wrist_image": image,
                "state": state,
                "instruction": "adjust the bottle",
            },
            "reward": float(success),
            "terminated": success,
            "truncated": False,
            "info": {
                "episode_id": self.episode_id,
                "seed": self.seed,
                "step_index": self.step_index,
                "transition_id": self.transition_id,
                "success": success,
                "requested_action": action,
                "post_action_state": state.copy(),
            },
        }

    def reset(self, seed: int, episode_id: int) -> dict:
        self.seed = seed
        self.episode_id = episode_id
        self.step_index = 0
        return self._packet()

    def step(self, action) -> dict:
        action = np.asarray(action, dtype=np.float32).copy()
        self.actions.append(action)
        self.step_index += 1
        self.transition_id += 1
        return self._packet(action=action)


def make_cfg(**overrides):
    config = {
        "env_type": "robotwin2",
        "seed": 7,
        "auto_reset": False,
        "ignore_terminations": False,
        "use_rel_reward": True,
        "use_custom_reward": True,
        "reward_coef": 1.0,
        "max_episode_steps": 100,
        "use_fixed_reset_state_ids": False,
        "strict_k1": True,
        "center_crop": False,
        "task_config": {
            "task_name": "adjust_bottle",
            "setting": "demo_clean",
            "action_type": "qpos",
            "instruction": "adjust the bottle",
        },
        "rlt_policy_switch": {"enable": True, "actor_start_step": 10},
    }
    config.update(overrides)
    return OmegaConf.create(config)


def test_k1_twenty_step_transition_alignment():
    bridge = FakeBridge()
    env = RoboTwin2Env(
        make_cfg(),
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
        _bridge_client=bridge,
    )
    obs, reset_info = env.reset()
    assert reset_info["seed"].item() == 7
    torch.testing.assert_close(obs["states"], torch.zeros(1, 14))

    previous_state = obs["states"].clone()
    for step in range(1, 21):
        action = np.full((1, 1, 14), step / 100.0, dtype=np.float32)
        obs_list, rewards, terminated, truncated, infos = env.chunk_step(action)
        obs = obs_list[0]
        info = infos[0]

        assert previous_state.shape == (1, 14)
        assert obs["states"].shape == (1, 14)
        torch.testing.assert_close(obs["states"], torch.full((1, 14), float(step)))
        torch.testing.assert_close(info["post_action_state"], obs["states"])
        np.testing.assert_allclose(info["requested_action"].numpy(), action[:, 0])
        assert info["episode_id"].item() == 0
        assert info["step_index"].item() == step
        assert info["transition_id"].item() == step
        assert info["action_valid_mask"].item()
        assert info["rlt_switch_flags"].item() is (step >= 10)
        assert rewards.shape == (1, 1)
        assert not rewards.any()
        assert not terminated.any()
        assert not truncated.any()
        previous_state = obs["states"].clone()

    assert len(bridge.actions) == 20


def test_action_chunk_stops_at_episode_boundary_and_preserves_final_observation():
    bridge = FakeBridge(success_step=2)
    env = RoboTwin2Env(
        make_cfg(auto_reset=True, strict_k1=False),
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
        _bridge_client=bridge,
    )
    env.reset()
    actions = np.zeros((1, 3, 14), dtype=np.float32)
    obs_list, rewards, terminations, truncations, infos = env.chunk_step(actions)

    assert len(bridge.actions) == 2
    assert rewards.tolist() == [[0.0, 1.0, 0.0]]
    assert terminations.tolist() == [[False, True, False]]
    assert not truncations.any()
    assert [info["action_valid_mask"].item() for info in infos] == [True, True, False]
    torch.testing.assert_close(
        infos[1]["final_observation"]["states"], torch.full((1, 14), 2.0)
    )
    torch.testing.assert_close(obs_list[-1]["states"], torch.zeros(1, 14))


def test_strict_k1_rejects_longer_chunks():
    env = RoboTwin2Env(
        make_cfg(),
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
        _bridge_client=FakeBridge(),
    )
    env.reset()
    with pytest.raises(ValueError, match="strict_k1"):
        env.chunk_step(np.zeros((1, 2, 14), dtype=np.float32))


def test_robotwin2_uses_simulator_transition_replay():
    cfg = OmegaConf.create({"env": {"train": {"env_type": "robotwin2"}}})
    assert use_simulator_transition_replay(cfg)


def test_bridge_protocol_correlates_responses_and_surfaces_errors():
    request = make_request(3, "ping")
    assert validate_response(make_response(request, result={"ok": 1}), 3) == {"ok": 1}

    with pytest.raises(BridgeProtocolError, match="response id mismatch"):
        validate_response(make_response(request, result=None), 4)
    with pytest.raises(BridgeProtocolError, match="simulator failed"):
        validate_response(make_response(request, error="simulator failed"), 3)


def test_robotwin2_route_switches_reference_to_actor():
    from rlinf.algorithms.rlt.route import (
        RLTRouteContext,
        SimulatorRLTRoute,
        build_rlt_route,
    )

    cfg = OmegaConf.create(
        {
            "env": {"train": {"env_type": "robotwin2"}},
            "algorithm": {
                "rlt_schedule": {"enable": False, "warmup_post_collect_updates": 0}
            },
        }
    )
    route = build_rlt_route(cfg)
    assert isinstance(route, SimulatorRLTRoute)
    student = torch.full((1, 1, 14), 2.0)
    reference = torch.full((1, 1, 14), 1.0)

    def route_once(switch: bool):
        result = {"forward_inputs": {"ref_chunk": reference.reshape(1, -1).clone()}}
        output = route.route(
            RLTRouteContext(
                env_obs={},
                rlt_obs={"ref_chunk": reference.reshape(1, -1)},
                student_actions=student,
                result=result,
                mode="train",
                rlt_switch_flags=torch.tensor([[switch]]),
            )
        )
        return output

    reference_output = route_once(False)
    torch.testing.assert_close(reference_output.actions, reference)
    assert not reference_output.result["forward_inputs"]["actor_switch"].item()
    assert not reference_output.result["forward_inputs"]["record_transition"].item()

    actor_output = route_once(True)
    torch.testing.assert_close(actor_output.actions, student)
    assert actor_output.result["forward_inputs"]["actor_switch"].item()
    assert actor_output.result["forward_inputs"]["record_transition"].item()


def test_robotwin2_replay_appends_aligned_current_and_next_features():
    from types import SimpleNamespace

    from rlinf.algorithms.rlt.transition import update_rlt_transitions

    class Builder:
        def __init__(self):
            self.rows = []

        def append_transitions(self, current, next_obs):
            self.rows.append((current, next_obs))

    def policy_output(value: float):
        current = {
            "z_rl": torch.full((1, 3), value),
            "proprio": torch.full((1, 2), value),
            "ref_chunk": torch.full((1, 14), value),
        }
        transition = {
            f"rlt_transition_{key}": tensor + 0.5 for key, tensor in current.items()
        }
        return SimpleNamespace(forward_inputs=current | transition)

    pending = [None]
    builder = Builder()
    update_rlt_transitions(
        0, pending, [builder], policy_output(1.0), cache_current=True
    )
    assert pending[0] is not None
    assert builder.rows == []

    update_rlt_transitions(
        0, pending, [builder], policy_output(2.0), cache_current=False
    )
    assert pending[0] is None
    assert len(builder.rows) == 1
    current, next_obs = builder.rows[0]
    torch.testing.assert_close(current["z_rl"], torch.full((1, 3), 1.0))
    torch.testing.assert_close(next_obs["z_rl"], torch.full((1, 3), 2.5))


def test_robotwin2_registry_and_native_qpos_passthrough():
    from rlinf.envs import get_env_cls
    from rlinf.envs.action_utils import prepare_actions

    assert get_env_cls("robotwin2", make_cfg()) is RoboTwin2Env
    raw_actions = torch.tensor([[[1.5] * 14]], dtype=torch.float32)
    prepared = prepare_actions(
        raw_chunk_actions=raw_actions,
        env_type="robotwin2",
        model_type="openpi",
        num_action_chunks=1,
        action_dim=14,
    )
    np.testing.assert_array_equal(prepared, raw_actions.numpy())
