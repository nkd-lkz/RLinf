# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.algorithms.rlt.transition import use_simulator_transition_replay
from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv
from rlinf.workers.env.env_worker import EnvWorker


def _switch_only_env(*, trigger_mode: str, actor_start_step: int = 0):
    env = object.__new__(RoboTwinEnv)
    env.num_envs = 2
    env._elapsed_steps = torch.zeros(2, dtype=torch.long)
    env._rlt_switch_flags = torch.zeros(2, dtype=torch.bool)
    env._rlt_switch_cfg = {
        "enable": True,
        "trigger_mode": trigger_mode,
        "actor_start_step": actor_start_step,
        "latch_until_done": True,
    }
    return env


def test_full_task_switch_is_active_on_reset():
    env = _switch_only_env(trigger_mode="full_task")

    env._update_rlt_switch()
    infos = env._attach_rlt_switch_info({}, chunk_step=1)

    assert env._rlt_switch_flags.tolist() == [True, True]
    assert infos["rlt_switch_flags"].shape == (2, 1)
    assert infos["rlt_switch_flags"].all()


def test_elapsed_step_switch_latches_per_environment():
    env = _switch_only_env(trigger_mode="elapsed_steps", actor_start_step=2)

    env._elapsed_steps[:] = torch.tensor([2, 1])
    env._update_rlt_switch()
    assert env._rlt_switch_flags.tolist() == [True, False]

    env._elapsed_steps[:] = torch.tensor([0, 2])
    env._update_rlt_switch()
    assert env._rlt_switch_flags.tolist() == [True, True]

    env._reset_rlt_switch(env_idx=[0])
    assert env._rlt_switch_flags.tolist() == [False, True]


def test_rlt_chunk_step_rejects_k_greater_than_one_before_simulation():
    env = _switch_only_env(trigger_mode="full_task")
    actions = np.zeros((2, 2, 14), dtype=np.float32)

    with pytest.raises(ValueError, match="requires num_action_chunks=1"):
        env.chunk_step(actions)


def test_native_robotwin_uses_per_step_transition_replay():
    cfg = OmegaConf.create({"env": {"train": {"env_type": "robotwin"}}})

    assert use_simulator_transition_replay(cfg)


class _BootstrapEnv:
    is_start = False

    def reset(self):
        return {"states": torch.zeros(1, 14)}, {
            "rlt_switch_flags": torch.ones(1, 1, dtype=torch.bool)
        }


def test_bootstrap_forwards_reset_rlt_switch_flags():
    worker = object.__new__(EnvWorker)
    worker.cfg = OmegaConf.create({"env": {"train": {"auto_reset": False}}})
    worker.stage_num = 1
    worker.train_num_envs_per_stage = 1
    worker.model_cfg = OmegaConf.create({"num_action_chunks": 1})
    worker.env_list = [_BootstrapEnv()]
    worker.enable_online_lerobot = False

    outputs = EnvWorker.bootstrap_step.__wrapped__.__wrapped__(worker)

    assert outputs[0].rlt_switch_flags.tolist() == [[True]]
