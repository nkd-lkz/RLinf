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

from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
TASK = "beat_block_hammer"
PROMPT = "grab the hammer and hit the block"
DATASET_KEY = "robotwin-beat-block-hammer-clean50"


def _load(relative_path: str):
    return OmegaConf.load(REPO_ROOT / relative_path)


def _assert_hammer_camera_config(task_config) -> None:
    assert task_config.task_name == TASK
    assert list(task_config.embodiment) == ["aloha-agilex"]
    assert task_config.camera.collect_head_camera is True
    assert task_config.camera.collect_wrist_camera is True


def test_hammer_pi05_eval_config_is_task_specific():
    cfg = _load(
        "evaluations/robotwin/"
        "robotwin_beat_block_hammer_openpi_pi05_smoke.yaml"
    )

    _assert_hammer_camera_config(cfg.env.eval.task_config)
    assert cfg.rollout.model.action_dim == 14
    assert cfg.rollout.model.num_action_chunks == 50
    assert cfg.rollout.model.openpi.config_name == "pi05_aloha_robotwin"
    assert "adjust_bottle" not in cfg.runner.logger.experiment_name


def test_hammer_pi05_sft_config_uses_hammer_data_and_prompt():
    cfg = _load(
        "examples/sft/config/robotwin_beat_block_hammer_sft_openpi_pi05.yaml"
    )

    assert cfg.actor.model.action_dim == 14
    assert cfg.actor.model.num_action_chunks == 50
    assert cfg.actor.model.openpi.num_images_in_input == 3
    assert cfg.actor.model.openpi.config_name == "pi05_aloha_robotwin"
    assert cfg.actor.model.openpi_data.repo_id == DATASET_KEY
    assert cfg.actor.model.openpi_data.default_prompt == PROMPT
    assert "beat-block-hammer" in cfg.data.train_data_paths


def test_hammer_rlt_stage1_freezes_vla_objective():
    cfg = _load(
        "examples/sft/config/"
        "robotwin_beat_block_hammer_rlt_stage1_openpi_pi05.yaml"
    )

    assert cfg.actor.model.action_dim == 14
    assert cfg.actor.model.num_action_chunks == 50
    assert cfg.actor.model.openpi.action_horizon == 50
    assert cfg.actor.model.openpi.action_chunk == 50
    assert cfg.actor.model.openpi.rlt_alpha == 0.0
    assert cfg.actor.model.openpi_data.repo_id == DATASET_KEY
    assert cfg.actor.model.openpi_data.default_prompt == PROMPT


def test_hammer_rlt_stage2_uses_paper_chunk_and_native_reference_horizon():
    cfg = _load(
        "examples/embodiment/config/"
        "robotwin_beat_block_hammer_rlt_stage2_ac_mlp.yaml"
    )

    for env_cfg in (cfg.env.train, cfg.env.eval):
        _assert_hammer_camera_config(env_cfg.task_config)
        assert env_cfg.action_dim == 14
        assert env_cfg.num_action_chunks == 10
        assert env_cfg.ref_num_action_chunks == 50
        assert env_cfg.max_episode_steps % env_cfg.num_action_chunks == 0
        assert env_cfg.max_steps_per_rollout_epoch % env_cfg.num_action_chunks == 0

    assert cfg.actor.model.action_dim == 14
    assert cfg.actor.model.num_action_chunks == 10
    assert cfg.actor.model.ref_num_action_chunks == 50
    assert cfg.rollout.model.num_action_chunks == 10
    assert cfg.rollout.model.ref_num_action_chunks == 50
    assert cfg.rollout.rlt_feature_model.num_action_chunks == 50
    assert cfg.rollout.rlt_feature_model.openpi.action_chunk == 50
    assert cfg.rollout.rlt_feature_model.openpi.action_horizon == 50
    assert cfg.rollout.rlt_feature_model.openpi_data.repo_id == DATASET_KEY
    assert cfg.rollout.rlt_feature_model.openpi_data.default_prompt == PROMPT
