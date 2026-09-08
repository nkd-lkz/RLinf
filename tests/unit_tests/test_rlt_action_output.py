# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

import torch

from rlinf.models.embodiment.mlp_policy.rlt_mlp_policy import RLTMLPPolicy
from rlinf.models.embodiment.mlp_policy.rlt_td3_mlp_policy import RLTTD3MLPPolicy


def _obs():
    return {
        "z_rl": torch.zeros(1, 3),
        "proprio": torch.zeros(1, 2),
        "ref_chunk": torch.tensor([[[1.5, -1.2, 0.25, 0.8], [1.6, -1.1, 0.30, 0.7]]]),
    }


def _zero_parameters(model):
    for parameter in model.parameters():
        parameter.data.zero_()


def test_ac_residual_action_is_decoded_around_reference_qpos():
    model = RLTMLPPolicy(
        z_dim=3,
        proprio_dim=2,
        action_dim=4,
        num_action_chunks=2,
        action_output_mode="residual_to_reference",
        residual_scale=[0.05, 0.05, 0.05, 0.1],
    )
    _zero_parameters(model)
    actions, _, _ = model.sac_forward(_obs(), deterministic=True)
    torch.testing.assert_close(actions, _obs()["ref_chunk"].reshape(1, -1))
    assert actions.max() > 1.0


def test_td3_residual_action_is_decoded_around_reference_qpos():
    model = RLTTD3MLPPolicy(
        z_dim=3,
        proprio_dim=2,
        action_dim=4,
        num_action_chunks=2,
        action_output_mode="residual_to_reference",
        residual_scale=[0.05, 0.05, 0.05, 0.1],
    )
    _zero_parameters(model)
    actions, _, _ = model.sac_forward(
        _obs(), deterministic=True, apply_action_noise=False
    )
    torch.testing.assert_close(actions, _obs()["ref_chunk"].reshape(1, -1))
    assert actions.max() > 1.0


def test_default_ac_mode_preserves_historical_tanh_action_space():
    model = RLTMLPPolicy(
        z_dim=3,
        proprio_dim=2,
        action_dim=4,
        num_action_chunks=2,
    )
    _zero_parameters(model)
    actions, _, _ = model.sac_forward(_obs(), deterministic=True)
    torch.testing.assert_close(actions, torch.zeros(1, 8))
