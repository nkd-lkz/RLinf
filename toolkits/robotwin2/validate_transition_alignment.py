# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Validate K=1 transition ordering against a real RoboTwin 2.0 task."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from rlinf.envs.robotwin2.bridge_client import RoboTwin2BridgeClient


def _required_path(
    cli_value: str | None, env_name: str, *, preserve_symlink: bool = False
) -> str:
    value = cli_value or os.environ.get(env_name)
    if not value:
        raise ValueError(f"Pass the corresponding option or set {env_name}.")
    path = Path(value).expanduser()
    return str(path.absolute() if preserve_symlink else path.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin-root")
    parser.add_argument("--sim-python")
    parser.add_argument("--task", default="adjust_bottle")
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--instruction", default="adjust the bottle")
    parser.add_argument("--action-type", choices=("qpos", "ee"), default="qpos")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--startup-timeout", type=float, default=180.0)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument(
        "--bridge-log",
        default="/data/kaize/rlinf/robotwin2/logs/transition-alignment-bridge.log",
    )
    parser.add_argument(
        "--output",
        default="/data/kaize/rlinf/robotwin2/logs/transition-alignment.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be positive.")
    robotwin_root = _required_path(args.robotwin_root, "ROBOTWIN_ROOT")
    sim_python = _required_path(
        args.sim_python, "ROBOTWIN2_PYTHON", preserve_symlink=True
    )
    records = []

    started = time.perf_counter()
    client_started = time.perf_counter()
    with RoboTwin2BridgeClient(
        python_executable=sim_python,
        robotwin_root=robotwin_root,
        task_name=args.task,
        task_config=args.task_config,
        action_type=args.action_type,
        instruction=args.instruction,
        startup_timeout_s=args.startup_timeout,
        request_timeout_s=args.request_timeout,
        log_path=args.bridge_log,
    ) as bridge:
        startup_s = time.perf_counter() - client_started
        reset_started = time.perf_counter()
        packet = bridge.reset(seed=args.seed, episode_id=0)
        reset_s = time.perf_counter() - reset_started
        initial_state = np.asarray(packet["observation"]["state"], dtype=np.float32)
        action_dim = 14 if args.action_type == "qpos" else 16
        if initial_state.shape != (14,):
            raise AssertionError(
                f"Expected a 14D qpos observation, got {initial_state.shape}."
            )

        previous_transition_id = int(packet["info"]["transition_id"])
        for step_index in range(1, args.steps + 1):
            # In qpos mode this is a hold command. It tests transport and temporal
            # ordering without intentionally moving the task objects.
            if args.action_type != "qpos":
                raise ValueError(
                    "The automatic hold-action check currently supports qpos only."
                )
            action = np.asarray(packet["observation"]["state"], dtype=np.float32)
            if action.shape != (action_dim,):
                raise AssertionError(
                    f"Expected action shape {(action_dim,)}, got {action.shape}."
                )

            step_started = time.perf_counter()
            next_packet = bridge.step(action)
            latency_s = time.perf_counter() - step_started
            info = next_packet["info"]
            next_state = np.asarray(
                next_packet["observation"]["state"], dtype=np.float32
            )
            echoed_action = np.asarray(info["requested_action"], dtype=np.float32)
            post_action_state = np.asarray(info["post_action_state"], dtype=np.float32)

            if int(info["episode_id"]) != 0:
                raise AssertionError("episode_id changed within the 20-step window.")
            if int(info["step_index"]) != step_index:
                raise AssertionError(
                    "step_index is not aligned with the request order."
                )
            if int(info["transition_id"]) != previous_transition_id + 1:
                raise AssertionError("transition_id is not strictly consecutive.")
            if not np.array_equal(echoed_action, action):
                raise AssertionError(
                    "The executed/requested action echo changed in transit."
                )
            if not np.array_equal(post_action_state, next_state):
                raise AssertionError("post_action_state differs from returned s_(t+1).")
            for camera_key in (
                "main_image",
                "left_wrist_image",
                "right_wrist_image",
            ):
                image = next_packet["observation"].get(camera_key)
                if image is not None and (
                    np.asarray(image).ndim != 3 or np.asarray(image).shape[-1] != 3
                ):
                    raise AssertionError(f"{camera_key} is not an HWC RGB image.")

            records.append(
                {
                    "episode_id": int(info["episode_id"]),
                    "step_index": int(info["step_index"]),
                    "transition_id": int(info["transition_id"]),
                    "latency_s": latency_s,
                    "server_action_s": float(info.get("server_action_s", 0.0)),
                    "server_observation_s": float(
                        info.get("server_observation_s", 0.0)
                    ),
                    "server_compute_s": float(info.get("server_compute_s", 0.0)),
                    "bridge_overhead_s": max(
                        0.0, latency_s - float(info.get("server_compute_s", 0.0))
                    ),
                    "reward": float(next_packet["reward"]),
                    "terminated": bool(next_packet["terminated"]),
                    "truncated": bool(next_packet["truncated"]),
                    "state_l2_delta": float(np.linalg.norm(next_state - initial_state)),
                }
            )
            previous_transition_id = int(info["transition_id"])
            packet = next_packet
            if packet["terminated"] or packet["truncated"]:
                if step_index != args.steps:
                    raise AssertionError(
                        f"Episode ended at step {step_index}, before requested {args.steps} steps."
                    )

    result = {
        "status": "passed",
        "task": args.task,
        "task_config": args.task_config,
        "action_type": args.action_type,
        "seed": args.seed,
        "steps": args.steps,
        "wall_time_s": time.perf_counter() - started,
        "startup_s": startup_s,
        "reset_s": reset_s,
        "mean_step_latency_s": float(np.mean([row["latency_s"] for row in records])),
        "mean_server_compute_s": float(
            np.mean([row["server_compute_s"] for row in records])
        ),
        "mean_bridge_overhead_s": float(
            np.mean([row["bridge_overhead_s"] for row in records])
        ),
        "p95_step_latency_s": float(
            np.percentile([row["latency_s"] for row in records], 95)
        ),
        "records": records,
    }
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "records"}, indent=2
        )
    )
    print(f"full report: {output}")


if __name__ == "__main__":
    main()
