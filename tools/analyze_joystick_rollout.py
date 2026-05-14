"""Analyze command tracking for a trained TrexJoystick PPO checkpoint."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jp
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mjx_gym import trex_joystick
from tools.analyze_policy_rollout import _load_policy


def _apply_nested_config(config, values: dict) -> None:
    for key, value in values.items():
        if isinstance(value, dict) and key in config:
            _apply_nested_config(config[key], value)
        else:
            config[key] = value


def _checkpoint_config_path(checkpoint: Path) -> Path:
    if checkpoint.is_dir() and checkpoint.name == "checkpoints":
        return checkpoint / "config.json"
    if checkpoint.parent.name == "checkpoints":
        return checkpoint.parent / "config.json"
    return checkpoint / "config.json"


def _load_env_config(args: argparse.Namespace):
    config = trex_joystick.default_config()
    config_path = _checkpoint_config_path(args.checkpoint)
    if args.use_checkpoint_config and config_path.exists():
        _apply_nested_config(config, json.loads(config_path.read_text()))
    if args.config_overrides:
        _apply_nested_config(config, json.loads(args.config_overrides))
    config.impl = args.impl
    if args.reset_pose == "standing":
        config.reset_standing_prob = 1.0
    elif args.reset_pose == "side":
        config.reset_standing_prob = 0.0
    return config


def analyze(args: argparse.Namespace) -> None:
    if args.skip_steps >= args.steps:
        raise ValueError("--skip-steps must be smaller than --steps")

    config = _load_env_config(args)
    env = trex_joystick.TrexJoystick(config)
    policy = _load_policy(args.checkpoint)
    jit_policy = jax.jit(policy)
    jit_step = jax.jit(env.step)

    rng = jax.random.PRNGKey(args.seed)
    state = env.reset(rng)
    command = jp.array([args.forward, args.turn])

    reward_sum = 0.0
    metric_sums = Counter()
    sample_steps = 0
    mean_forward = 0.0
    mean_lateral = 0.0
    mean_vertical = 0.0
    mean_turn = 0.0
    max_forward_error = 0.0
    max_turn_error = 0.0
    mean_abs_action = 0.0
    max_abs_action = 0.0
    mean_foot_speed = 0.0
    max_foot_speed = 0.0
    mean_stride_extent = 0.0
    max_stride_extent = 0.0
    mean_gait_anti_phase = 0.0
    mean_gait_symmetry = 0.0
    mean_contact_duty_symmetry = 0.0
    mean_foot_contact_balance = 0.0
    mean_foot_slip = 0.0
    left_contact_duty = 0.0
    right_contact_duty = 0.0
    min_torso_height = np.inf
    max_torso_height = -np.inf
    min_orientation = np.inf
    max_orientation = -np.inf
    first_xy: np.ndarray | None = None
    last_xy: np.ndarray | None = None
    previous_foot_centers: np.ndarray | None = None

    for step_index in range(args.steps):
        state.info["command"] = command
        state.info["steps_until_next_cmd"] = args.steps + 1
        state = state.replace(obs=env._get_obs(state.data, state.info))
        rng, action_rng = jax.random.split(rng)
        action, _ = jit_policy(state.obs, action_rng)
        state = jit_step(state, action)
        data = jax.device_get(state.data)
        action_np = np.asarray(jax.device_get(action))

        if step_index < args.skip_steps:
            continue

        local_linvel = np.asarray(jax.device_get(env.get_local_linvel(state.data)))
        local_angvel = np.asarray(jax.device_get(env.get_local_angvel(state.data)))
        forward_error = abs(args.forward - float(local_linvel[0]))
        turn_error = abs(args.turn - float(local_angvel[1]))
        xy = np.asarray(data.qpos[:2])
        if first_xy is None:
            first_xy = xy.copy()
        last_xy = xy.copy()

        sample_steps += 1
        reward_sum += float(jax.device_get(state.reward))
        mean_forward += float(local_linvel[0])
        mean_lateral += float(local_linvel[2])
        mean_vertical += float(local_linvel[1])
        mean_turn += float(local_angvel[1])
        max_forward_error = max(max_forward_error, forward_error)
        max_turn_error = max(max_turn_error, turn_error)
        mean_abs_action += float(np.mean(np.abs(action_np)))
        max_abs_action = max(max_abs_action, float(np.max(np.abs(action_np))))
        foot_centers = np.asarray(jax.device_get(env._foot_centers_world(state.data)))
        if previous_foot_centers is not None:
            foot_speed = np.linalg.norm(
                (foot_centers - previous_foot_centers) / env.dt, axis=1
            )
            mean_foot_speed += float(np.mean(foot_speed))
            max_foot_speed = max(max_foot_speed, float(np.max(foot_speed)))
        previous_foot_centers = foot_centers.copy()
        left_offset, right_offset = jax.device_get(
            env._mjx_foot_offsets_in_torso_frame(state.data)
        )
        left_stride = abs(float(left_offset[0] - env._standing_left_foot_offset[0]))
        right_stride = abs(float(right_offset[0] - env._standing_right_foot_offset[0]))
        stride_extent = 0.5 * (left_stride + right_stride)
        mean_stride_extent += stride_extent
        max_stride_extent = max(max_stride_extent, stride_extent)
        mean_gait_anti_phase += float(
            jax.device_get(env._reward_gait_anti_phase(state.data))
        )
        mean_gait_symmetry += float(
            jax.device_get(env._reward_gait_symmetry(state.data))
        )
        mean_contact_duty_symmetry += float(
            jax.device_get(env._reward_contact_duty_symmetry(state.data, state.info))
        )
        mean_foot_contact_balance += float(
            jax.device_get(env._reward_foot_contact_balance(state.data))
        )
        mean_foot_slip += float(
            jax.device_get(env._cost_foot_slip(state.data, state.info))
        )
        left_contact, right_contact = jax.device_get(
            env._foot_contact_scores(state.data)
        )
        left_contact_duty += float(left_contact > 0.5)
        right_contact_duty += float(right_contact > 0.5)

        torso_height = float(np.asarray(data.site_xpos)[env._imu_site_id, 2])
        min_torso_height = min(min_torso_height, torso_height)
        max_torso_height = max(max_torso_height, torso_height)
        orientation = float(env._reward_orientation(env.get_gravity(state.data)))
        min_orientation = min(min_orientation, orientation)
        max_orientation = max(max_orientation, orientation)
        for key, value in state.metrics.items():
            metric_sums[key] += float(jax.device_get(value))

    print(f"checkpoint: {args.checkpoint}")
    print(f"reset_pose: {args.reset_pose}")
    print(f"command_forward: {args.forward:.3f}")
    print(f"command_turn: {args.turn:.3f}")
    print(f"steps: {args.steps}")
    print(f"skip_steps: {args.skip_steps}")
    print(f"sample_steps: {sample_steps}")
    print(f"episode_reward_sum: {reward_sum:.3f}")
    print(f"mean_forward_vel: {mean_forward / sample_steps:.3f}")
    print(f"mean_lateral_vel: {mean_lateral / sample_steps:.3f}")
    print(f"mean_vertical_vel: {mean_vertical / sample_steps:.3f}")
    print(f"mean_turn_vel: {mean_turn / sample_steps:.3f}")
    print(f"mean_forward_error: {abs(args.forward - mean_forward / sample_steps):.3f}")
    print(f"mean_turn_error: {abs(args.turn - mean_turn / sample_steps):.3f}")
    print(f"max_forward_error: {max_forward_error:.3f}")
    print(f"max_turn_error: {max_turn_error:.3f}")
    print(f"mean_abs_action: {mean_abs_action / sample_steps:.3f}")
    print(f"max_abs_action: {max_abs_action:.3f}")
    print(f"mean_foot_speed: {mean_foot_speed / max(sample_steps - 1, 1):.3f}")
    print(f"max_foot_speed: {max_foot_speed:.3f}")
    print(f"mean_stride_extent: {mean_stride_extent / sample_steps:.3f}")
    print(f"max_stride_extent: {max_stride_extent:.3f}")
    print(f"mean_gait_anti_phase: {mean_gait_anti_phase / sample_steps:.3f}")
    print(f"mean_gait_symmetry: {mean_gait_symmetry / sample_steps:.3f}")
    print(
        f"mean_contact_duty_symmetry: {mean_contact_duty_symmetry / sample_steps:.3f}"
    )
    print(f"mean_foot_contact_balance: {mean_foot_contact_balance / sample_steps:.3f}")
    print(f"mean_foot_slip: {mean_foot_slip / sample_steps:.3f}")
    print(f"left_contact_duty: {left_contact_duty / sample_steps:.3f}")
    print(f"right_contact_duty: {right_contact_duty / sample_steps:.3f}")
    if first_xy is not None and last_xy is not None:
        print(f"base_xy_displacement: {np.linalg.norm(last_xy - first_xy):.3f}")
    print(f"torso_height_range: {min_torso_height:.3f} {max_torso_height:.3f}")
    print(f"orientation_reward_range: {min_orientation:.3f} {max_orientation:.3f}")
    print("reward_terms:")
    for key in sorted(metric_sums):
        print(f"  {key}: {metric_sums[key]:.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--impl", default="warp", choices=("jax", "warp"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--skip-steps", type=int, default=500)
    parser.add_argument("--forward", type=float, default=0.0)
    parser.add_argument("--turn", type=float, default=0.0)
    parser.add_argument(
        "--reset-pose", choices=("mixed", "standing", "side"), default="standing"
    )
    parser.add_argument(
        "--no-checkpoint-config",
        action="store_false",
        dest="use_checkpoint_config",
        help="Ignore checkpoints/config.json and use the default joystick config.",
    )
    parser.add_argument(
        "--config-overrides",
        default="",
        help="Optional JSON object of env config overrides for this rollout.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
