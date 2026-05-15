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

from mjx_gym import trex_constants as consts
from mjx_gym import trex_joystick
from tools.analyze_policy_rollout import _load_policy

_TASKS = {
    "TrexBalance": (trex_joystick.TrexBalance, trex_joystick.balance_config),
    "TrexWalk": (trex_joystick.TrexWalk, trex_joystick.walk_config),
    "TrexJoystick": (trex_joystick.TrexJoystick, trex_joystick.joystick_config),
    "TrexRun": (trex_joystick.TrexRun, trex_joystick.run_config),
}


def _apply_nested_config(config, values: dict) -> None:
    for key, value in values.items():
        if "." in key:
            head, tail = key.split(".", 1)
            _apply_nested_config(config[head], {tail: value})
            continue
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
    config = _TASKS[args.task][1]()
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
    env = _TASKS[args.task][0](config)
    policy = _load_policy(args.checkpoint)
    jit_policy = jax.jit(policy)
    jit_step = jax.jit(env.step)
    action_actuator_ids = np.asarray(
        jax.device_get(env._action_actuator_ids), dtype=int
    )
    leg_qpos_ids = np.asarray(jax.device_get(env._leg_qpos_ids), dtype=int)
    leg_qvel_ids = np.asarray(
        [
            env._mj_model.jnt_dofadr[env._mj_model.joint(name).id]
            for name in consts.LEG_JOINTS
        ],
        dtype=int,
    )
    action_names = list(consts.ACTION_ACTUATORS)
    leg_joint_names = list(consts.LEG_JOINTS)

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
    action_abs_sums = np.zeros(env.action_size)
    action_max_abs = np.zeros(env.action_size)
    action_sat_counts = np.zeros(env.action_size)
    applied_abs_sums = np.zeros(env.action_size)
    applied_max_abs = np.zeros(env.action_size)
    applied_sat_counts = np.zeros(env.action_size)
    ctrl_sums = np.zeros(env.action_size)
    ctrl_min = np.full(env.action_size, np.inf)
    ctrl_max = np.full(env.action_size, -np.inf)
    actuator_force_abs_sums = np.zeros(env.action_size)
    actuator_force_max_abs = np.zeros(env.action_size)
    leg_qpos_min = np.full(len(leg_qpos_ids), np.inf)
    leg_qpos_max = np.full(len(leg_qpos_ids), -np.inf)
    leg_qvel_max_abs = np.zeros(len(leg_qpos_ids))
    mean_foot_speed = 0.0
    max_foot_speed = 0.0
    mean_stride_extent = 0.0
    max_stride_extent = 0.0
    mean_gait_anti_phase = 0.0
    mean_gait_symmetry = 0.0
    mean_contact_duty_symmetry = 0.0
    mean_foot_contact_balance = 0.0
    mean_foot_slip = 0.0
    mean_stance_support_error = 0.0
    mean_stance_offset_x = 0.0
    mean_stance_offset_z = 0.0
    mean_feet_phase_height = 0.0
    mean_left_clearance = 0.0
    mean_right_clearance = 0.0
    mean_left_target_clearance = 0.0
    mean_right_target_clearance = 0.0
    mean_left_clearance_error = 0.0
    mean_right_clearance_error = 0.0
    left_contact_duty = 0.0
    right_contact_duty = 0.0
    min_torso_height = np.inf
    max_torso_height = -np.inf
    min_orientation = np.inf
    max_orientation = -np.inf
    first_done_step: int | None = None
    first_xy: np.ndarray | None = None
    last_xy: np.ndarray | None = None
    previous_foot_centers: np.ndarray | None = None
    phase_bin_counts = np.zeros(4)
    phase_bin_left_clearance = np.zeros(4)
    phase_bin_right_clearance = np.zeros(4)
    phase_bin_left_target = np.zeros(4)
    phase_bin_right_target = np.zeros(4)
    phase_bin_left_contact = np.zeros(4)
    phase_bin_right_contact = np.zeros(4)

    for step_index in range(args.steps):
        state.info["command"] = command
        state.info["steps_until_next_cmd"] = args.steps + 1
        state = state.replace(obs=env._get_obs(state.data, state.info))
        rng, action_rng = jax.random.split(rng)
        action, _ = jit_policy(state.obs, action_rng)
        state = jit_step(state, action)
        data = jax.device_get(state.data)
        action_np = np.asarray(jax.device_get(action))
        applied_action_np = np.asarray(jax.device_get(state.info["last_act"]))
        done = bool(np.asarray(jax.device_get(state.done)))
        if done and first_done_step is None:
            first_done_step = step_index
        if done and args.stop_on_done:
            break

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
        action_abs = np.abs(action_np)
        applied_abs = np.abs(applied_action_np)
        action_abs_sums += action_abs
        action_max_abs = np.maximum(action_max_abs, action_abs)
        action_sat_counts += action_abs > 0.98
        applied_abs_sums += applied_abs
        applied_max_abs = np.maximum(applied_max_abs, applied_abs)
        applied_sat_counts += applied_abs > 0.98
        ctrl_np = np.asarray(data.ctrl)[action_actuator_ids]
        ctrl_sums += ctrl_np
        ctrl_min = np.minimum(ctrl_min, ctrl_np)
        ctrl_max = np.maximum(ctrl_max, ctrl_np)
        actuator_force_abs = np.abs(np.asarray(data.actuator_force)[
            action_actuator_ids
        ])
        actuator_force_abs_sums += actuator_force_abs
        actuator_force_max_abs = np.maximum(actuator_force_max_abs, actuator_force_abs)
        qpos_np = np.asarray(data.qpos)[leg_qpos_ids]
        qvel_np = np.asarray(data.qvel)[leg_qvel_ids]
        leg_qpos_min = np.minimum(leg_qpos_min, qpos_np)
        leg_qpos_max = np.maximum(leg_qpos_max, qpos_np)
        leg_qvel_max_abs = np.maximum(leg_qvel_max_abs, np.abs(qvel_np))
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
        contact_target = np.asarray(
            jax.device_get(env._phase_contact_targets(state.info["gait_phase"]))
        )
        stance_weight = contact_target / max(float(np.sum(contact_target)), 1e-6)
        left_xz = np.array([float(left_offset[0]), float(left_offset[2])])
        right_xz = np.array([float(right_offset[0]), float(right_offset[2])])
        stance_xz = stance_weight[0] * left_xz + stance_weight[1] * right_xz
        support_xz = np.asarray(jax.device_get(env._standing_support_offset_xz))
        support_error = stance_xz - support_xz
        mean_stance_support_error += float(np.linalg.norm(support_error))
        mean_stance_offset_x += float(stance_xz[0])
        mean_stance_offset_z += float(stance_xz[1])
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
        left_clearance, right_clearance = jax.device_get(
            env._foot_clearance_scores(state.data)
        )
        left_target, right_target = jax.device_get(
            env._phase_foot_clearance_targets(state.info["gait_phase"])
        )
        phase_score = jax.device_get(
            env._reward_feet_phase_height_from_clearance(
                jp.array([left_clearance, right_clearance]),
                state.info["gait_phase"],
                state.info["command"],
            )
        )
        mean_feet_phase_height += float(phase_score)
        mean_left_clearance += float(left_clearance)
        mean_right_clearance += float(right_clearance)
        mean_left_target_clearance += float(left_target)
        mean_right_target_clearance += float(right_target)
        mean_left_clearance_error += abs(float(left_clearance - left_target))
        mean_right_clearance_error += abs(float(right_clearance - right_target))
        left_contact, right_contact = jax.device_get(
            env._foot_contact_scores(state.data)
        )
        phase = float(jax.device_get(state.info["gait_phase"]))
        phase_bin = int(np.floor((phase % (2.0 * np.pi)) / (0.5 * np.pi))) % 4
        phase_bin_counts[phase_bin] += 1.0
        phase_bin_left_clearance[phase_bin] += float(left_clearance)
        phase_bin_right_clearance[phase_bin] += float(right_clearance)
        phase_bin_left_target[phase_bin] += float(left_target)
        phase_bin_right_target[phase_bin] += float(right_target)
        phase_bin_left_contact[phase_bin] += float(left_contact > 0.5)
        phase_bin_right_contact[phase_bin] += float(right_contact > 0.5)
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
    print(f"task: {args.task}")
    print(f"reset_pose: {args.reset_pose}")
    print(f"command_forward: {args.forward:.3f}")
    print(f"command_turn: {args.turn:.3f}")
    print(f"steps: {args.steps}")
    print(f"skip_steps: {args.skip_steps}")
    print(f"sample_steps: {sample_steps}")
    print(f"first_done_step: {first_done_step}")
    if sample_steps == 0:
        return
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
    print(f"mean_stance_support_error: {mean_stance_support_error / sample_steps:.3f}")
    print(f"mean_stance_offset_x: {mean_stance_offset_x / sample_steps:.3f}")
    print(f"mean_stance_offset_z: {mean_stance_offset_z / sample_steps:.3f}")
    print(
        f"mean_contact_duty_symmetry: {mean_contact_duty_symmetry / sample_steps:.3f}"
    )
    print(f"mean_foot_contact_balance: {mean_foot_contact_balance / sample_steps:.3f}")
    print(f"mean_foot_slip: {mean_foot_slip / sample_steps:.3f}")
    print(f"mean_feet_phase_height: {mean_feet_phase_height / sample_steps:.3f}")
    print(f"mean_left_clearance: {mean_left_clearance / sample_steps:.3f}")
    print(f"mean_right_clearance: {mean_right_clearance / sample_steps:.3f}")
    print(
        f"mean_left_target_clearance: {mean_left_target_clearance / sample_steps:.3f}"
    )
    print(
        f"mean_right_target_clearance: {mean_right_target_clearance / sample_steps:.3f}"
    )
    print(
        f"mean_left_clearance_error: {mean_left_clearance_error / sample_steps:.3f}"
    )
    print(
        f"mean_right_clearance_error: {mean_right_clearance_error / sample_steps:.3f}"
    )
    print(f"left_contact_duty: {left_contact_duty / sample_steps:.3f}")
    print(f"right_contact_duty: {right_contact_duty / sample_steps:.3f}")
    print("phase_bins:")
    for index, count in enumerate(phase_bin_counts):
        if count <= 0:
            print(f"  bin_{index}: count=0")
            continue
        print(
            "  "
            f"bin_{index}: count={int(count)} "
            f"left_clearance={phase_bin_left_clearance[index] / count:.3f} "
            f"right_clearance={phase_bin_right_clearance[index] / count:.3f} "
            f"left_target={phase_bin_left_target[index] / count:.3f} "
            f"right_target={phase_bin_right_target[index] / count:.3f} "
            f"left_contact={phase_bin_left_contact[index] / count:.3f} "
            f"right_contact={phase_bin_right_contact[index] / count:.3f}"
        )
    if first_xy is not None and last_xy is not None:
        print(f"base_xy_displacement: {np.linalg.norm(last_xy - first_xy):.3f}")
    print(f"torso_height_range: {min_torso_height:.3f} {max_torso_height:.3f}")
    print(f"orientation_reward_range: {min_orientation:.3f} {max_orientation:.3f}")
    print("reward_terms:")
    for key in sorted(metric_sums):
        print(f"  {key}: {metric_sums[key]:.3f}")
    print("action_actuator_stats:")
    for index, name in enumerate(action_names):
        print(
            "  "
            f"{name}: "
            f"mean_abs_action={action_abs_sums[index] / sample_steps:.3f} "
            f"max_abs_action={action_max_abs[index]:.3f} "
            f"action_sat_frac={action_sat_counts[index] / sample_steps:.3f} "
            f"mean_abs_applied={applied_abs_sums[index] / sample_steps:.3f} "
            f"max_abs_applied={applied_max_abs[index]:.3f} "
            f"applied_sat_frac={applied_sat_counts[index] / sample_steps:.3f} "
            f"mean_ctrl={ctrl_sums[index] / sample_steps:.3f} "
            f"ctrl_range={ctrl_min[index]:.3f}..{ctrl_max[index]:.3f} "
            f"mean_abs_force={actuator_force_abs_sums[index] / sample_steps:.3f} "
            f"max_abs_force={actuator_force_max_abs[index]:.3f}"
        )
    print("leg_joint_ranges:")
    for index, name in enumerate(leg_joint_names):
        print(
            "  "
            f"{name}: "
            f"qpos_range={leg_qpos_min[index]:.3f}..{leg_qpos_max[index]:.3f} "
            f"max_abs_qvel={leg_qvel_max_abs[index]:.3f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--task", choices=tuple(_TASKS.keys()), default="TrexJoystick"
    )
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
        "--stop-on-done",
        action="store_true",
        help="Stop rollout analysis at the first environment termination.",
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
