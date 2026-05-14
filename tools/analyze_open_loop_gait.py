"""Analyze simple open-loop gait commands in the TrexJoystick environment."""

from __future__ import annotations

import argparse
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
from tools.analyze_joystick_rollout import _apply_nested_config


def _open_loop_action(
    action_size: int,
    phase: jax.Array,
    amplitude: float,
    phase_offset: float,
    sign: float,
    hip_amplitude: float,
    hip_phase_offset: float,
    hip_sign: float,
) -> jax.Array:
    wave = sign * amplitude * jp.sin(phase + phase_offset)
    hip_wave = hip_sign * hip_amplitude * jp.sin(phase + hip_phase_offset)
    action = jp.zeros(action_size)
    action = action.at[0].set(hip_wave)
    action = action.at[1].set(-hip_wave)
    action = action.at[2].set(wave)
    action = action.at[3].set(-wave)
    action = action.at[4].set(-wave)
    action = action.at[5].set(wave)
    action = action.at[6].set(0.75 * wave)
    action = action.at[7].set(-0.75 * wave)
    return action


def analyze(args: argparse.Namespace) -> None:
    config = trex_joystick.default_config()
    if args.config_overrides:
        _apply_nested_config(config, json.loads(args.config_overrides))
    config.impl = args.impl
    config.reset_standing_prob = 1.0
    env = trex_joystick.TrexJoystick(config)
    jit_step = jax.jit(env.step)

    rng = jax.random.PRNGKey(args.seed)
    state = env.reset(rng)
    command = jp.array([args.forward, args.turn])

    left_clearance = []
    right_clearance = []
    left_contact = []
    right_contact = []
    forward = []
    lateral = []
    orientation = []
    phase_scores = []
    first_done_step = None

    for step_index in range(args.steps):
        state.info["command"] = command
        state.info["steps_until_next_cmd"] = args.steps + 1
        state = state.replace(obs=env._get_obs(state.data, state.info))
        action = _open_loop_action(
            env.action_size,
            state.info["gait_phase"],
            args.amplitude,
            args.phase_offset,
            args.sign,
            args.hip_amplitude,
            args.hip_phase_offset,
            args.hip_sign,
        )
        state = jit_step(state, action)
        if bool(np.asarray(jax.device_get(state.done))) and first_done_step is None:
            first_done_step = step_index
        if first_done_step is not None and args.stop_on_done:
            break

        lc, rc = jax.device_get(env._foot_clearance_scores(state.data))
        lcon, rcon = jax.device_get(env._foot_contact_scores(state.data))
        local_linvel = np.asarray(jax.device_get(env.get_local_linvel(state.data)))
        phase_score = jax.device_get(
            env._reward_feet_phase_height_from_clearance(
                jp.array([lc, rc]), state.info["gait_phase"], command
            )
        )
        left_clearance.append(float(lc))
        right_clearance.append(float(rc))
        left_contact.append(float(lcon > 0.5))
        right_contact.append(float(rcon > 0.5))
        forward.append(float(local_linvel[0]))
        lateral.append(float(local_linvel[2]))
        orientation.append(
            float(jax.device_get(env._reward_orientation(env.get_gravity(state.data))))
        )
        phase_scores.append(float(phase_score))

    def _mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    def _max(values: list[float]) -> float:
        return float(np.max(values)) if values else 0.0

    print(f"impl: {args.impl}")
    print(f"steps: {args.steps}")
    print(f"sample_steps: {len(forward)}")
    print(f"first_done_step: {first_done_step}")
    print(f"amplitude: {args.amplitude:.3f}")
    print(f"phase_offset: {args.phase_offset:.3f}")
    print(f"sign: {args.sign:.1f}")
    print(f"hip_amplitude: {args.hip_amplitude:.3f}")
    print(f"hip_phase_offset: {args.hip_phase_offset:.3f}")
    print(f"hip_sign: {args.hip_sign:.1f}")
    print(f"mean_forward_vel: {_mean(forward):.3f}")
    print(f"mean_lateral_vel: {_mean(lateral):.3f}")
    print(f"mean_left_clearance: {_mean(left_clearance):.3f}")
    print(f"mean_right_clearance: {_mean(right_clearance):.3f}")
    print(f"max_left_clearance: {_max(left_clearance):.3f}")
    print(f"max_right_clearance: {_max(right_clearance):.3f}")
    print(f"left_contact_duty: {_mean(left_contact):.3f}")
    print(f"right_contact_duty: {_mean(right_contact):.3f}")
    print(f"mean_feet_phase_height: {_mean(phase_scores):.3f}")
    if orientation:
        print(f"orientation_reward_range: {min(orientation):.3f} {max(orientation):.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--impl", default="warp", choices=("jax", "warp"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--forward", type=float, default=0.2)
    parser.add_argument("--turn", type=float, default=0.0)
    parser.add_argument("--amplitude", type=float, default=0.5)
    parser.add_argument("--phase-offset", type=float, default=0.0)
    parser.add_argument("--sign", type=float, default=1.0, choices=(-1.0, 1.0))
    parser.add_argument("--hip-amplitude", type=float, default=0.0)
    parser.add_argument("--hip-phase-offset", type=float, default=0.0)
    parser.add_argument("--hip-sign", type=float, default=1.0, choices=(-1.0, 1.0))
    parser.add_argument("--stop-on-done", action="store_true")
    parser.add_argument("--config-overrides", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
