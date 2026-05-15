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

_TASKS = {
    "TrexJoystick": (trex_joystick.TrexJoystick, trex_joystick.default_config),
    "TrexRun": (trex_joystick.TrexRun, trex_joystick.run_config),
}


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


def _parse_phase_template(value: str, action_size: int) -> np.ndarray:
    """Parses a phase template JSON string or @path into an action table."""
    text = Path(value[1:]).read_text() if value.startswith("@") else value
    template = np.asarray(json.loads(text), dtype=np.float32)
    if template.ndim != 2:
        raise ValueError("--phase-template must be a 2D JSON array.")
    if template.shape[0] < 2:
        raise ValueError("--phase-template must include at least two phase rows.")
    if template.shape[1] != action_size:
        raise ValueError(
            f"--phase-template rows must have {action_size} values, got "
            f"{template.shape[1]}."
        )
    return template


def _phase_template_action(phase: jax.Array, template: np.ndarray) -> jax.Array:
    table = jp.asarray(template)
    phase_count = table.shape[0]
    scaled_phase = jp.mod(phase, 2.0 * jp.pi) * (phase_count / (2.0 * jp.pi))
    lower = jp.floor(scaled_phase).astype(jp.int32)
    upper = (lower + 1) % phase_count
    alpha = scaled_phase - lower
    return (1.0 - alpha) * table[lower] + alpha * table[upper]


def analyze(args: argparse.Namespace) -> None:
    env_cls, config_fn = _TASKS[args.task]
    config = config_fn()
    if args.config_overrides:
        _apply_nested_config(config, json.loads(args.config_overrides))
    config.impl = args.impl
    config.reset_standing_prob = 1.0
    env = env_cls(config)
    jit_step = jax.jit(env.step)
    phase_template = (
        _parse_phase_template(args.phase_template, env.action_size)
        if args.phase_template
        else None
    )

    rng = jax.random.PRNGKey(args.seed)
    state = env.reset(rng)
    if args.initial_forward_velocity:
        qvel = state.data.qvel.at[0].set(args.initial_forward_velocity)
        state = state.replace(data=state.data.replace(qvel=qvel))
    command = jp.array([args.forward, args.turn])

    left_clearance = []
    right_clearance = []
    left_contact = []
    right_contact = []
    forward = []
    lateral = []
    orientation = []
    phase_scores = []
    phase_bins = [
        {
            "count": 0,
            "left_clearance": [],
            "right_clearance": [],
            "left_target": [],
            "right_target": [],
            "left_contact": [],
            "right_contact": [],
        }
        for _ in range(4)
    ]
    first_done_step = None

    for step_index in range(args.steps):
        state.info["command"] = command
        state.info["steps_until_next_cmd"] = args.steps + 1
        state = state.replace(obs=env._get_obs(state.data, state.info))
        if phase_template is None:
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
        else:
            template_action = _phase_template_action(
                state.info["gait_phase"], phase_template
            )
            if args.template_space == "raw":
                action = template_action
            else:
                residual_scale = env._residual_scale(state.info["command"])
                action_center = env._action_center(state.info)
                action = (template_action - action_center) / jp.maximum(
                    residual_scale, 1e-6
                )
                action = jp.clip(action, -1.0, 1.0)
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
        target = np.asarray(
            jax.device_get(env._phase_foot_clearance_targets(state.info["gait_phase"]))
        )
        phase_value = float(jax.device_get(state.info["gait_phase"]))
        bin_index = int(((phase_value % (2.0 * np.pi)) / (2.0 * np.pi)) * 4.0) % 4
        phase_bins[bin_index]["count"] += 1
        phase_bins[bin_index]["left_clearance"].append(float(lc))
        phase_bins[bin_index]["right_clearance"].append(float(rc))
        phase_bins[bin_index]["left_target"].append(float(target[0]))
        phase_bins[bin_index]["right_target"].append(float(target[1]))
        phase_bins[bin_index]["left_contact"].append(float(lcon > 0.5))
        phase_bins[bin_index]["right_contact"].append(float(rcon > 0.5))
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
    print(f"task: {args.task}")
    print(f"steps: {args.steps}")
    print(f"sample_steps: {len(forward)}")
    print(f"first_done_step: {first_done_step}")
    print(f"initial_forward_velocity: {args.initial_forward_velocity:.3f}")
    if phase_template is None:
        print(f"amplitude: {args.amplitude:.3f}")
        print(f"phase_offset: {args.phase_offset:.3f}")
        print(f"sign: {args.sign:.1f}")
        print(f"hip_amplitude: {args.hip_amplitude:.3f}")
        print(f"hip_phase_offset: {args.hip_phase_offset:.3f}")
        print(f"hip_sign: {args.hip_sign:.1f}")
    else:
        print(f"phase_template_rows: {phase_template.shape[0]}")
        print(f"template_space: {args.template_space}")
    print(f"mean_forward_vel: {_mean(forward):.3f}")
    print(f"mean_lateral_vel: {_mean(lateral):.3f}")
    print(f"mean_left_clearance: {_mean(left_clearance):.3f}")
    print(f"mean_right_clearance: {_mean(right_clearance):.3f}")
    print(f"max_left_clearance: {_max(left_clearance):.3f}")
    print(f"max_right_clearance: {_max(right_clearance):.3f}")
    print(f"left_contact_duty: {_mean(left_contact):.3f}")
    print(f"right_contact_duty: {_mean(right_contact):.3f}")
    print(f"mean_feet_phase_height: {_mean(phase_scores):.3f}")
    print("phase_bins:")
    for index, values in enumerate(phase_bins):
        print(
            "  "
            f"bin_{index}: count={values['count']} "
            f"left_clearance={_mean(values['left_clearance']):.3f} "
            f"right_clearance={_mean(values['right_clearance']):.3f} "
            f"left_target={_mean(values['left_target']):.3f} "
            f"right_target={_mean(values['right_target']):.3f} "
            f"left_contact={_mean(values['left_contact']):.3f} "
            f"right_contact={_mean(values['right_contact']):.3f}"
        )
    if orientation:
        print(f"orientation_reward_range: {min(orientation):.3f} {max(orientation):.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(_TASKS.keys()), default="TrexJoystick")
    parser.add_argument("--impl", default="warp", choices=("jax", "warp"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--forward", type=float, default=0.2)
    parser.add_argument("--turn", type=float, default=0.0)
    parser.add_argument(
        "--initial-forward-velocity",
        type=float,
        default=0.0,
        help="Optional initial root qvel x velocity for template sustain tests.",
    )
    parser.add_argument("--amplitude", type=float, default=0.5)
    parser.add_argument("--phase-offset", type=float, default=0.0)
    parser.add_argument("--sign", type=float, default=1.0, choices=(-1.0, 1.0))
    parser.add_argument("--hip-amplitude", type=float, default=0.0)
    parser.add_argument("--hip-phase-offset", type=float, default=0.0)
    parser.add_argument("--hip-sign", type=float, default=1.0, choices=(-1.0, 1.0))
    parser.add_argument(
        "--phase-template",
        default="",
        help=(
            "Optional JSON 2D action table, or @path to one, sampled uniformly "
            "over gait phase. Overrides the sine action pattern."
        ),
    )
    parser.add_argument(
        "--template-space",
        choices=("raw", "applied"),
        default="raw",
        help=(
            "Whether --phase-template entries are raw policy actions or final "
            "applied actions after action center/residual scaling."
        ),
    )
    parser.add_argument("--stop-on-done", action="store_true")
    parser.add_argument("--config-overrides", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
