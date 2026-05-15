"""Search for simple phase-action-center tables with black-box rollouts."""

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

from tools.analyze_joystick_rollout import _apply_nested_config
from tools.analyze_open_loop_gait import _TASKS
from tools.analyze_open_loop_gait import _phase_template_action


def _symmetric_sine_table(params: np.ndarray, bins: int) -> np.ndarray:
    """Builds an applied-action table from a compact symmetric sine gait."""
    if params.shape != (14,):
        raise ValueError(f"Expected 14 parameters, got {params.shape}.")

    stand = np.array(
        [0.0, 0.0, -0.1666667, -0.1666667, 0.1111111, 0.1111111, -0.7083333,
         -0.7083333, 0.0, 0.0],
        dtype=np.float32,
    )
    add_amp, femur_amp, tibia_amp, ankle_amp = np.abs(params[:4])
    add_phase, femur_phase, tibia_phase, ankle_phase = params[4:8]
    femur_bias, tibia_bias, ankle_bias = params[8:11]
    tail_sag_amp, tail_ml_amp, tail_phase = params[11:14]

    table = np.zeros((bins, 10), dtype=np.float32)
    for row in range(bins):
        phase = 2.0 * np.pi * row / bins
        right_phase = phase
        left_phase = phase + np.pi
        table[row] = stand
        table[row, 0] += add_amp * np.sin(right_phase + add_phase)
        table[row, 1] -= add_amp * np.sin(left_phase + add_phase)
        table[row, 2] += femur_bias + femur_amp * np.sin(right_phase + femur_phase)
        table[row, 3] += femur_bias + femur_amp * np.sin(left_phase + femur_phase)
        table[row, 4] += tibia_bias + tibia_amp * np.sin(right_phase + tibia_phase)
        table[row, 5] += tibia_bias + tibia_amp * np.sin(left_phase + tibia_phase)
        table[row, 6] += ankle_bias + ankle_amp * np.sin(right_phase + ankle_phase)
        table[row, 7] += ankle_bias + ankle_amp * np.sin(left_phase + ankle_phase)
        table[row, 8] += tail_sag_amp * np.sin(phase + tail_phase)
        table[row, 9] += tail_ml_amp * np.sin(phase + tail_phase + 0.5 * np.pi)
    return np.clip(table, -1.0, 1.0)


def _initial_mean() -> np.ndarray:
    return np.array(
        [
            0.08,
            0.30,
            0.30,
            0.30,
            0.0,
            0.0,
            np.pi,
            -0.5 * np.pi,
            0.0,
            0.0,
            0.0,
            0.10,
            0.10,
            0.0,
        ],
        dtype=np.float64,
    )


def _initial_std() -> np.ndarray:
    return np.array(
        [
            0.10,
            0.25,
            0.25,
            0.25,
            np.pi,
            np.pi,
            np.pi,
            np.pi,
            0.25,
            0.25,
            0.25,
            0.20,
            0.20,
            np.pi,
        ],
        dtype=np.float64,
    )


def _evaluate_template(env, jit_step, args: argparse.Namespace, table: np.ndarray) -> dict:
    rng = jax.random.PRNGKey(args.seed)
    state = env.reset(rng)
    if args.initial_forward_velocity:
        qvel = state.data.qvel.at[0].set(args.initial_forward_velocity)
        state = state.replace(data=state.data.replace(qvel=qvel))
    command = jp.array([args.forward, args.turn])

    forward = []
    lateral = []
    orientation = []
    height = []
    left_contact = []
    right_contact = []
    first_done_step = None

    for step_index in range(args.steps):
        state.info["command"] = command
        state.info["steps_until_next_cmd"] = args.steps + 1
        state = state.replace(obs=env._get_obs(state.data, state.info))
        template_action = _phase_template_action(state.info["gait_phase"], table)
        residual_scale = env._residual_scale(state.info["command"])
        action_center = env._action_center(state.info)
        action = (template_action - action_center) / jp.maximum(residual_scale, 1e-6)
        action = jp.clip(action, -1.0, 1.0)
        state = jit_step(state, action)

        done = bool(np.asarray(jax.device_get(state.done)))
        if done and first_done_step is None:
            first_done_step = step_index
            if args.stop_on_done:
                break

        local_linvel = np.asarray(jax.device_get(env.get_local_linvel(state.data)))
        forward.append(float(local_linvel[0]))
        lateral.append(float(local_linvel[2]))
        orientation.append(
            float(jax.device_get(env._reward_orientation(env.get_gravity(state.data))))
        )
        height.append(float(np.asarray(jax.device_get(state.data.site_xpos))[env._imu_site_id, 2]))
        lc, rc = jax.device_get(env._foot_contact_scores(state.data))
        left_contact.append(float(lc > 0.5))
        right_contact.append(float(rc > 0.5))

    samples = max(len(forward), 1)
    mean_forward = float(np.mean(forward)) if forward else 0.0
    mean_lateral = float(np.mean(lateral)) if lateral else 0.0
    mean_orientation = float(np.mean(orientation)) if orientation else 0.0
    min_orientation = float(np.min(orientation)) if orientation else 0.0
    min_height = float(np.min(height)) if height else 0.0
    max_height = float(np.max(height)) if height else 0.0
    left_duty = float(np.mean(left_contact)) if left_contact else 0.0
    right_duty = float(np.mean(right_contact)) if right_contact else 0.0
    survival = samples / args.steps
    speed_score = np.exp(-np.square(args.forward - mean_forward) / max(args.speed_sigma, 1e-6))
    contact_score = np.exp(-4.0 * np.square(left_duty - right_duty)) * np.clip(
        left_duty + right_duty, 0.0, 1.0
    )
    height_penalty = np.square(max(max_height - args.max_height, 0.0))
    score = (
        5.0 * survival
        + 4.0 * mean_orientation
        + 2.0 * speed_score
        + contact_score
        - 0.5 * abs(mean_lateral)
        - 2.0 * height_penalty
    )
    if first_done_step is not None:
        score -= 2.0
    return {
        "score": float(score),
        "first_done_step": first_done_step,
        "mean_forward": mean_forward,
        "mean_lateral": mean_lateral,
        "mean_orientation": mean_orientation,
        "min_orientation": min_orientation,
        "min_height": min_height,
        "max_height": max_height,
        "left_contact_duty": left_duty,
        "right_contact_duty": right_duty,
    }


def search(args: argparse.Namespace) -> tuple[np.ndarray, dict]:
    env_cls, config_fn = _TASKS[args.task]
    config = config_fn()
    if args.config_overrides:
        _apply_nested_config(config, json.loads(args.config_overrides))
    config.impl = args.impl
    config.reset_standing_prob = 1.0
    env = env_cls(config)
    jit_step = jax.jit(env.step)

    rng = np.random.default_rng(args.seed)
    mean = _initial_mean()
    std = _initial_std() * args.initial_std_scale
    best_params = mean.copy()
    best_metrics = {"score": -np.inf}

    for generation in range(args.generations):
        candidates = rng.normal(mean, std, size=(args.population, mean.shape[0]))
        candidates[0] = mean
        scored = []
        for params in candidates:
            table = _symmetric_sine_table(params, args.bins)
            metrics = _evaluate_template(env, jit_step, args, table)
            scored.append((metrics["score"], params, metrics))
            if metrics["score"] > best_metrics["score"]:
                best_params = params.copy()
                best_metrics = metrics
        scored.sort(key=lambda item: item[0], reverse=True)
        elites = np.asarray([item[1] for item in scored[: args.elites]])
        mean = np.mean(elites, axis=0)
        std = np.maximum(np.std(elites, axis=0), args.min_std)
        print(
            f"generation={generation} "
            f"best={scored[0][0]:.3f} "
            f"overall={best_metrics['score']:.3f} "
            f"forward={best_metrics['mean_forward']:.3f} "
            f"done={best_metrics['first_done_step']}"
        )

    return _symmetric_sine_table(best_params, args.bins), best_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--task", choices=tuple(_TASKS.keys()), default="TrexRun")
    parser.add_argument("--impl", choices=("jax", "warp"), default="warp")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--forward", type=float, default=1.0)
    parser.add_argument("--turn", type=float, default=0.0)
    parser.add_argument("--initial-forward-velocity", type=float, default=0.0)
    parser.add_argument("--bins", type=int, default=8)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--elites", type=int, default=4)
    parser.add_argument("--initial-std-scale", type=float, default=1.0)
    parser.add_argument("--min-std", type=float, default=0.03)
    parser.add_argument("--speed-sigma", type=float, default=0.5)
    parser.add_argument("--max-height", type=float, default=2.8)
    parser.add_argument("--stop-on-done", action="store_true")
    parser.add_argument("--config-overrides", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table, metrics = search(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(table.tolist(), indent=2) + "\n")
    metrics_path = args.output.with_suffix(args.output.suffix + ".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")
    print(f"wrote {metrics_path}")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
