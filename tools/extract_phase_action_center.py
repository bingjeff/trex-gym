"""Extract a phase-binned action-center table from a trained joystick policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import jax
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.analyze_joystick_rollout import _TASKS
from tools.analyze_joystick_rollout import _load_env_config
from tools.analyze_policy_rollout import _load_policy


def _phase_bin_index(phase: float, bins: int) -> int:
    return int(np.floor((phase % (2.0 * np.pi)) / (2.0 * np.pi) * bins)) % bins


def _action_center_from_bins(sums: np.ndarray, counts: np.ndarray) -> np.ndarray:
    if np.any(counts <= 0):
        missing = [str(index) for index, count in enumerate(counts) if count <= 0]
        raise ValueError(f"No samples for phase bin(s): {', '.join(missing)}")
    return sums / counts[:, None]


def extract(args: argparse.Namespace) -> np.ndarray:
    if args.skip_steps >= args.steps:
        raise ValueError("--skip-steps must be smaller than --steps")
    if args.bins < 2:
        raise ValueError("--bins must be at least 2")

    config = _load_env_config(args)
    env = _TASKS[args.task][0](config)
    policy = _load_policy(args.checkpoint)
    jit_policy = jax.jit(policy)
    jit_step = jax.jit(env.step)

    rng = jax.random.PRNGKey(args.seed)
    state = env.reset(rng)
    command = jax.numpy.array([args.forward, args.turn])

    sums = np.zeros((args.bins, env.action_size), dtype=np.float64)
    counts = np.zeros(args.bins, dtype=np.float64)

    for step_index in range(args.steps):
        state.info["command"] = command
        state.info["steps_until_next_cmd"] = args.steps + 1
        state = state.replace(obs=env._get_obs(state.data, state.info))
        phase = float(jax.device_get(state.info["gait_phase"]))
        rng, action_rng = jax.random.split(rng)
        action, _ = jit_policy(state.obs, action_rng)
        state = jit_step(state, action)

        if step_index < args.skip_steps:
            continue

        if args.space == "raw":
            sample = np.asarray(jax.device_get(action), dtype=np.float64)
        else:
            sample = np.asarray(jax.device_get(state.info["last_act"]), dtype=np.float64)
        bin_index = _phase_bin_index(phase, args.bins)
        sums[bin_index] += sample
        counts[bin_index] += 1.0

    table = _action_center_from_bins(sums, counts)
    return np.clip(table, -1.0, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--task", choices=tuple(_TASKS.keys()), default="TrexJoystick")
    parser.add_argument("--impl", default="warp", choices=("jax", "warp"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--skip-steps", type=int, default=500)
    parser.add_argument("--forward", type=float, default=0.5)
    parser.add_argument("--turn", type=float, default=0.0)
    parser.add_argument("--bins", type=int, default=8)
    parser.add_argument("--space", choices=("raw", "applied"), default="applied")
    parser.add_argument(
        "--reset-pose", choices=("mixed", "standing", "side"), default="standing"
    )
    parser.add_argument(
        "--no-checkpoint-config",
        action="store_false",
        dest="use_checkpoint_config",
        help="Ignore checkpoints/config.json and use the task default config.",
    )
    parser.add_argument(
        "--config-overrides",
        default="",
        help="Optional JSON object of env config overrides for extraction.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = extract(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(table.tolist(), indent=2) + "\n")
    print(f"wrote {args.output}")
    print(f"rows: {table.shape[0]}")
    print(f"cols: {table.shape[1]}")
    print(f"space: {args.space}")


if __name__ == "__main__":
    main()
