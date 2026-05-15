"""Drive a trained TrexJoystick policy with a gamepad in the MuJoCo viewer."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jp
import mujoco
import mujoco.viewer
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mjx_gym import trex_joystick
from tools.analyze_policy_rollout import _load_policy


class Gamepad:
    """Small pygame wrapper for polling a single controller."""

    def __init__(self, index: int, deadzone: float):
        try:
            import pygame
        except ImportError as exc:
            raise SystemExit(
                "pygame is required for gamepad input. Run `uv sync` first."
            ) from exc

        self._pygame = pygame
        self._deadzone = deadzone
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() <= index:
            raise SystemExit(
                f"No gamepad at index {index}; detected "
                f"{pygame.joystick.get_count()} controller(s)."
            )
        self._joystick = pygame.joystick.Joystick(index)
        self._joystick.init()

    @property
    def name(self) -> str:
        return self._joystick.get_name()

    def command(self, max_forward: float, max_reverse: float, max_turn: float):
        self._pygame.event.pump()
        forward_axis = self._axis(1)
        turn_axis = self._axis(0)

        throttle = self._deadband(-forward_axis)
        turn = self._deadband(turn_axis)
        if throttle >= 0.0:
            forward = throttle * max_forward
        else:
            forward = throttle * max_reverse
        return jp.array([forward, turn * max_turn])

    def _axis(self, axis: int) -> float:
        if self._joystick.get_numaxes() <= axis:
            return 0.0
        return float(self._joystick.get_axis(axis))

    def _deadband(self, value: float) -> float:
        if abs(value) < self._deadzone:
            return 0.0
        scaled = (abs(value) - self._deadzone) / (1.0 - self._deadzone)
        return float(np.sign(value) * np.clip(scaled, 0.0, 1.0))


def _copy_to_mujoco_viewer(data, mj_data: mujoco.MjData) -> None:
    host_data = jax.device_get(data)
    mj_data.qpos[:] = np.asarray(host_data.qpos)
    mj_data.qvel[:] = np.asarray(host_data.qvel)
    mj_data.ctrl[:] = np.asarray(host_data.ctrl)


def drive(args: argparse.Namespace) -> None:
    if args.task == "run":
        config = trex_joystick.run_config()
        env_cls = trex_joystick.TrexRun
    else:
        config = trex_joystick.joystick_config()
        env_cls = trex_joystick.TrexJoystick
    config.impl = args.impl
    config.reset_standing_prob = 1.0 if args.start == "standing" else 0.0
    env = env_cls(config)
    policy = jax.jit(_load_policy(args.checkpoint))
    step = jax.jit(env.step)

    rng = jax.random.PRNGKey(args.seed)
    state = env.reset(rng)
    if args.check_load:
        state.info["command"] = jp.zeros(2)
        state.info["steps_until_next_cmd"] = args.hold_steps
        state = state.replace(obs=env._get_obs(state.data, state.info))
        rng, action_rng = jax.random.split(rng)
        action, _ = policy(state.obs, action_rng)
        state = step(state, action)
        print(f"Loaded {args.checkpoint}")
        print(f"Action size: {env.action_size}")
        print(
            f"First action mean abs: {float(np.mean(np.abs(jax.device_get(action)))):.3f}"
        )
        print(f"Post-step reward: {float(jax.device_get(state.reward)):.3f}")
        return

    gamepad = Gamepad(args.gamepad, args.deadzone)
    mj_data = mujoco.MjData(env.mj_model)
    _copy_to_mujoco_viewer(state.data, mj_data)
    mujoco.mj_forward(env.mj_model, mj_data)

    print(f"Loaded {args.checkpoint}")
    print(f"Using gamepad: {gamepad.name}")
    print("Left stick: forward/back and turn. Close the viewer to quit.")

    with mujoco.viewer.launch_passive(env.mj_model, mj_data) as viewer:
        next_step = time.monotonic()
        while viewer.is_running():
            loop_start = time.monotonic()
            command = gamepad.command(args.max_forward, args.max_reverse, args.max_turn)
            state.info["command"] = command
            state.info["steps_until_next_cmd"] = args.hold_steps
            state = state.replace(obs=env._get_obs(state.data, state.info))
            rng, action_rng = jax.random.split(rng)
            action, _ = policy(state.obs, action_rng)
            state = step(state, action)

            _copy_to_mujoco_viewer(state.data, mj_data)
            mujoco.mj_forward(env.mj_model, mj_data)
            viewer.sync()

            next_step += env.dt
            sleep_time = next_step - time.monotonic()
            if sleep_time > 0.0:
                time.sleep(sleep_time)
            elif time.monotonic() - loop_start > 2.0 * env.dt:
                next_step = time.monotonic()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--task", choices=("joystick", "run"), default="joystick")
    parser.add_argument("--impl", choices=("jax", "warp"), default="jax")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gamepad", type=int, default=0)
    parser.add_argument("--deadzone", type=float, default=0.12)
    parser.add_argument("--max-forward", type=float, default=10.0)
    parser.add_argument("--max-reverse", type=float, default=0.25)
    parser.add_argument("--max-turn", type=float, default=1.0)
    parser.add_argument("--hold-steps", type=int, default=10_000)
    parser.add_argument("--start", choices=("standing", "side"), default="standing")
    parser.add_argument(
        "--check-load",
        action="store_true",
        help="Load the checkpoint and run one policy/env step without viewer input.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    drive(parse_args())
