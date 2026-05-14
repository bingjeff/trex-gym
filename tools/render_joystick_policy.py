"""Render fixed-command or scripted-command videos for a TrexJoystick checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jp
import mediapy
import mujoco
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mjx_gym import trex_constants
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


def _copy_to_mujoco(data, mj_data: mujoco.MjData) -> None:
    host_data = jax.device_get(data)
    mj_data.qpos[:] = np.asarray(host_data.qpos)
    mj_data.qvel[:] = np.asarray(host_data.qvel)
    mj_data.ctrl[:] = np.asarray(host_data.ctrl)


def _camera_for_frame(
    model: mujoco.MjModel, data: mujoco.MjData, args: argparse.Namespace
) -> str | mujoco.MjvCamera:
    if args.camera != "follow":
        return args.camera
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = data.xpos[model.body(args.follow_body).id]
    camera.distance = args.camera_distance
    camera.azimuth = args.camera_azimuth
    camera.elevation = args.camera_elevation
    return camera


def render(args: argparse.Namespace) -> None:
    env = _TASKS[args.task][0](_load_env_config(args))
    policy = jax.jit(_load_policy(args.checkpoint))
    step = jax.jit(env.step)

    rng = jax.random.PRNGKey(args.seed)
    state = env.reset(rng)
    command_sequence = _parse_command_sequence(args.command_sequence)

    model = env.mj_model
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
    mj_data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    frames = []
    frame_steps = _parse_frame_steps(args.frame_steps)
    try:
        for step_index in range(args.steps):
            command = _command_for_step(step_index, command_sequence, args)
            state.info["command"] = command
            state.info["steps_until_next_cmd"] = args.steps + 1
            state = state.replace(obs=env._get_obs(state.data, state.info))
            rng, action_rng = jax.random.split(rng)
            action, _ = policy(state.obs, action_rng)
            state = step(state, action)

            if step_index < args.skip_steps:
                continue
            if (step_index - args.skip_steps) % args.frame_stride != 0:
                continue

            _copy_to_mujoco(state.data, mj_data)
            mujoco.mj_forward(model, mj_data)
            renderer.update_scene(
                mj_data, camera=_camera_for_frame(model, mj_data, args)
            )
            frame = renderer.render()
            if step_index in frame_steps:
                args.frames_dir.mkdir(parents=True, exist_ok=True)
                frame_path = (
                    args.frames_dir / f"{args.output.stem}_{step_index:04d}.png"
                )
                Image.fromarray(frame).save(frame_path)
                print(f"wrote {frame_path}")
            frames.append(frame)
    finally:
        renderer.close()

    if not frames:
        raise ValueError("No frames were captured; check --steps and --skip-steps.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mediapy.write_video(args.output, frames, fps=args.fps)
    print(f"wrote {args.output} frames={len(frames)} fps={args.fps}")


def _parse_frame_steps(value: str) -> set[int]:
    if not value:
        return set()
    return {int(item) for item in value.split(",") if item}


def _parse_command_sequence(value: str) -> tuple[tuple[int, float, float], ...]:
    """Parses START:FORWARD:TURN entries separated by semicolons."""
    if not value:
        return ()

    sequence: list[tuple[int, float, float]] = []
    for item in value.split(";"):
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 3:
            raise ValueError(
                "--command-sequence entries must look like START:FORWARD:TURN"
            )
        start_step = int(parts[0])
        if start_step < 0:
            raise ValueError("--command-sequence start steps must be non-negative")
        sequence.append((start_step, float(parts[1]), float(parts[2])))

    if not sequence:
        return ()
    sequence.sort(key=lambda item: item[0])
    if sequence[0][0] != 0:
        raise ValueError("--command-sequence must include an entry starting at step 0")
    for previous, current in zip(sequence, sequence[1:]):
        if previous[0] == current[0]:
            raise ValueError("--command-sequence start steps must be unique")
    return tuple(sequence)


def _command_for_step(
    step_index: int,
    command_sequence: tuple[tuple[int, float, float], ...],
    args: argparse.Namespace,
) -> jax.Array:
    if not command_sequence:
        return jp.array([args.forward, args.turn])

    forward = command_sequence[0][1]
    turn = command_sequence[0][2]
    for start_step, sequence_forward, sequence_turn in command_sequence:
        if step_index < start_step:
            break
        forward = sequence_forward
        turn = sequence_turn
    return jp.array([forward, turn])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--task", choices=tuple(_TASKS.keys()), default="TrexJoystick"
    )
    parser.add_argument("--impl", choices=("jax", "warp"), default="warp")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--skip-steps", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument(
        "--frame-steps",
        default="",
        help="Comma-separated simulation step indices to save as PNG frames.",
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=Path("frames"),
        help="Directory for PNG frames requested by --frame-steps.",
    )
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--forward", type=float, default=0.0)
    parser.add_argument("--turn", type=float, default=0.0)
    parser.add_argument(
        "--command-sequence",
        default="",
        help=(
            "Optional semicolon-separated START:FORWARD:TURN schedule, e.g. "
            "'0:0:0;400:10:0;900:3:1'. Overrides --forward/--turn per step."
        ),
    )
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
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--camera", default="follow")
    parser.add_argument("--follow-body", default=trex_constants.ROOT_BODY)
    parser.add_argument("--camera-distance", type=float, default=10.0)
    parser.add_argument("--camera-azimuth", type=float, default=-125.0)
    parser.add_argument("--camera-elevation", type=float, default=-18.0)
    return parser.parse_args()


if __name__ == "__main__":
    render(parse_args())
