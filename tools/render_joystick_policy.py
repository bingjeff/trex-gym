"""Render fixed-command videos for a trained TrexJoystick PPO checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import jax
import jax.numpy as jp
import mediapy
import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mjx_gym import trex_constants
from mjx_gym import trex_joystick
from tools.analyze_policy_rollout import _load_policy


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
    config = trex_joystick.default_config()
    config.impl = args.impl
    if args.reset_pose == "standing":
        config.reset_standing_prob = 1.0
    elif args.reset_pose == "side":
        config.reset_standing_prob = 0.0

    env = trex_joystick.TrexJoystick(config)
    policy = jax.jit(_load_policy(args.checkpoint))
    step = jax.jit(env.step)

    rng = jax.random.PRNGKey(args.seed)
    state = env.reset(rng)
    command = jp.array([args.forward, args.turn])

    model = env.mj_model
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
    mj_data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    frames = []
    try:
        for step_index in range(args.steps):
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
            frames.append(renderer.render())
    finally:
        renderer.close()

    if not frames:
        raise ValueError("No frames were captured; check --steps and --skip-steps.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mediapy.write_video(args.output, frames, fps=args.fps)
    print(f"wrote {args.output} frames={len(frames)} fps={args.fps}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--impl", choices=("jax", "warp"), default="warp")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--skip-steps", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--forward", type=float, default=0.0)
    parser.add_argument("--turn", type=float, default=0.0)
    parser.add_argument(
        "--reset-pose", choices=("mixed", "standing", "side"), default="standing"
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
