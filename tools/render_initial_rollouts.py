"""Render initial T-Rex rollouts with randomized resets and actions."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys

import mediapy
import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mjx_gym import trex_constants
from mjx_gym import trex_getup


@dataclass
class RolloutSummary:
    index: int
    seed: int
    side: float
    yaw: float
    x: float
    y: float
    root_z: float
    min_initial_geom: str
    min_initial_geom_z: float
    mean_abs_action: float
    max_abs_action: float
    max_abs_ctrl: float
    max_qvel: float
    video: str


def _quat_mul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=float,
    )


def _side_lying_quat(side: float, yaw: float) -> np.ndarray:
    roll = side * (np.pi / 2.0)
    roll_quat = np.array([np.cos(roll / 2.0), np.sin(roll / 2.0), 0.0, 0.0])
    yaw_quat = np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])
    return _quat_mul(yaw_quat, roll_quat)


def _sample_reset(
    model: mujoco.MjModel,
    config,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    qpos = trex_constants.side_lying_qpos(model)
    side = 1.0 if rng.random() < 0.5 else -1.0
    yaw = rng.uniform(-config.reset_yaw_range, config.reset_yaw_range)
    xy = rng.uniform(-config.reset_xy_range, config.reset_xy_range, size=2)
    joint_noise = rng.uniform(
        -config.reset_joint_noise,
        config.reset_joint_noise,
        size=model.nq - 7,
    )
    height_noise = rng.uniform(0.0, config.reset_height_noise)

    qpos[0:2] = xy
    qpos[2] += height_noise
    qpos[3:7] = _side_lying_quat(side, yaw)
    qpos[7:] = joint_noise
    qvel = rng.normal(size=model.nv) * config.reset_qvel_noise
    metadata = {
        "side": side,
        "yaw": yaw,
        "x": xy[0],
        "y": xy[1],
        "root_z": qpos[2],
    }
    return qpos, qvel, metadata


def _sample_action(
    rng: np.random.Generator,
    size: int,
    mode: str,
    std: float,
) -> np.ndarray:
    if mode == "normal":
        return np.clip(rng.normal(scale=std, size=size), -1.0, 1.0)
    if mode == "uniform":
        return rng.uniform(-1.0, 1.0, size=size)
    if mode == "zero":
        return np.zeros(size)
    raise ValueError(f"Unknown action mode: {mode}")


def _action_to_ctrl(env: trex_getup.TrexGetup, action: np.ndarray) -> np.ndarray:
    clipped_action = np.clip(action, -1.0, 1.0)
    positive_scale = np.asarray(env._action_ctrl_positive_scale)
    negative_scale = np.asarray(env._action_ctrl_negative_scale)
    neutral = np.asarray(env._action_ctrl_neutral)
    target_scale = np.where(clipped_action >= 0.0, positive_scale, negative_scale)
    target = neutral + clipped_action * target_scale * env._config.action_scale
    ctrl = np.zeros(env.mj_model.nu)
    ctrl[np.asarray(env._action_actuator_ids, dtype=int)] = target
    return ctrl


def _lowest_geom(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[str, float]:
    z_values = data.geom_xpos[:, 2]
    geom_id = int(np.argmin(z_values))
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id), z_values[geom_id]


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


def _render_rollout(
    env: trex_getup.TrexGetup,
    rollout_rng: np.random.Generator,
    video_path: Path,
    args: argparse.Namespace,
) -> RolloutSummary:
    model = env.mj_model
    data = mujoco.MjData(model)
    qpos, qvel, reset_metadata = _sample_reset(model, env._config, rollout_rng)
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mujoco.mj_forward(model, data)
    min_geom_name, min_geom_z = _lowest_geom(model, data)

    frames = []
    actions = []
    max_abs_ctrl = 0.0
    max_qvel = float(np.max(np.abs(data.qvel)))

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    try:
        held_action = np.zeros(env.action_size)
        for step_index in range(args.episode_steps):
            if step_index % args.action_hold == 0:
                held_action = _sample_action(
                    rollout_rng, env.action_size, args.action_mode, args.action_std
                )
            ctrl = _action_to_ctrl(env, held_action)
            data.ctrl[:] = ctrl
            for _ in range(env.n_substeps):
                mujoco.mj_step(model, data)
            renderer.update_scene(data, camera=_camera_for_frame(model, data, args))
            frames.append(renderer.render())
            actions.append(held_action.copy())
            max_abs_ctrl = max(max_abs_ctrl, float(np.max(np.abs(ctrl))))
            max_qvel = max(max_qvel, float(np.max(np.abs(data.qvel))))
    finally:
        renderer.close()

    mediapy.write_video(video_path, frames, fps=args.fps)
    action_array = np.asarray(actions)
    return RolloutSummary(
        index=args.rollout_index,
        seed=args.seed,
        side=reset_metadata["side"],
        yaw=reset_metadata["yaw"],
        x=reset_metadata["x"],
        y=reset_metadata["y"],
        root_z=reset_metadata["root_z"],
        min_initial_geom=min_geom_name,
        min_initial_geom_z=float(min_geom_z),
        mean_abs_action=float(np.mean(np.abs(action_array))),
        max_abs_action=float(np.max(np.abs(action_array))),
        max_abs_ctrl=max_abs_ctrl,
        max_qvel=max_qvel,
        video=video_path.name,
    )


def _write_summary(path: Path, summaries: list[RolloutSummary]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(RolloutSummary.__annotations__))
        writer.writeheader()
        for summary in summaries:
            writer.writerow(summary.__dict__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("videos/initial_rollouts"))
    parser.add_argument("--num-rollouts", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode-steps", type=int, default=None)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--camera", default="follow")
    parser.add_argument("--follow-body", default=trex_constants.ROOT_BODY)
    parser.add_argument("--camera-distance", type=float, default=8.0)
    parser.add_argument("--camera-azimuth", type=float, default=-125.0)
    parser.add_argument("--camera-elevation", type=float, default=-18.0)
    parser.add_argument(
        "--action-mode",
        choices=("normal", "uniform", "zero"),
        default="normal",
    )
    parser.add_argument("--action-std", type=float, default=1.0)
    parser.add_argument("--action-hold", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.action_hold < 1:
        raise ValueError("--action-hold must be at least 1")

    env = trex_getup.TrexGetup()
    if args.episode_steps is None:
        args.episode_steps = int(env._config.episode_length)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_sequence = np.random.SeedSequence(args.seed)
    rollout_seeds = seed_sequence.spawn(args.num_rollouts)

    summaries = []
    for index, rollout_seed in enumerate(rollout_seeds):
        args.rollout_index = index
        video_path = args.output_dir / f"initial_rollout_{index:02d}.mp4"
        rollout_rng = np.random.default_rng(rollout_seed)
        summary = _render_rollout(env, rollout_rng, video_path, args)
        summaries.append(summary)
        print(
            "wrote "
            f"{video_path} side={summary.side:+.0f} "
            f"yaw={summary.yaw:.3f} "
            f"mean_abs_action={summary.mean_abs_action:.3f} "
            f"max_qvel={summary.max_qvel:.3f}",
            flush=True,
        )

    summary_path = args.output_dir / "summary.csv"
    _write_summary(summary_path, summaries)
    print(f"wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
