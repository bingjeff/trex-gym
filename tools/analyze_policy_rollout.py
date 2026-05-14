"""Analyze contacts and reward terms for a trained TrexGetup PPO checkpoint."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

from brax.training.agents.ppo import checkpoint as ppo_checkpoint
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics
from brax.training import networks as brax_networks
import jax
import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mjx_gym import trex_getup


def _geom_bottom(model: mujoco.MjModel, data, geom_ids: np.ndarray) -> np.ndarray:
    geom_xpos = np.asarray(data.geom_xpos)[geom_ids]
    geom_xmat = np.asarray(data.geom_xmat)[geom_ids].reshape((-1, 3, 3))
    geom_size = model.geom_size[geom_ids]
    radius = geom_size[:, 0]
    half_length = geom_size[:, 1]
    local_z_vertical = np.abs(geom_xmat[:, 2, 2])
    return geom_xpos[:, 2] - radius - half_length * local_z_vertical


def _geom_name(model: mujoco.MjModel, geom_id: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    return name or f"geom_{geom_id}"


def _is_foot_geom(name: str) -> bool:
    return "toe" in name or "tarsometatarsus" in name


def _load_policy(checkpoint_path: Path):
    checkpoint_path = checkpoint_path.resolve()
    config_path = checkpoint_path / "ppo_network_config.json"
    loaded = json.loads(config_path.read_text())
    kwargs = loaded["network_factory_kwargs"]
    if isinstance(kwargs.get("activation"), str):
        kwargs["activation"] = brax_networks.ACTIVATION[kwargs["activation"]]
    for key in (
        "policy_network_kernel_init_fn",
        "value_network_kernel_init_fn",
        "mean_kernel_init_fn",
    ):
        if isinstance(kwargs.get(key), str):
            kwargs[key] = brax_networks.KERNEL_INITIALIZER[kwargs[key]]
    observation_size = {
        key: tuple(value["shape"])
        for key, value in loaded["observation_size"].items()
    }
    preprocess = (
        running_statistics.normalize
        if loaded["normalize_observations"]
        else lambda x, y: x
    )
    ppo_network = ppo_networks.make_ppo_networks(
        observation_size,
        loaded["action_size"],
        preprocess_observations_fn=preprocess,
        **kwargs,
    )
    params = ppo_checkpoint.load(checkpoint_path)
    return ppo_networks.make_inference_fn(ppo_network)(params, deterministic=True)


def analyze(args: argparse.Namespace) -> None:
    if args.skip_steps >= args.steps:
        raise ValueError("--skip-steps must be smaller than --steps")
    config = trex_getup.default_config()
    config.impl = args.impl
    env = trex_getup.TrexGetup(config)
    model = env.mj_model
    floor_id = model.geom("floor").id
    policy = _load_policy(args.checkpoint)
    jit_policy = jax.jit(policy)
    jit_step = jax.jit(env.step)

    rng = jax.random.PRNGKey(args.seed)
    state = env.reset(rng)
    reward_sum = 0.0
    metric_sums = Counter()
    contact_steps = Counter()
    contact_pairs = Counter()
    sample_steps = 0
    min_non_foot_bottom = np.inf
    min_left_foot_bottom = np.inf
    min_right_foot_bottom = np.inf
    min_torso_height = np.inf
    max_torso_height = -np.inf
    min_orientation = np.inf
    max_orientation = -np.inf
    mean_abs_action = 0.0
    max_abs_action = 0.0
    mean_base_lin_vel = 0.0
    max_base_lin_vel = 0.0
    mean_base_ang_vel = 0.0
    max_base_ang_vel = 0.0
    first_base_xy: np.ndarray | None = None
    last_base_xy: np.ndarray | None = None
    max_base_xy_displacement = 0.0

    non_foot_geom_ids = np.asarray(env._non_foot_geom_ids, dtype=int)
    left_foot_geom_ids = np.asarray(env._left_foot_geom_ids, dtype=int)
    right_foot_geom_ids = np.asarray(env._right_foot_geom_ids, dtype=int)

    for step_index in range(args.steps):
        rng, action_rng = jax.random.split(rng)
        action, _ = jit_policy(state.obs, action_rng)
        state = jit_step(state, action)
        data = jax.device_get(state.data)
        action_np = np.asarray(jax.device_get(action))

        if step_index < args.skip_steps:
            continue

        sample_steps += 1
        reward_sum += float(jax.device_get(state.reward))
        mean_abs_action += float(np.mean(np.abs(action_np)))
        max_abs_action = max(max_abs_action, float(np.max(np.abs(action_np))))
        qvel = np.asarray(data.qvel)
        base_xy = np.asarray(data.qpos[:2])
        if first_base_xy is None:
            first_base_xy = base_xy.copy()
        last_base_xy = base_xy.copy()
        max_base_xy_displacement = max(
            max_base_xy_displacement,
            float(np.linalg.norm(base_xy - first_base_xy)),
        )
        base_lin_vel = float(np.linalg.norm(qvel[:3]))
        base_ang_vel = float(np.linalg.norm(qvel[3:6]))
        mean_base_lin_vel += base_lin_vel
        max_base_lin_vel = max(max_base_lin_vel, base_lin_vel)
        mean_base_ang_vel += base_ang_vel
        max_base_ang_vel = max(max_base_ang_vel, base_ang_vel)
        for key, value in state.metrics.items():
            metric_sums[key] += float(jax.device_get(value))

        non_foot_bottom = _geom_bottom(model, data, non_foot_geom_ids)
        left_foot_bottom = _geom_bottom(model, data, left_foot_geom_ids)
        right_foot_bottom = _geom_bottom(model, data, right_foot_geom_ids)
        min_non_foot_bottom = min(min_non_foot_bottom, float(np.min(non_foot_bottom)))
        min_left_foot_bottom = min(min_left_foot_bottom, float(np.min(left_foot_bottom)))
        min_right_foot_bottom = min(
            min_right_foot_bottom, float(np.min(right_foot_bottom))
        )

        torso_height = float(np.asarray(data.site_xpos)[env._imu_site_id, 2])
        min_torso_height = min(min_torso_height, torso_height)
        max_torso_height = max(max_torso_height, torso_height)
        orientation = float(env._reward_orientation(env.get_gravity(state.data)))
        min_orientation = min(min_orientation, orientation)
        max_orientation = max(max_orientation, orientation)

        data_impl = data._impl
        contact = getattr(data_impl, "contact", None)
        if contact is None:
            continue
        ncon_value = getattr(data_impl, "ncon", None)
        if ncon_value is None:
            ncon_value = getattr(data_impl, "nacon", 0)
        ncon = int(np.asarray(ncon_value))
        floor_contact_names = []
        non_floor_contacts = 0
        non_foot_floor_contacts = 0
        for contact_index in range(ncon):
            if float(np.asarray(contact.dist[contact_index])) > args.contact_margin:
                continue
            geom_a, geom_b = np.asarray(contact.geom[contact_index], dtype=int)
            if floor_id not in (geom_a, geom_b):
                non_floor_contacts += 1
                pair = tuple(sorted((_geom_name(model, geom_a), _geom_name(model, geom_b))))
                contact_pairs[pair] += 1
                continue
            other = geom_b if geom_a == floor_id else geom_a
            name = _geom_name(model, int(other))
            floor_contact_names.append(name)
            if not _is_foot_geom(name):
                non_foot_floor_contacts += 1
                contact_pairs[("floor", name)] += 1

        if floor_contact_names:
            contact_steps["any_floor"] += 1
        if any(_is_foot_geom(name) for name in floor_contact_names):
            contact_steps["foot_floor"] += 1
        if non_foot_floor_contacts:
            contact_steps["non_foot_floor"] += 1
        if non_floor_contacts:
            contact_steps["non_floor"] += 1

    print(f"checkpoint: {args.checkpoint}")
    print(f"steps: {args.steps}")
    print(f"skip_steps: {args.skip_steps}")
    print(f"sample_steps: {sample_steps}")
    print(f"episode_reward_sum: {reward_sum:.3f}")
    print(f"mean_abs_action: {mean_abs_action / sample_steps:.3f}")
    print(f"max_abs_action: {max_abs_action:.3f}")
    print(f"mean_base_lin_vel: {mean_base_lin_vel / sample_steps:.3f}")
    print(f"max_base_lin_vel: {max_base_lin_vel:.3f}")
    print(f"mean_base_ang_vel: {mean_base_ang_vel / sample_steps:.3f}")
    print(f"max_base_ang_vel: {max_base_ang_vel:.3f}")
    if first_base_xy is not None and last_base_xy is not None:
        print(f"base_xy_displacement: {np.linalg.norm(last_base_xy - first_base_xy):.3f}")
        print(f"max_base_xy_displacement: {max_base_xy_displacement:.3f}")
    print(f"torso_height_range: {min_torso_height:.3f} {max_torso_height:.3f}")
    print(f"orientation_reward_range: {min_orientation:.3f} {max_orientation:.3f}")
    print(f"min_non_foot_bottom: {min_non_foot_bottom:.3f}")
    print(f"min_left_foot_bottom: {min_left_foot_bottom:.3f}")
    print(f"min_right_foot_bottom: {min_right_foot_bottom:.3f}")
    print("contact_steps:")
    if sample_steps and not hasattr(data._impl, "contact"):
        print("  skipped: contact data is not exposed by this MJX implementation")
    for key in ("any_floor", "foot_floor", "non_foot_floor", "non_floor"):
        print(f"  {key}: {contact_steps[key]}")
    print("reward_terms:")
    for key in sorted(metric_sums):
        print(f"  {key}: {metric_sums[key]:.3f}")
    if contact_pairs:
        print("non_foot_or_non_floor_contacts:")
        for pair, count in contact_pairs.most_common(args.max_contact_pairs):
            print(f"  {pair}: {count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--impl", default="warp", choices=("jax", "warp"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=750)
    parser.add_argument("--skip-steps", type=int, default=0)
    parser.add_argument("--contact-margin", type=float, default=0.0)
    parser.add_argument("--max-contact-pairs", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
