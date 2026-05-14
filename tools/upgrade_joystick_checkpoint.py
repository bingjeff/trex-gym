"""Upgrade an older TrexJoystick PPO checkpoint to the current obs shape."""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
from pathlib import Path
import sys

from brax.training.agents.ppo import checkpoint as ppo_checkpoint
from flax.training import orbax_utils
import jax
from ml_collections import config_dict
import numpy as np
from orbax import checkpoint as ocp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mjx_gym import trex_joystick


def _pad_vector(values, new_size: int, fill: float):
    array = np.asarray(values)
    if array.ndim != 1 or array.shape[0] > new_size:
        raise ValueError(f"Cannot pad vector {array.shape} to {new_size}")
    padded = np.full((new_size,), fill, dtype=array.dtype)
    padded[: array.shape[0]] = array
    return padded


def _pad_kernel_rows(kernel, new_rows: int):
    array = np.asarray(kernel)
    if array.ndim != 2 or array.shape[0] > new_rows:
        raise ValueError(f"Cannot pad kernel {array.shape} to {new_rows} rows")
    padded = np.zeros((new_rows, array.shape[1]), dtype=array.dtype)
    padded[: array.shape[0], :] = array
    return padded


def _pad_running_statistics(running_statistics, obs_sizes: dict[str, int]):
    mean = copy.deepcopy(running_statistics.mean)
    std = copy.deepcopy(running_statistics.std)
    summed_variance = copy.deepcopy(running_statistics.summed_variance)

    for key, size in obs_sizes.items():
        mean[key] = _pad_vector(mean[key], size, 0.0)
        std[key] = _pad_vector(std[key], size, 1.0)
        summed_variance[key] = _pad_vector(summed_variance[key], size, 1.0)

    return dataclasses.replace(
        running_statistics,
        mean=mean,
        std=std,
        summed_variance=summed_variance,
    )


def upgrade(args: argparse.Namespace) -> None:
    input_checkpoint = args.input_checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    env = trex_joystick.TrexJoystick()
    obs = env.reset(jax.random.PRNGKey(0)).obs
    obs_sizes = {key: int(value.shape[0]) for key, value in obs.items()}

    params = ppo_checkpoint.load(input_checkpoint)
    if len(params) != 3:
        raise ValueError(f"Expected PPO params list of length 3, got {len(params)}")

    running_statistics, policy_params, value_params = copy.deepcopy(params)
    running_statistics = _pad_running_statistics(running_statistics, obs_sizes)
    policy_params["params"]["hidden_0"]["kernel"] = _pad_kernel_rows(
        policy_params["params"]["hidden_0"]["kernel"], obs_sizes["state"]
    )
    value_params["params"]["hidden_0"]["kernel"] = _pad_kernel_rows(
        value_params["params"]["hidden_0"]["kernel"], obs_sizes["privileged_state"]
    )
    upgraded_params = [running_statistics, policy_params, value_params]

    config = config_dict.ConfigDict(
        json.loads((input_checkpoint / "ppo_network_config.json").read_text())
    )
    config.observation_size = {
        key: {"shape": list(value.shape)} for key, value in obs.items()
    }
    config.action_size = env.action_size

    output_checkpoint = output_dir / f"{args.step:012d}"
    output_checkpoint.mkdir(parents=True, exist_ok=True)
    checkpointer = ocp.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(upgraded_params)
    checkpointer.save(
        output_checkpoint,
        upgraded_params,
        force=True,
        save_args=save_args,
    )
    (output_checkpoint / "ppo_network_config.json").write_text(
        config.to_json_best_effort()
    )
    print(output_checkpoint)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_checkpoint", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--step", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    upgrade(parse_args())
