"""Convert a TrexGetup PPO checkpoint into a TrexJoystick warm start."""

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
        raise ValueError(
            f"Expected vector no larger than {(new_size,)}, got {array.shape}"
        )
    padded = np.full((new_size,), fill, dtype=array.dtype)
    padded[: array.shape[0]] = array
    return padded


def _pad_kernel_rows(kernel, new_rows: int):
    array = np.asarray(kernel)
    if array.ndim != 2 or array.shape[0] > new_rows:
        raise ValueError(
            f"Expected no more than {new_rows} input rows, got {array.shape}"
        )
    padded = np.zeros((new_rows, array.shape[1]), dtype=array.dtype)
    padded[: array.shape[0], :] = array
    return padded


def _pad_running_statistics(running_statistics):
    mean = copy.deepcopy(running_statistics.mean)
    std = copy.deepcopy(running_statistics.std)
    summed_variance = copy.deepcopy(running_statistics.summed_variance)

    mean["state"] = _pad_vector(mean["state"], 88, 0.0)
    mean["privileged_state"] = _pad_vector(mean["privileged_state"], 174, 0.0)
    std["state"] = _pad_vector(std["state"], 88, 1.0)
    std["privileged_state"] = _pad_vector(std["privileged_state"], 174, 1.0)
    summed_variance["state"] = _pad_vector(summed_variance["state"], 88, 1.0)
    summed_variance["privileged_state"] = _pad_vector(
        summed_variance["privileged_state"], 174, 1.0
    )

    return dataclasses.replace(
        running_statistics,
        mean=mean,
        std=std,
        summed_variance=summed_variance,
    )


def convert(args: argparse.Namespace) -> None:
    params = ppo_checkpoint.load(args.input_checkpoint)
    if len(params) != 3:
        raise ValueError(f"Expected PPO params list of length 3, got {len(params)}")

    running_statistics, policy_params, value_params = copy.deepcopy(params)
    running_statistics = _pad_running_statistics(running_statistics)
    policy_params["params"]["hidden_0"]["kernel"] = _pad_kernel_rows(
        policy_params["params"]["hidden_0"]["kernel"], 88
    )
    value_params["params"]["hidden_0"]["kernel"] = _pad_kernel_rows(
        value_params["params"]["hidden_0"]["kernel"], 174
    )
    converted_params = [running_statistics, policy_params, value_params]

    config = config_dict.ConfigDict(
        json.loads((args.input_checkpoint / "ppo_network_config.json").read_text())
    )
    env = trex_joystick.TrexJoystick()
    obs = env.reset(jax.random.PRNGKey(0)).obs
    config.observation_size = {
        key: {"shape": list(value.shape)} for key, value in obs.items()
    }
    config.action_size = env.action_size

    output_checkpoint = args.output_dir / f"{args.step:012d}"
    output_checkpoint.mkdir(parents=True, exist_ok=True)
    checkpointer = ocp.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(converted_params)
    checkpointer.save(
        output_checkpoint,
        converted_params,
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
    convert(parse_args())
