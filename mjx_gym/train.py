"""Training entry point that registers local T-Rex Playground environments."""

from absl import flags
from absl.flags import _exceptions as flags_exceptions
from ml_collections import config_dict
from learning import train_jax_ppo
from mujoco_playground import registry
from mujoco_playground._src import locomotion

from mjx_gym import trex_getup
from mjx_gym import trex_joystick

_TREX_NUM_RESETS_PER_EVAL = flags.DEFINE_integer(
    "trex_num_resets_per_eval",
    None,
    "Override TrexGetup PPO num_resets_per_eval.",
)


def register_environments() -> None:
    locomotion.register_environment(
        "TrexGetup",
        trex_getup.TrexGetup,
        trex_getup.default_config,
    )
    locomotion.register_environment(
        "TrexJoystick",
        trex_joystick.TrexJoystick,
        trex_joystick.default_config,
    )
    registry.ALL_ENVS = (
        registry.dm_control_suite.ALL_ENVS
        + locomotion.ALL_ENVS
        + registry.manipulation.ALL_ENVS
    )


def trex_ppo_config(env_name: str, impl: str | None = None) -> config_dict.ConfigDict:
    del impl
    env_config = (
        trex_joystick.default_config()
        if env_name == "TrexJoystick"
        else trex_getup.default_config()
    )
    try:
        num_resets_per_eval = _TREX_NUM_RESETS_PER_EVAL.value
    except flags_exceptions.UnparsedFlagAccessError:
        num_resets_per_eval = None
    if num_resets_per_eval is None:
        num_resets_per_eval = 10
    return config_dict.create(
        num_timesteps=50_000_000,
        num_evals=5,
        reward_scaling=1.0,
        episode_length=env_config.episode_length,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=20,
        num_minibatches=8,
        num_updates_per_batch=4,
        discounting=0.97,
        learning_rate=3e-4,
        entropy_cost=1e-2,
        num_envs=1024,
        batch_size=256,
        max_grad_norm=1.0,
        network_factory=config_dict.create(
            policy_hidden_layer_sizes=(128, 128, 128),
            value_hidden_layer_sizes=(256, 256, 256),
            policy_obs_key="state",
            value_obs_key="privileged_state",
        ),
        num_resets_per_eval=num_resets_per_eval,
    )


def patch_training_config() -> None:
    original_get_rl_config = train_jax_ppo.get_rl_config

    def get_rl_config(env_name: str) -> config_dict.ConfigDict:
        if env_name in ("TrexGetup", "TrexJoystick"):
            return trex_ppo_config(env_name, train_jax_ppo._IMPL.value)
        return original_get_rl_config(env_name)

    train_jax_ppo.get_rl_config = get_rl_config


def main() -> None:
    register_environments()
    patch_training_config()
    train_jax_ppo.run()


if __name__ == "__main__":
    main()
