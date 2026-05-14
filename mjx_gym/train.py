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
    "Override T-Rex PPO num_resets_per_eval.",
)


_TASKS = {
    "TrexGetup": (trex_getup.TrexGetup, trex_getup.default_config),
    "TrexBalance": (trex_joystick.TrexBalance, trex_joystick.balance_config),
    "TrexWalk": (trex_joystick.TrexWalk, trex_joystick.walk_config),
    "TrexJoystick": (trex_joystick.TrexJoystick, trex_joystick.joystick_config),
    "TrexRun": (trex_joystick.TrexRun, trex_joystick.run_config),
}


_PPO_TIMESTEPS = {
    "TrexGetup": 50_000_000,
    "TrexBalance": 50_000_000,
    "TrexWalk": 100_000_000,
    "TrexJoystick": 150_000_000,
    "TrexRun": 150_000_000,
}


def register_environments() -> None:
    for name, (env_cls, config_fn) in _TASKS.items():
        locomotion.register_environment(name, env_cls, config_fn)
    registry.ALL_ENVS = (
        registry.dm_control_suite.ALL_ENVS
        + locomotion.ALL_ENVS
        + registry.manipulation.ALL_ENVS
    )


def trex_ppo_config(env_name: str, impl: str | None = None) -> config_dict.ConfigDict:
    del impl
    env_config = _TASKS[env_name][1]()
    try:
        num_resets_per_eval = _TREX_NUM_RESETS_PER_EVAL.value
    except flags_exceptions.UnparsedFlagAccessError:
        num_resets_per_eval = None
    if num_resets_per_eval is None:
        num_resets_per_eval = 10
    return config_dict.create(
        num_timesteps=_PPO_TIMESTEPS[env_name],
        num_evals=10,
        reward_scaling=1.0,
        episode_length=env_config.episode_length,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=20,
        num_minibatches=32,
        num_updates_per_batch=4,
        discounting=0.97,
        learning_rate=3e-4,
        entropy_cost=5e-3,
        num_envs=4096,
        batch_size=1024,
        clipping_epsilon=0.2,
        max_grad_norm=1.0,
        restore_value_fn=False,
        network_factory=config_dict.create(
            policy_hidden_layer_sizes=(512, 256, 128),
            value_hidden_layer_sizes=(512, 256, 128),
            policy_obs_key="state",
            value_obs_key="privileged_state",
        ),
        num_resets_per_eval=num_resets_per_eval,
    )


def patch_training_config() -> None:
    original_get_rl_config = train_jax_ppo.get_rl_config

    def get_rl_config(env_name: str) -> config_dict.ConfigDict:
        if env_name in _TASKS:
            return trex_ppo_config(env_name, train_jax_ppo._IMPL.value)
        return original_get_rl_config(env_name)

    train_jax_ppo.get_rl_config = get_rl_config


def main() -> None:
    register_environments()
    patch_training_config()
    train_jax_ppo.run()


if __name__ == "__main__":
    main()
