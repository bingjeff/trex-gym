"""Getup-and-stand task for the simplified T-Rex MJX model."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mjx_gym import trex_constants as consts


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.004,
        Kp=35.0,
        episode_length=300,
        action_repeat=1,
        action_scale=0.25,
        torso_height=1.0,
        reward_config=config_dict.create(
            scales=config_dict.create(
                orientation=1.0,
                torso_height=1.0,
                stand_still=0.25,
                action_rate=-0.001,
                torques=-1e-6,
                dof_vel=-0.01,
            ),
        ),
        impl="jax",
        naconmax=4096,
        njmax=512,
    )


class TrexGetup(mjx_env.MjxEnv):
    """Recover from a side-lying pose and stand.

    The policy controls all generated position actuators: eight leg joint
    targets and two coupled tail tendon targets.
    """

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(config, config_overrides)
        self._mj_model = mujoco.MjModel.from_xml_string(
            consts.trex_getup_xml(position_kp=self._config.Kp)
        )
        self._mj_model.opt.timestep = self._config.sim_dt
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        self._xml_path = str(consts.URDF_PATH)
        self._imu_site_id = self._mj_model.site(consts.IMU_SITE).id
        self._action_actuator_ids = jp.array(
            [self._mj_model.actuator(name).id for name in consts.ACTION_ACTUATORS],
            dtype=jp.int32,
        )
        self._default_ctrl = jp.zeros(self._mj_model.nu)
        self._side_qpos = jp.array(consts.side_lying_qpos(self._mj_model))

    def reset(self, rng: jax.Array) -> mjx_env.State:
        del rng
        qvel = jp.zeros(self.mjx_model.nv)
        data = mjx_env.make_data(
            self.mj_model,
            qpos=self._side_qpos,
            qvel=qvel,
            ctrl=self._default_ctrl,
            impl=self.mjx_model.impl.value,
            naconmax=self._config.naconmax,
            njmax=self._config.njmax,
        )
        data = mjx.forward(self.mjx_model, data)
        info = {
            "last_act": jp.zeros(self.action_size),
            "last_last_act": jp.zeros(self.action_size),
        }
        metrics = {}
        for key in self._config.reward_config.scales.keys():
            metrics[f"reward/{key}"] = jp.zeros(())
        obs = self._get_obs(data, info)
        return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        ctrl = self._default_ctrl.at[self._action_actuator_ids].set(
            action * self._config.action_scale
        )
        data = mjx_env.step(self.mjx_model, state.data, ctrl, self.n_substeps)
        obs = self._get_obs(data, state.info)
        done = jp.zeros(())
        rewards = self._get_reward(data, action, state.info)
        rewards = {
            key: value * self._config.reward_config.scales[key]
            for key, value in rewards.items()
        }
        reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

        state.info["last_last_act"] = state.info["last_act"]
        state.info["last_act"] = action
        for key, value in rewards.items():
            state.metrics[f"reward/{key}"] = value
        return state.replace(data=data, obs=obs, reward=reward, done=done)

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> Dict[str, jax.Array]:
        gyro = self.get_gyro(data)
        gravity = self.get_gravity(data)
        joint_angles = data.qpos[7:]
        joint_vel = data.qvel[6:]
        state = jp.concatenate(
            [
                gyro,
                gravity,
                joint_angles,
                joint_vel,
                info["last_act"],
            ]
        )
        torso_height = data.site_xpos[self._imu_site_id][2:3]
        privileged_state = jp.concatenate(
            [
                state,
                data.qpos,
                data.qvel,
                data.actuator_force,
                torso_height,
            ]
        )
        return {"state": state, "privileged_state": privileged_state}

    def _get_reward(
        self, data: mjx.Data, action: jax.Array, info: dict[str, Any]
    ) -> dict[str, jax.Array]:
        gravity = self.get_gravity(data)
        torso_height = data.site_xpos[self._imu_site_id][2]
        return {
            "orientation": self._reward_orientation(gravity),
            "torso_height": self._reward_height(torso_height),
            "stand_still": self._reward_stand_still(action, gravity, torso_height),
            "action_rate": self._cost_action_rate(action, info),
            "torques": self._cost_torques(data.actuator_force),
            "dof_vel": self._cost_dof_vel(data.qvel[6:]),
        }

    def _reward_orientation(self, gravity: jax.Array) -> jax.Array:
        target = jp.array([0.0, 0.0, -1.0])
        return jp.exp(-2.0 * jp.sum(jp.square(target - gravity)))

    def _reward_height(self, torso_height: jax.Array) -> jax.Array:
        height_error = jp.maximum(self._config.torso_height - torso_height, 0.0)
        return jp.exp(-2.0 * jp.square(height_error))

    def _reward_stand_still(
        self, action: jax.Array, gravity: jax.Array, torso_height: jax.Array
    ) -> jax.Array:
        upright = self._reward_orientation(gravity) > 0.95
        high = torso_height > self._config.torso_height * 0.95
        return (upright * high) * jp.exp(-jp.sum(jp.square(action)))

    def _cost_action_rate(self, action: jax.Array, info: dict[str, Any]) -> jax.Array:
        first = jp.sum(jp.square(action - info["last_act"]))
        second = jp.sum(
            jp.square(action - 2 * info["last_act"] + info["last_last_act"])
        )
        return first + second

    def _cost_torques(self, torques: jax.Array) -> jax.Array:
        return jp.sum(jp.square(torques))

    def _cost_dof_vel(self, qvel: jax.Array) -> jax.Array:
        return jp.sum(jp.square(qvel))

    def get_gyro(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(self.mj_model, data, consts.GYRO_SENSOR)

    def get_gravity(self, data: mjx.Data) -> jax.Array:
        return data.site_xmat[self._imu_site_id].T @ jp.array([0.0, 0.0, -1.0])

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return len(consts.ACTION_ACTUATORS)

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
