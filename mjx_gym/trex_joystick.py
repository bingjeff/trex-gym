"""Joystick locomotion task for the simplified T-Rex MJX model."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mjx_gym import trex_constants as consts
from mjx_gym import trex_getup


def default_config() -> config_dict.ConfigDict:
    config = trex_getup.default_config()
    config.episode_length = 1000
    config.reset_standing_prob = 0.5
    config.reset_command_interval_mean = 3.0
    config.command_config = config_dict.create(
        forward_min=-0.25,
        forward_max=2.0,
        turn_max=1.0,
        zero_prob=0.25,
        turn_zero_prob=0.35,
    )
    config.reward_config.tracking_sigma = 0.25
    config.reward_config.turn_tracking_sigma = 0.25
    config.reward_config.scales = config_dict.create(
        orientation=1.0,
        torso_height=1.0,
        non_foot_clearance=1.0,
        foot_support=0.5,
        foot_balance=0.25,
        foot_placement=0.25,
        standing_pose=0.15,
        tracking_forward_vel=2.0,
        tracking_turn_vel=1.0,
        lateral_vel=-0.25,
        vertical_vel=-0.25,
        stand_still=0.5,
        action_rate=-1e-5,
        torques=-1e-9,
        dof_vel=-1e-6,
    )
    return config


class TrexJoystick(trex_getup.TrexGetup):
    """Recover from a fall and track forward velocity / turn-rate commands."""

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(config, config_overrides)
        self._command_zero = jp.zeros(2)

    def reset(self, rng: jax.Array) -> mjx_env.State:
        (
            rng,
            standing_rng,
            yaw_rng,
            xy_rng,
            joint_rng,
            qvel_rng,
            height_rng,
            command_rng,
            interval_rng,
        ) = jax.random.split(rng, 9)
        yaw = jax.random.uniform(
            yaw_rng,
            (),
            minval=-self._config.reset_yaw_range,
            maxval=self._config.reset_yaw_range,
        )
        xy = jax.random.uniform(
            xy_rng,
            (2,),
            minval=-self._config.reset_xy_range,
            maxval=self._config.reset_xy_range,
        )
        joint_noise = jax.random.uniform(
            joint_rng,
            (self.mjx_model.nq - 7,),
            minval=-self._config.reset_joint_noise,
            maxval=self._config.reset_joint_noise,
        )
        height_noise = jax.random.uniform(
            height_rng,
            (),
            minval=0.0,
            maxval=self._config.reset_height_noise,
        )

        side_qpos = self._side_qpos.at[0:2].set(xy)
        side_qpos = side_qpos.at[2].add(height_noise)
        side_qpos = side_qpos.at[3:7].set(trex_getup._yaw_quat(yaw))
        side_qpos = side_qpos.at[7:].set(joint_noise)

        standing_qpos = self._standing_qpos.at[0:2].set(xy)
        standing_qpos = standing_qpos.at[2].add(height_noise)
        standing_quat = trex_getup._quat_mul(
            trex_getup._yaw_quat(yaw), self._standing_qpos[3:7]
        )
        standing_qpos = standing_qpos.at[3:7].set(standing_quat)

        use_standing = jax.random.bernoulli(
            standing_rng, self._config.reset_standing_prob
        )
        qpos = jp.where(use_standing, standing_qpos, side_qpos)
        qvel = jax.random.normal(qvel_rng, (self.mjx_model.nv,)) * (
            self._config.reset_qvel_noise
        )
        data = mjx_env.make_data(
            self.mj_model,
            qpos=qpos,
            qvel=qvel,
            ctrl=self._default_ctrl,
            impl=self.mjx_model.impl.value,
            naconmax=self._config.naconmax,
            njmax=self._config.njmax,
        )
        data = mjx.forward(self.mjx_model, data)

        info = {
            "rng": rng,
            "command": self._sample_command(command_rng),
            "steps_until_next_cmd": self._sample_command_interval(interval_rng),
            "last_act": jp.zeros(self.action_size),
            "last_last_act": jp.zeros(self.action_size),
        }
        metrics = {}
        for key in self._config.reward_config.scales.keys():
            metrics[f"reward/{key}"] = jp.zeros(())
        obs = self._get_obs(data, info)
        return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        clipped_action = jp.clip(action, -1.0, 1.0)
        target_scale = jp.where(
            clipped_action >= 0.0,
            self._action_ctrl_positive_scale,
            self._action_ctrl_negative_scale,
        )
        target = (
            self._action_ctrl_neutral
            + clipped_action * target_scale * self._config.action_scale
        )
        ctrl = self._default_ctrl.at[self._action_actuator_ids].set(target)
        data = mjx_env.step(self.mjx_model, state.data, ctrl, self.n_substeps)
        done = jp.zeros(())
        rewards = self._get_reward(data, action, state.info)
        rewards = {
            key: value * self._config.reward_config.scales[key]
            for key, value in rewards.items()
        }
        reward = jp.clip(
            sum(rewards.values()) * self.dt,
            self._config.reward_clip_min,
            self._config.reward_clip_max,
        )

        state.info["last_last_act"] = state.info["last_act"]
        state.info["last_act"] = action
        state.info["steps_until_next_cmd"] -= 1
        state.info["rng"], command_rng, interval_rng = jax.random.split(
            state.info["rng"], 3
        )
        should_resample = state.info["steps_until_next_cmd"] <= 0
        state.info["command"] = jp.where(
            should_resample,
            self._sample_command(command_rng),
            state.info["command"],
        )
        state.info["steps_until_next_cmd"] = jp.where(
            should_resample,
            self._sample_command_interval(interval_rng),
            state.info["steps_until_next_cmd"],
        )
        obs = self._get_obs(data, state.info)
        for key, value in rewards.items():
            state.metrics[f"reward/{key}"] = value
        return state.replace(data=data, obs=obs, reward=reward, done=done)

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> Dict[str, jax.Array]:
        gyro = self.get_gyro(data)
        gravity = self.get_gravity(data)
        local_linvel = self.get_local_linvel(data)
        local_angvel = self.get_local_angvel(data)
        joint_angles = data.qpos[7:]
        joint_vel = data.qvel[6:]
        state = jp.concatenate(
            [
                local_linvel,
                local_angvel,
                gyro,
                gravity,
                joint_angles,
                joint_vel,
                info["last_act"],
                info["command"],
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
        orientation = self._reward_orientation(gravity)
        height = self._reward_height(torso_height)
        clearance = self._reward_non_foot_clearance(data)
        locomotion_gate = orientation * height * clearance
        local_linvel = self.get_local_linvel(data)
        local_angvel = self.get_local_angvel(data)
        return {
            "orientation": orientation,
            "torso_height": orientation * height,
            "non_foot_clearance": clearance,
            "foot_support": orientation * self._reward_foot_support(data),
            "foot_balance": orientation * self._reward_foot_balance(data),
            "foot_placement": orientation * self._reward_foot_placement(data),
            "standing_pose": orientation * self._reward_standing_pose(data.qpos),
            "tracking_forward_vel": locomotion_gate
            * self._reward_tracking_forward_vel(info["command"], local_linvel),
            "tracking_turn_vel": locomotion_gate
            * self._reward_tracking_turn_vel(info["command"], local_angvel),
            "lateral_vel": locomotion_gate * jp.square(local_linvel[2]),
            "vertical_vel": locomotion_gate * jp.square(local_linvel[1]),
            "stand_still": locomotion_gate
            * self._reward_commanded_stand_still(
                info["command"], local_linvel, local_angvel
            ),
            "action_rate": self._cost_action_rate(action, info),
            "torques": self._cost_torques(data.actuator_force),
            "dof_vel": self._cost_dof_vel(data.qvel[6:]),
        }

    def _sample_command(self, rng: jax.Array) -> jax.Array:
        forward_rng, turn_rng, zero_rng, turn_zero_rng = jax.random.split(rng, 4)
        forward = jax.random.uniform(
            forward_rng,
            (),
            minval=self._config.command_config.forward_min,
            maxval=self._config.command_config.forward_max,
        )
        turn = jax.random.uniform(
            turn_rng,
            (),
            minval=-self._config.command_config.turn_max,
            maxval=self._config.command_config.turn_max,
        )
        turn = jp.where(
            jax.random.bernoulli(
                turn_zero_rng, self._config.command_config.turn_zero_prob
            ),
            0.0,
            turn,
        )
        command = jp.array([forward, turn])
        return jp.where(
            jax.random.bernoulli(zero_rng, self._config.command_config.zero_prob),
            self._command_zero,
            command,
        )

    def _sample_command_interval(self, rng: jax.Array) -> jax.Array:
        interval = (
            jax.random.exponential(rng) * self._config.reset_command_interval_mean
        )
        return jp.maximum(1, jp.round(interval / self.dt)).astype(jp.int32)

    def _reward_tracking_forward_vel(
        self, command: jax.Array, local_linvel: jax.Array
    ) -> jax.Array:
        error = jp.square(command[0] - local_linvel[0])
        return jp.exp(-error / self._config.reward_config.tracking_sigma)

    def _reward_tracking_turn_vel(
        self, command: jax.Array, local_angvel: jax.Array
    ) -> jax.Array:
        error = jp.square(command[1] - local_angvel[1])
        return jp.exp(-error / self._config.reward_config.turn_tracking_sigma)

    def _reward_commanded_stand_still(
        self,
        command: jax.Array,
        local_linvel: jax.Array,
        local_angvel: jax.Array,
    ) -> jax.Array:
        speed = jp.sum(jp.square(local_linvel)) + jp.square(local_angvel[1])
        return (jp.linalg.norm(command) < 0.05) * jp.exp(-2.0 * speed)

    def get_global_linvel(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(self.mj_model, data, consts.GLOBAL_LINVEL_SENSOR)

    def get_global_angvel(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(self.mj_model, data, consts.GLOBAL_ANGVEL_SENSOR)

    def get_local_linvel(self, data: mjx.Data) -> jax.Array:
        torso_xmat = data.site_xmat[self._imu_site_id].reshape((3, 3))
        return torso_xmat.T @ self.get_global_linvel(data)

    def get_local_angvel(self, data: mjx.Data) -> jax.Array:
        torso_xmat = data.site_xmat[self._imu_site_id].reshape((3, 3))
        return torso_xmat.T @ self.get_global_angvel(data)
