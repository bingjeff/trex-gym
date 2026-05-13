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
    config.contact_duty_alpha = 0.02
    config.stand_action_smoothing = 0.5
    config.stand_pose_action = [
        0.0,
        0.0,
        -0.1666667,
        -0.1666667,
        0.1111111,
        0.1111111,
        -0.7083333,
        -0.7083333,
        0.0,
        0.0,
    ]
    config.stand_pose_orientation_threshold = 0.95
    config.stand_pose_height_fraction = 0.90
    config.stand_pose_clearance_threshold = 0.90
    config.command_config = config_dict.create(
        forward_min=0.0,
        forward_max=10.0,
        high_speed_min=7.0,
        high_speed_prob=0.50,
        turn_max=1.0,
        zero_prob=0.15,
        turn_zero_prob=0.25,
    )
    config.reward_config.tracking_sigma = 0.25
    config.reward_config.high_speed_tracking_sigma_scale = 0.5
    config.reward_config.turn_tracking_sigma = 0.25
    config.reward_config.scales = config_dict.create(
        orientation=2.0,
        torso_height=2.0,
        non_foot_clearance=2.0,
        foot_support=2.0,
        foot_balance=1.0,
        foot_placement=1.0,
        standing_pose=1.0,
        tracking_forward_vel=6.0,
        forward_progress=12.0,
        forward_speed_deficit=-20.0,
        tracking_turn_vel=4.0,
        running_stride=0.25,
        running_foot_clearance=0.25,
        gait_anti_phase=2.0,
        gait_symmetry=1.0,
        contact_duty_symmetry=1.0,
        foot_contact_balance=0.5,
        lateral_vel=-0.25,
        vertical_vel=-0.5,
        base_tilt_ang_vel=-1.0,
        foot_slip=-0.2,
        stand_still=4.0,
        standing_base_lin_vel=-10.0,
        standing_base_ang_vel=-5.0,
        standing_foot_vel=-1.0,
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
        self._stand_pose_action = jp.array(self._config.stand_pose_action)

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
            "stand_hold_act": jp.zeros(self.action_size),
            "was_standing_command": jp.zeros(()),
            "last_foot_centers": self._foot_centers_world(data),
            "contact_duty": jp.zeros(2),
        }
        metrics = {}
        for key in self._config.reward_config.scales.keys():
            metrics[f"reward/{key}"] = jp.zeros(())
        obs = self._get_obs(data, info)
        return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        clipped_action = jp.clip(action, -1.0, 1.0)
        standing_gate = self._standing_command_gate(state.info["command"])
        stand_pose_gate = self._stand_pose_gate(state.data, standing_gate)
        start_standing = standing_gate * (1.0 - state.info["was_standing_command"])
        smoothed_stand_act = state.info["stand_hold_act"] + (
            self._config.stand_action_smoothing
            * (clipped_action - state.info["stand_hold_act"])
        )
        stand_hold_act = jp.where(
            start_standing,
            clipped_action,
            smoothed_stand_act,
        )
        applied_stand_action = jp.where(
            stand_pose_gate, self._stand_pose_action, stand_hold_act
        )
        applied_action = jp.where(standing_gate, applied_stand_action, clipped_action)
        target_scale = jp.where(
            applied_action >= 0.0,
            self._action_ctrl_positive_scale,
            self._action_ctrl_negative_scale,
        )
        target = (
            self._action_ctrl_neutral
            + applied_action * target_scale * self._config.action_scale
        )
        ctrl = self._default_ctrl.at[self._action_actuator_ids].set(target)
        data = mjx_env.step(self.mjx_model, state.data, ctrl, self.n_substeps)
        done = jp.zeros(())
        rewards = self._get_reward(data, applied_action, state.info)
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
        state.info["last_act"] = applied_action
        state.info["stand_hold_act"] = jp.where(
            standing_gate, applied_action, clipped_action
        )
        state.info["was_standing_command"] = standing_gate
        state.info["last_foot_centers"] = self._foot_centers_world(data)
        state.info["contact_duty"] = self._updated_contact_duty(data, state.info)
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
                gyro,
                gravity,
                joint_angles,
                joint_vel,
                info["last_act"],
                local_linvel,
                local_angvel,
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
        standing_gate = self._standing_command_gate(info["command"])
        moving_gate = 1.0 - standing_gate
        running_gate = (
            locomotion_gate * moving_gate * self._running_speed_gate(info["command"])
        )
        local_linvel = self.get_local_linvel(data)
        local_angvel = self.get_local_angvel(data)
        achieved_running_gate = running_gate * self._achieved_running_speed_gate(
            info["command"], local_linvel
        )
        return {
            "orientation": orientation,
            "torso_height": orientation * height,
            "non_foot_clearance": clearance,
            "foot_support": standing_gate
            * orientation
            * self._reward_foot_support(data),
            "foot_balance": standing_gate
            * orientation
            * self._reward_foot_balance(data),
            "foot_placement": standing_gate
            * orientation
            * self._reward_foot_placement(data),
            "standing_pose": standing_gate
            * orientation
            * self._reward_standing_pose(data.qpos),
            "tracking_forward_vel": locomotion_gate
            * self._reward_tracking_forward_vel(info["command"], local_linvel),
            "forward_progress": locomotion_gate
            * self._reward_forward_progress(info["command"], local_linvel),
            "forward_speed_deficit": locomotion_gate
            * self._cost_forward_speed_deficit(info["command"], local_linvel),
            "tracking_turn_vel": locomotion_gate
            * self._reward_tracking_turn_vel(info["command"], local_angvel),
            "running_stride": achieved_running_gate * self._reward_running_stride(data),
            "running_foot_clearance": achieved_running_gate
            * self._reward_running_foot_clearance(data),
            "gait_anti_phase": achieved_running_gate
            * self._reward_gait_anti_phase(data),
            "gait_symmetry": achieved_running_gate * self._reward_gait_symmetry(data),
            "contact_duty_symmetry": achieved_running_gate
            * self._reward_contact_duty_symmetry(data, info),
            "foot_contact_balance": achieved_running_gate
            * self._reward_foot_contact_balance(data),
            "lateral_vel": locomotion_gate * jp.square(local_linvel[2]),
            "vertical_vel": locomotion_gate * jp.square(local_linvel[1]),
            "base_tilt_ang_vel": locomotion_gate
            * self._cost_base_tilt_ang_vel(local_angvel),
            "foot_slip": running_gate * self._cost_foot_slip(data, info),
            "stand_still": standing_gate
            * locomotion_gate
            * self._reward_commanded_stand_still(
                info["command"], local_linvel, local_angvel
            ),
            "standing_base_lin_vel": standing_gate
            * locomotion_gate
            * jp.sum(jp.square(local_linvel)),
            "standing_base_ang_vel": standing_gate
            * locomotion_gate
            * jp.sum(jp.square(local_angvel)),
            "standing_foot_vel": standing_gate
            * locomotion_gate
            * self._cost_foot_vel(data, info),
            "action_rate": self._cost_action_rate(action, info),
            "torques": self._cost_torques(data.actuator_force),
            "dof_vel": self._cost_dof_vel(data.qvel[6:]),
        }

    def _sample_command(self, rng: jax.Array) -> jax.Array:
        (
            forward_rng,
            high_speed_rng,
            high_speed_choice_rng,
            turn_rng,
            zero_rng,
            turn_zero_rng,
        ) = jax.random.split(rng, 6)
        forward = jax.random.uniform(
            forward_rng,
            (),
            minval=self._config.command_config.forward_min,
            maxval=self._config.command_config.forward_max,
        )
        high_speed_forward = jax.random.uniform(
            high_speed_rng,
            (),
            minval=self._config.command_config.high_speed_min,
            maxval=self._config.command_config.forward_max,
        )
        forward = jp.where(
            jax.random.bernoulli(
                high_speed_choice_rng, self._config.command_config.high_speed_prob
            ),
            high_speed_forward,
            forward,
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
        high_speed = jp.maximum(jp.abs(command[0]) - 2.0, 0.0)
        sigma = (
            self._config.reward_config.tracking_sigma
            + self._config.reward_config.high_speed_tracking_sigma_scale
            * jp.square(high_speed)
        )
        return jp.exp(-error / sigma)

    def _reward_forward_progress(
        self, command: jax.Array, local_linvel: jax.Array
    ) -> jax.Array:
        commanded_forward = jp.maximum(command[0], 0.0)
        moving_forward = commanded_forward > 0.05
        speed_fraction = local_linvel[0] / jp.maximum(commanded_forward, 1.0)
        return moving_forward * jp.clip(speed_fraction, 0.0, 1.0)

    def _cost_forward_speed_deficit(
        self, command: jax.Array, local_linvel: jax.Array
    ) -> jax.Array:
        commanded_forward = jp.maximum(command[0], 0.0)
        moving_forward = commanded_forward > 0.05
        deficit = jp.maximum(commanded_forward - local_linvel[0], 0.0)
        normalized = deficit / jp.maximum(commanded_forward, 1.0)
        return moving_forward * jp.square(normalized)

    def _reward_tracking_turn_vel(
        self, command: jax.Array, local_angvel: jax.Array
    ) -> jax.Array:
        error = jp.square(command[1] - local_angvel[1])
        return jp.exp(-error / self._config.reward_config.turn_tracking_sigma)

    def _cost_base_tilt_ang_vel(self, local_angvel: jax.Array) -> jax.Array:
        return jp.square(local_angvel[0]) + jp.square(local_angvel[2])

    def _reward_commanded_stand_still(
        self,
        command: jax.Array,
        local_linvel: jax.Array,
        local_angvel: jax.Array,
    ) -> jax.Array:
        speed = jp.sum(jp.square(local_linvel)) + jp.sum(jp.square(local_angvel))
        return self._standing_command_gate(command) * jp.exp(-10.0 * speed)

    def _standing_command_gate(self, command: jax.Array) -> jax.Array:
        return (jp.linalg.norm(command) < 0.05).astype(jp.float32)

    def _stand_pose_gate(self, data: mjx.Data, standing_gate: jax.Array) -> jax.Array:
        orientation = self._reward_orientation(self.get_gravity(data))
        torso_height = data.site_xpos[self._imu_site_id, 2]
        clearance = self._reward_non_foot_clearance(data)
        ready = (
            (orientation > self._config.stand_pose_orientation_threshold)
            & (
                torso_height
                > self._target_torso_height * self._config.stand_pose_height_fraction
            )
            & (clearance > self._config.stand_pose_clearance_threshold)
        )
        return standing_gate * ready.astype(jp.float32)

    def _running_speed_gate(self, command: jax.Array) -> jax.Array:
        return jp.clip((jp.abs(command[0]) - 1.0) / 4.0, 0.0, 1.0)

    def _achieved_running_speed_gate(
        self, command: jax.Array, local_linvel: jax.Array
    ) -> jax.Array:
        commanded_forward = jp.maximum(command[0], 1.0)
        speed_fraction = local_linvel[0] / commanded_forward
        return jp.clip((speed_fraction - 0.35) / 0.4, 0.0, 1.0)

    def _reward_running_stride(self, data: mjx.Data) -> jax.Array:
        left_offset, right_offset = self._mjx_foot_offsets_in_torso_frame(data)
        left_step = jp.abs(left_offset[0] - self._standing_left_foot_offset[0])
        right_step = jp.abs(right_offset[0] - self._standing_right_foot_offset[0])
        stride_extent = 0.5 * (left_step + right_step)
        return jp.clip(stride_extent / 0.8, 0.0, 1.0)

    def _reward_running_foot_clearance(self, data: mjx.Data) -> jax.Array:
        left_clearance = jp.max(
            jp.clip(self._geom_bottom(data, self._left_foot_geom_ids), 0.0, 0.5)
        )
        right_clearance = jp.max(
            jp.clip(self._geom_bottom(data, self._right_foot_geom_ids), 0.0, 0.5)
        )
        clearance = 0.5 * (left_clearance + right_clearance)
        return jp.clip(clearance / 0.25, 0.0, 1.0)

    def _reward_gait_anti_phase(self, data: mjx.Data) -> jax.Array:
        left_offset, right_offset = self._mjx_foot_offsets_in_torso_frame(data)
        left_step = left_offset[0] - self._standing_left_foot_offset[0]
        right_step = right_offset[0] - self._standing_right_foot_offset[0]
        foot_phase = self._anti_phase_score(left_step, right_step, epsilon=0.05)

        qpos = data.qpos[self._leg_qpos_ids]
        standing = self._standing_leg_qpos
        pair_scores = jp.array(
            [
                self._anti_phase_score(qpos[2] - standing[2], qpos[3] - standing[3]),
                self._anti_phase_score(qpos[4] - standing[4], qpos[5] - standing[5]),
                self._anti_phase_score(qpos[6] - standing[6], qpos[7] - standing[7]),
            ]
        )
        joint_phase = jp.mean(pair_scores)
        return self._stride_gate(left_step, right_step) * (
            0.6 * foot_phase + 0.4 * joint_phase
        )

    def _reward_gait_symmetry(self, data: mjx.Data) -> jax.Array:
        left_offset, right_offset = self._mjx_foot_offsets_in_torso_frame(data)
        left_step = left_offset[0] - self._standing_left_foot_offset[0]
        right_step = right_offset[0] - self._standing_right_foot_offset[0]
        foot_symmetry = jp.exp(-2.0 * jp.square(jp.abs(left_step) - jp.abs(right_step)))

        qpos = data.qpos[self._leg_qpos_ids]
        standing = self._standing_leg_qpos
        pair_errors = jp.array(
            [
                jp.abs(qpos[2] - standing[2]) - jp.abs(qpos[3] - standing[3]),
                jp.abs(qpos[4] - standing[4]) - jp.abs(qpos[5] - standing[5]),
                jp.abs(qpos[6] - standing[6]) - jp.abs(qpos[7] - standing[7]),
            ]
        )
        joint_symmetry = jp.exp(-2.0 * jp.mean(jp.square(pair_errors)))
        return self._stride_gate(left_step, right_step) * (
            0.6 * foot_symmetry + 0.4 * joint_symmetry
        )

    def _reward_foot_contact_balance(self, data: mjx.Data) -> jax.Array:
        left_contact, right_contact = self._foot_contact_scores(data)
        one_foot_stance = (
            left_contact + right_contact - 2.0 * left_contact * right_contact
        )
        any_contact = jp.clip(left_contact + right_contact, 0.0, 1.0)
        return 0.5 * any_contact + 0.5 * one_foot_stance

    def _reward_contact_duty_symmetry(
        self, data: mjx.Data, info: dict[str, Any]
    ) -> jax.Array:
        duty = self._updated_contact_duty(data, info)
        total = jp.sum(duty)
        symmetry = jp.exp(-8.0 * jp.square(duty[0] - duty[1]))
        one_foot_average = jp.exp(-4.0 * jp.square(total - 1.0))
        active = jp.clip(total / 0.2, 0.0, 1.0)
        return active * symmetry * one_foot_average

    def _updated_contact_duty(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        contacts = jp.array(self._foot_contact_scores(data))
        alpha = self._config.contact_duty_alpha
        return (1.0 - alpha) * info["contact_duty"] + alpha * contacts

    def _anti_phase_score(
        self, left: jax.Array, right: jax.Array, epsilon: float = 0.02
    ) -> jax.Array:
        phase = -(left * right) / (jp.abs(left) * jp.abs(right) + epsilon)
        return jp.clip(phase, 0.0, 1.0)

    def _stride_gate(self, left_step: jax.Array, right_step: jax.Array) -> jax.Array:
        stride = 0.5 * (jp.abs(left_step) + jp.abs(right_step))
        return jp.clip((stride - 0.15) / 0.65, 0.0, 1.0)

    def _foot_contact_scores(self, data: mjx.Data) -> tuple[jax.Array, jax.Array]:
        left_bottom = jp.min(jp.abs(self._geom_bottom(data, self._left_foot_geom_ids)))
        right_bottom = jp.min(
            jp.abs(self._geom_bottom(data, self._right_foot_geom_ids))
        )
        return (
            jp.exp(-200.0 * jp.square(left_bottom)),
            jp.exp(-200.0 * jp.square(right_bottom)),
        )

    def _foot_centers_world(self, data: mjx.Data) -> jax.Array:
        left_center = jp.mean(data.geom_xpos[self._left_foot_geom_ids], axis=0)
        right_center = jp.mean(data.geom_xpos[self._right_foot_geom_ids], axis=0)
        return jp.stack([left_center, right_center])

    def _cost_foot_slip(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        foot_delta = self._foot_centers_world(data) - info["last_foot_centers"]
        foot_vel = foot_delta / self.dt
        horizontal_speed_sq = jp.sum(jp.square(foot_vel[:, :2]), axis=1)
        left_contact, right_contact = self._foot_contact_scores(data)
        contact = jp.stack([left_contact, right_contact])
        return jp.sum(contact * horizontal_speed_sq) / (jp.sum(contact) + 1.0e-6)

    def _cost_foot_vel(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        foot_delta = self._foot_centers_world(data) - info["last_foot_centers"]
        foot_vel = foot_delta / self.dt
        return jp.sum(jp.square(foot_vel))

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
