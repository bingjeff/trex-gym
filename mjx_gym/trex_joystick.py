"""Joystick locomotion task for the simplified T-Rex MJX model."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground._src import gait
from mujoco_playground._src import mjx_env
from mjx_gym import trex_constants as consts
from mjx_gym import trex_getup


def default_config() -> config_dict.ConfigDict:
    config = trex_getup.default_config()
    config.episode_length = 1000
    config.reset_standing_prob = 0.5
    config.reset_command_interval_mean = 3.0
    config.curriculum_task = "joystick"
    config.march_command_forward = 0.1
    config.contact_duty_alpha = 0.02
    config.stand_action_smoothing = 0.5
    config.terminate_on_fall = False
    config.fall_orientation_threshold = 0.35
    config.fall_torso_height = 1.2
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
    config.running_action_residual_scale = [
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        1.0,
        1.0,
    ]
    config.gait_prior_scale = 0.35
    config.gait_frequency_min = 1.0
    config.gait_frequency_per_mps = 0.15
    config.gait_frequency_max = 2.5
    config.gait_swing_height = 0.12
    config.random_initial_gait_phase = True
    config.running_gate_start = 0.05
    config.running_gate_full = 0.25
    config.foot_contact_force_scale = 20000.0
    config.foot_contact_height = 0.03
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
        first_step_forward_progress=20.0,
        first_step_contact_balance=2.0,
        first_step_contact_duty_symmetry=2.0,
        forward_progress=12.0,
        forward_speed_deficit=-20.0,
        moving_forward_vel_error=-1.0,
        tracking_turn_vel=4.0,
        running_stride=0.25,
        running_foot_clearance=0.25,
        gait_prior_tracking=1.0,
        leg_action_alternation=2.0,
        gait_anti_phase=2.0,
        gait_symmetry=1.0,
        phase_contact=1.0,
        phase_contact_error=-2.0,
        phase_foot_clearance=1.0,
        phase_swing_clearance=1.0,
        phase_swing_release=1.0,
        phase_stance_contact=1.0,
        feet_phase_height=1.0,
        phase_clearance_error=-1.0,
        phase_clearance_max_error=-1.0,
        single_support_balance=1.0,
        feet_air_time=0.5,
        contact_duty_symmetry=1.0,
        foot_contact_balance=0.5,
        double_foot_contact=-2.0,
        lateral_vel=-0.25,
        vertical_vel=-0.5,
        base_tilt_ang_vel=-1.0,
        moving_orientation=-20.0,
        moving_torso_height=-20.0,
        moving_non_foot_clearance=-20.0,
        moving_lateral_vel=-1.0,
        moving_vertical_vel=-2.0,
        moving_action_deviation=-20.0,
        contact_duty_error=-8.0,
        no_foot_contact=-12.0,
        running_height_excess=-8.0,
        foot_slip=-0.2,
        hip_adduction_neutral=-0.5,
        fall=-1000.0,
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
        self._running_action_residual_scale = jp.array(
            self._config.running_action_residual_scale
        )
        self._floor_geom_id = self._mj_model.geom("floor").id
        self._standing_support_offset_xz = 0.5 * jp.array(
            [
                self._standing_left_foot_offset[0] + self._standing_right_foot_offset[0],
                self._standing_left_foot_offset[2] + self._standing_right_foot_offset[2],
            ]
        )

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
            phase_rng,
        ) = jax.random.split(rng, 10)
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

        command = self._sample_command(command_rng)
        if self._is_march_task():
            command = self._march_command()
        random_phase = jax.random.uniform(
            phase_rng, (), minval=0.0, maxval=2.0 * jp.pi
        )
        initial_gait_phase = jp.where(
            self._config.random_initial_gait_phase
            & (self._standing_command_gate(command) < 0.5),
            random_phase,
            0.0,
        )

        info = {
            "rng": rng,
            "command": command,
            "steps_until_next_cmd": self._sample_command_interval(interval_rng),
            "last_act": jp.zeros(self.action_size),
            "last_last_act": jp.zeros(self.action_size),
            "stand_hold_act": jp.zeros(self.action_size),
            "was_standing_command": jp.zeros(()),
            "last_foot_centers": self._foot_centers_world(data),
            "contact_duty": jp.zeros(2),
            "feet_air_time": jp.zeros(2),
            "last_contact": jp.zeros(2, dtype=bool),
            "gait_phase": initial_gait_phase,
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
        running_action = jp.clip(
            self._stand_pose_action
            + clipped_action * self._running_action_residual_scale
            + self._gait_prior_action(state.info),
            -1.0,
            1.0,
        )
        applied_action = jp.where(standing_gate, applied_stand_action, running_action)
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
        contact = jp.array(self._foot_contact_scores(data)) > 0.2
        contact_filt = contact | state.info["last_contact"]
        feet_air_time = state.info["feet_air_time"] + self.dt
        first_contact = (feet_air_time > 0.0) & contact_filt
        done = self._fall_done(data)
        rewards = self._get_reward(
            data, applied_action, state.info, first_contact, feet_air_time
        )
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
        state.info["feet_air_time"] = feet_air_time * ~contact
        state.info["last_contact"] = contact
        state.info["gait_phase"] = self._updated_gait_phase(state.info, standing_gate)
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
        if self._is_march_task():
            state.info["command"] = self._march_command()
            state.info["steps_until_next_cmd"] = self._config.episode_length
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
        gait_phase = jp.array(
            [jp.sin(info["gait_phase"]), jp.cos(info["gait_phase"])]
        )
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
                gait_phase,
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
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        first_contact: jax.Array,
        feet_air_time: jax.Array,
    ) -> dict[str, jax.Array]:
        if self._is_march_task():
            return self._get_march_reward(
                data, action, info, first_contact, feet_air_time
            )

        gravity = self.get_gravity(data)
        torso_height = data.site_xpos[self._imu_site_id][2]
        orientation = self._reward_orientation(gravity)
        height = self._reward_height(torso_height)
        clearance = self._reward_non_foot_clearance(data)
        locomotion_gate = orientation * height * clearance
        standing_gate = self._standing_command_gate(info["command"])
        moving_gate = 1.0 - standing_gate
        running_height_gate = self._reward_running_height_gate(torso_height)
        running_gate = (
            locomotion_gate
            * moving_gate
            * running_height_gate
            * self._running_speed_gate(info["command"])
        )
        local_linvel = self.get_local_linvel(data)
        local_angvel = self.get_local_angvel(data)
        moving_support_gate = self._moving_foot_support_gate(data)
        speed_tracking_gate = standing_gate + (
            moving_gate * moving_support_gate * running_height_gate
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
            * speed_tracking_gate
            * self._reward_tracking_forward_vel(info["command"], local_linvel),
            "first_step_forward_progress": moving_gate
            * locomotion_gate
            * self._reward_first_step_forward_progress(info["command"], local_linvel),
            "first_step_contact_balance": moving_gate
            * locomotion_gate
            * self._reward_foot_contact_balance(data),
            "first_step_contact_duty_symmetry": moving_gate
            * locomotion_gate
            * self._reward_contact_duty_symmetry(data, info),
            "forward_progress": locomotion_gate
            * speed_tracking_gate
            * self._reward_forward_progress(info["command"], local_linvel),
            "forward_speed_deficit": moving_gate
            * self._running_speed_gate(info["command"])
            * self._cost_forward_speed_deficit(info["command"], local_linvel),
            "moving_forward_vel_error": moving_gate
            * self._cost_forward_speed_error(info["command"], local_linvel),
            "tracking_turn_vel": locomotion_gate
            * speed_tracking_gate
            * self._reward_tracking_turn_vel(info["command"], local_angvel),
            "running_stride": running_gate * self._reward_running_stride(data),
            "running_foot_clearance": running_gate
            * self._reward_running_foot_clearance(data),
            "gait_prior_tracking": moving_gate
            * self._reward_gait_prior_tracking(action, info),
            "leg_action_alternation": running_gate
            * self._reward_leg_action_alternation(action),
            "gait_anti_phase": running_gate
            * self._reward_gait_anti_phase(data),
            "gait_symmetry": running_gate * self._reward_gait_symmetry(data),
            "phase_contact": running_gate * self._reward_phase_contact(data, info),
            "phase_contact_error": running_gate
            * self._cost_phase_contact_error(data, info),
            "phase_foot_clearance": running_gate
            * self._reward_phase_foot_clearance(data, info),
            "phase_swing_clearance": moving_gate
            * self._reward_phase_swing_clearance(data, info["gait_phase"]),
            "phase_swing_release": moving_gate
            * self._reward_phase_swing_release(data, info["gait_phase"]),
            "phase_stance_contact": moving_gate
            * self._reward_phase_stance_contact(data, info["gait_phase"]),
            "feet_phase_height": moving_gate
            * self._reward_feet_phase_height(data, info["gait_phase"], info["command"]),
            "phase_clearance_error": moving_gate
            * self._cost_phase_clearance_error(data, info["gait_phase"]),
            "phase_clearance_max_error": moving_gate
            * self._cost_phase_clearance_max_error(data, info["gait_phase"]),
            "single_support_balance": moving_gate
            * self._reward_single_support_balance(data, info["gait_phase"]),
            "feet_air_time": running_gate
            * self._reward_feet_air_time(feet_air_time, first_contact, info["command"]),
            "contact_duty_symmetry": running_gate
            * self._reward_contact_duty_symmetry(data, info),
            "foot_contact_balance": running_gate
            * self._reward_foot_contact_balance(data),
            "double_foot_contact": moving_gate
            * self._running_speed_gate(info["command"])
            * self._cost_double_foot_contact(data),
            "lateral_vel": locomotion_gate * jp.square(local_linvel[2]),
            "vertical_vel": locomotion_gate * jp.square(local_linvel[1]),
            "base_tilt_ang_vel": locomotion_gate
            * self._cost_base_tilt_ang_vel(local_angvel),
            "moving_orientation": moving_gate
            * jp.square(1.0 - orientation),
            "moving_torso_height": moving_gate
            * jp.square(1.0 - height),
            "moving_non_foot_clearance": moving_gate
            * jp.square(1.0 - clearance),
            "moving_lateral_vel": moving_gate * jp.square(local_linvel[2]),
            "moving_vertical_vel": moving_gate * jp.square(local_linvel[1]),
            "moving_action_deviation": moving_gate
            * (1.0 - locomotion_gate)
            * self._cost_moving_action_deviation(action),
            "contact_duty_error": moving_gate
            * self._cost_contact_duty_error(data, info),
            "no_foot_contact": moving_gate
            * self._cost_no_foot_contact(data),
            "running_height_excess": moving_gate
            * self._running_speed_gate(info["command"])
            * self._cost_running_height_excess(torso_height),
            "foot_slip": running_gate * self._cost_foot_slip(data, info),
            "hip_adduction_neutral": moving_gate
            * self._cost_hip_adduction_neutral(data),
            "fall": self._fall_done(data),
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

    def _get_march_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        first_contact: jax.Array,
        feet_air_time: jax.Array,
    ) -> dict[str, jax.Array]:
        gravity = self.get_gravity(data)
        torso_height = data.site_xpos[self._imu_site_id][2]
        orientation = self._reward_orientation(gravity)
        height = self._reward_height(torso_height)
        clearance = self._reward_non_foot_clearance(data)
        posture_gate = orientation * height * clearance
        local_linvel = self.get_local_linvel(data)
        local_angvel = self.get_local_angvel(data)
        return {
            "orientation": orientation,
            "torso_height": orientation * height,
            "non_foot_clearance": clearance,
            "phase_swing_clearance": self._reward_phase_swing_clearance(
                data, info["gait_phase"]
            ),
            "phase_swing_release": self._reward_phase_swing_release(
                data, info["gait_phase"]
            ),
            "phase_stance_contact": self._reward_phase_stance_contact(
                data, info["gait_phase"]
            ),
            "feet_phase_height": self._reward_feet_phase_height(
                data, info["gait_phase"], info["command"]
            ),
            "phase_clearance_error": self._cost_phase_clearance_error(
                data, info["gait_phase"]
            ),
            "phase_clearance_max_error": self._cost_phase_clearance_max_error(
                data, info["gait_phase"]
            ),
            "single_support_balance": self._reward_single_support_balance(
                data, info["gait_phase"]
            ),
            "feet_air_time": self._reward_feet_air_time(
                feet_air_time, first_contact, info["command"]
            ),
            "phase_contact": self._reward_phase_contact(data, info),
            "phase_contact_error": self._cost_phase_contact_error(data, info),
            "phase_foot_clearance": self._reward_phase_foot_clearance(data, info),
            "contact_duty_symmetry": self._reward_contact_duty_symmetry(data, info),
            "contact_duty_error": self._cost_contact_duty_error(data, info),
            "foot_contact_balance": self._reward_foot_contact_balance(data),
            "double_foot_contact": self._cost_double_foot_contact(data),
            "gait_prior_tracking": self._reward_gait_prior_tracking(action, info),
            "leg_action_alternation": self._reward_leg_action_alternation(action),
            "gait_anti_phase": self._reward_gait_anti_phase(data),
            "gait_symmetry": self._reward_gait_symmetry(data),
            "moving_lateral_vel": jp.square(local_linvel[2]),
            "moving_vertical_vel": jp.square(local_linvel[1]),
            "moving_forward_vel_error": jp.square(local_linvel[0]),
            "base_tilt_ang_vel": self._cost_base_tilt_ang_vel(local_angvel),
            "moving_orientation": jp.square(1.0 - orientation),
            "moving_torso_height": jp.square(1.0 - height),
            "moving_non_foot_clearance": jp.square(1.0 - clearance),
            "no_foot_contact": self._cost_no_foot_contact(data),
            "running_height_excess": self._cost_running_height_excess(torso_height),
            "foot_slip": posture_gate * self._cost_foot_slip(data, info),
            "hip_adduction_neutral": self._cost_hip_adduction_neutral(data),
            "fall": self._fall_done(data),
            "action_rate": self._cost_action_rate(action, info),
            "torques": self._cost_torques(data.actuator_force),
            "dof_vel": self._cost_dof_vel(data.qvel[6:]),
        }

    def _is_march_task(self) -> bool:
        return self._config.curriculum_task == "march"

    def _march_command(self) -> jax.Array:
        return jp.array([self._config.march_command_forward, 0.0])

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

    def _gait_prior_action(self, info: dict[str, Any]) -> jax.Array:
        phase = info["gait_phase"]
        right = jp.sin(phase)
        left = -right
        gait = jp.zeros(self.action_size)
        gait = gait.at[2].set(right)
        gait = gait.at[3].set(left)
        gait = gait.at[4].set(-right)
        gait = gait.at[5].set(-left)
        gait = gait.at[6].set(0.75 * right)
        gait = gait.at[7].set(0.75 * left)
        moving_gate = 1.0 - self._standing_command_gate(info["command"])
        speed_gate = self._running_speed_gate(info["command"])
        return moving_gate * speed_gate * self._config.gait_prior_scale * gait

    def _updated_gait_phase(
        self, info: dict[str, Any], standing_gate: jax.Array
    ) -> jax.Array:
        frequency = jp.clip(
            self._config.gait_frequency_min
            + self._config.gait_frequency_per_mps * jp.maximum(info["command"][0], 0.0),
            self._config.gait_frequency_min,
            self._config.gait_frequency_max,
        )
        phase = info["gait_phase"] + 2.0 * jp.pi * frequency * self.dt
        phase = jp.mod(phase, 2.0 * jp.pi)
        return jp.where(standing_gate, 0.0, phase)

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

    def _reward_first_step_forward_progress(
        self, command: jax.Array, local_linvel: jax.Array
    ) -> jax.Array:
        commanded_forward = jp.maximum(command[0], 0.0)
        moving_forward = commanded_forward > 0.05
        target = jp.clip(commanded_forward, 0.25, 0.75)
        speed_fraction = local_linvel[0] / target
        return moving_forward * jp.clip(speed_fraction, 0.0, 1.0)

    def _cost_forward_speed_deficit(
        self, command: jax.Array, local_linvel: jax.Array
    ) -> jax.Array:
        commanded_forward = jp.maximum(command[0], 0.0)
        moving_forward = commanded_forward > 0.05
        deficit = jp.maximum(commanded_forward - local_linvel[0], 0.0)
        normalized = deficit / jp.maximum(commanded_forward, 1.0)
        return moving_forward * jp.square(normalized)

    def _cost_forward_speed_error(
        self, command: jax.Array, local_linvel: jax.Array
    ) -> jax.Array:
        commanded_forward = jp.maximum(command[0], 0.0)
        moving_forward = commanded_forward > 0.05
        normalized_error = (local_linvel[0] - commanded_forward) / jp.maximum(
            commanded_forward, 1.0
        )
        return moving_forward * jp.square(normalized_error)

    def _reward_tracking_turn_vel(
        self, command: jax.Array, local_angvel: jax.Array
    ) -> jax.Array:
        error = jp.square(command[1] - local_angvel[1])
        return jp.exp(-error / self._config.reward_config.turn_tracking_sigma)

    def _cost_base_tilt_ang_vel(self, local_angvel: jax.Array) -> jax.Array:
        return jp.square(local_angvel[0]) + jp.square(local_angvel[2])

    def _cost_moving_action_deviation(self, action: jax.Array) -> jax.Array:
        leg_action = action[:8]
        stand_leg_action = self._stand_pose_action[:8]
        return jp.mean(jp.square(leg_action - stand_leg_action))

    def _cost_no_foot_contact(self, data: mjx.Data) -> jax.Array:
        contact_sum = sum(self._foot_contact_scores(data))
        return jp.square(jp.maximum(1.0 - contact_sum, 0.0))

    def _moving_foot_support_gate(self, data: mjx.Data) -> jax.Array:
        contact_sum = sum(self._foot_contact_scores(data))
        return jp.clip(contact_sum / 0.75, 0.0, 1.0)

    def _cost_running_height_excess(self, torso_height: jax.Array) -> jax.Array:
        max_running_height = self._target_torso_height + 0.10
        return jp.square(jp.maximum(torso_height - max_running_height, 0.0))

    def _reward_running_height_gate(self, torso_height: jax.Array) -> jax.Array:
        max_running_height = self._target_torso_height + 0.10
        excess = jp.maximum(torso_height - max_running_height, 0.0)
        return jp.exp(-10.0 * jp.square(excess))

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

    def _fall_done(self, data: mjx.Data) -> jax.Array:
        if not self._config.terminate_on_fall:
            return jp.zeros(())
        orientation = self._reward_orientation(self.get_gravity(data))
        torso_height = data.site_xpos[self._imu_site_id, 2]
        fallen = (orientation < self._config.fall_orientation_threshold) | (
            torso_height < self._config.fall_torso_height
        )
        return fallen.astype(jp.float32)

    def _running_speed_gate(self, command: jax.Array) -> jax.Array:
        speed = jp.linalg.norm(command)
        width = jp.maximum(
            self._config.running_gate_full - self._config.running_gate_start, 1e-6
        )
        return jp.clip((speed - self._config.running_gate_start) / width, 0.0, 1.0)

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
        left_clearance = jp.min(
            jp.clip(self._geom_bottom(data, self._left_foot_geom_ids), 0.0, 0.5)
        )
        right_clearance = jp.min(
            jp.clip(self._geom_bottom(data, self._right_foot_geom_ids), 0.0, 0.5)
        )
        clearance = 0.5 * (left_clearance + right_clearance)
        return jp.clip(clearance / 0.25, 0.0, 1.0)

    def _reward_gait_prior_tracking(
        self, action: jax.Array, info: dict[str, Any]
    ) -> jax.Array:
        target = jp.clip(
            self._stand_pose_action + self._gait_prior_action(info),
            -1.0,
            1.0,
        )
        leg_error = jp.mean(jp.square(action[:8] - target[:8]))
        return jp.exp(-leg_error / 0.04)

    def _reward_leg_action_alternation(self, action: jax.Array) -> jax.Array:
        gait_action = action - self._stand_pose_action
        pair_scores = jp.array(
            [
                self._anti_phase_score(gait_action[2], gait_action[3]),
                self._anti_phase_score(gait_action[4], gait_action[5]),
                self._anti_phase_score(gait_action[6], gait_action[7]),
            ]
        )
        pair_magnitudes = jp.array(
            [
                0.5 * (jp.abs(gait_action[2]) + jp.abs(gait_action[3])),
                0.5 * (jp.abs(gait_action[4]) + jp.abs(gait_action[5])),
                0.5 * (jp.abs(gait_action[6]) + jp.abs(gait_action[7])),
            ]
        )
        amplitude_gate = jp.clip((jp.mean(pair_magnitudes) - 0.05) / 0.35, 0.0, 1.0)
        return amplitude_gate * jp.mean(pair_scores)

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

    def _reward_phase_contact(
        self, data: mjx.Data, info: dict[str, Any]
    ) -> jax.Array:
        error = self._cost_phase_contact_error(data, info)
        return jp.exp(-2.0 * error)

    def _cost_phase_contact_error(
        self, data: mjx.Data, info: dict[str, Any]
    ) -> jax.Array:
        contact = jp.array(self._foot_contact_scores(data))
        target = self._phase_contact_targets(info["gait_phase"])
        return jp.sum(jp.square(contact - target))

    def _reward_phase_foot_clearance(
        self, data: mjx.Data, info: dict[str, Any]
    ) -> jax.Array:
        left_contact, right_contact = self._foot_contact_scores(data)
        left_clearance, right_clearance = self._foot_clearance_scores(data)
        clearance = jp.array([left_clearance, right_clearance])
        contact = jp.array([left_contact, right_contact])
        target_clearance = self._phase_foot_clearance_targets(info["gait_phase"])
        contact_target = self._phase_contact_targets(info["gait_phase"])
        stance_contact = jp.sum(contact * contact_target) / jp.maximum(
            jp.sum(contact_target), 1e-6
        )
        clearance_score = jp.exp(
            -jp.sum(jp.square(clearance - target_clearance))
            / self._phase_clearance_error_denominator()
        )
        return stance_contact * clearance_score

    def _reward_feet_phase_height(
        self, data: mjx.Data, phase: jax.Array, command: jax.Array
    ) -> jax.Array:
        left_clearance, right_clearance = self._foot_clearance_scores(data)
        return self._reward_feet_phase_height_from_clearance(
            jp.array([left_clearance, right_clearance]), phase, command
        )

    def _reward_phase_swing_clearance(
        self, data: mjx.Data, phase: jax.Array
    ) -> jax.Array:
        left_clearance, right_clearance = self._foot_clearance_scores(data)
        return self._reward_phase_swing_clearance_from_clearance(
            jp.array([left_clearance, right_clearance]), phase
        )

    def _reward_phase_swing_release(
        self, data: mjx.Data, phase: jax.Array
    ) -> jax.Array:
        left_contact, right_contact = self._foot_contact_scores(data)
        return self._reward_phase_swing_release_from_contact(
            jp.array([left_contact, right_contact]), phase
        )

    def _reward_phase_stance_contact(
        self, data: mjx.Data, phase: jax.Array
    ) -> jax.Array:
        left_contact, right_contact = self._foot_contact_scores(data)
        return self._reward_phase_stance_contact_from_contact(
            jp.array([left_contact, right_contact]), phase
        )

    def _cost_phase_clearance_error(
        self, data: mjx.Data, phase: jax.Array
    ) -> jax.Array:
        left_clearance, right_clearance = self._foot_clearance_scores(data)
        return self._cost_phase_clearance_error_from_clearance(
            jp.array([left_clearance, right_clearance]), phase
        )

    def _cost_phase_clearance_max_error(
        self, data: mjx.Data, phase: jax.Array
    ) -> jax.Array:
        left_clearance, right_clearance = self._foot_clearance_scores(data)
        return self._cost_phase_clearance_max_error_from_clearance(
            jp.array([left_clearance, right_clearance]), phase
        )

    def _reward_feet_phase_height_from_clearance(
        self, clearance: jax.Array, phase: jax.Array, command: jax.Array
    ) -> jax.Array:
        target_clearance = self._phase_foot_clearance_targets(phase)
        error = jp.sum(jp.square(clearance - target_clearance))
        moving = jp.linalg.norm(command) > 0.05
        return moving * jp.exp(-error / self._phase_clearance_error_denominator())

    def _reward_phase_swing_clearance_from_clearance(
        self, clearance: jax.Array, phase: jax.Array
    ) -> jax.Array:
        target_clearance = self._phase_foot_clearance_targets(phase)
        swing_weight = self._phase_swing_weights(phase)
        normalized_error = (clearance - target_clearance) / jp.maximum(
            self._config.gait_swing_height, 1e-6
        )
        per_foot = jp.exp(-4.0 * jp.square(normalized_error))
        return jp.sum(swing_weight * per_foot) / jp.maximum(
            jp.sum(swing_weight), 1e-6
        )

    def _reward_phase_swing_release_from_contact(
        self, contact: jax.Array, phase: jax.Array
    ) -> jax.Array:
        swing_weight = self._phase_swing_weights(phase)
        return jp.sum(swing_weight * (1.0 - contact)) / jp.maximum(
            jp.sum(swing_weight), 1e-6
        )

    def _reward_phase_stance_contact_from_contact(
        self, contact: jax.Array, phase: jax.Array
    ) -> jax.Array:
        stance_weight = self._phase_stance_weights(phase)
        return jp.sum(stance_weight * contact) / jp.maximum(
            jp.sum(stance_weight), 1e-6
        )

    def _phase_clearance_error_denominator(self) -> jax.Array:
        swing_height = jp.maximum(self._config.gait_swing_height, 1e-6)
        return jp.square(swing_height) / 1.44

    def _cost_phase_clearance_error_from_clearance(
        self, clearance: jax.Array, phase: jax.Array
    ) -> jax.Array:
        normalized_error = self._phase_clearance_normalized_error(clearance, phase)
        return jp.mean(jp.square(normalized_error))

    def _cost_phase_clearance_max_error_from_clearance(
        self, clearance: jax.Array, phase: jax.Array
    ) -> jax.Array:
        normalized_error = self._phase_clearance_normalized_error(clearance, phase)
        return jp.max(jp.square(normalized_error))

    def _phase_clearance_normalized_error(
        self, clearance: jax.Array, phase: jax.Array
    ) -> jax.Array:
        target_clearance = self._phase_foot_clearance_targets(phase)
        swing_height = jp.maximum(self._config.gait_swing_height, 1e-6)
        return (clearance - target_clearance) / swing_height

    def _phase_foot_clearance_targets(self, phase: jax.Array) -> jax.Array:
        phase = self._wrap_gait_phase(phase)
        foot_phase = jp.array([phase, self._wrap_gait_phase(phase + jp.pi)])
        return gait.get_rz(foot_phase, swing_height=self._config.gait_swing_height)

    def _phase_contact_targets(self, phase: jax.Array) -> jax.Array:
        swing_fraction = jp.clip(
            self._phase_foot_clearance_targets(phase) / self._config.gait_swing_height,
            0.0,
            1.0,
        )
        return 1.0 - swing_fraction

    def _phase_swing_weights(self, phase: jax.Array) -> jax.Array:
        return jp.clip(
            self._phase_foot_clearance_targets(phase) / self._config.gait_swing_height,
            0.0,
            1.0,
        )

    def _phase_stance_weights(self, phase: jax.Array) -> jax.Array:
        return 1.0 - self._phase_swing_weights(phase)

    def _wrap_gait_phase(self, phase: jax.Array) -> jax.Array:
        return jp.fmod(phase + jp.pi, 2.0 * jp.pi) - jp.pi

    def _reward_single_support_balance(
        self, data: mjx.Data, phase: jax.Array
    ) -> jax.Array:
        left_offset, right_offset = self._mjx_foot_offsets_in_torso_frame(data)
        return self._reward_single_support_balance_from_offsets(
            left_offset, right_offset, self._phase_contact_targets(phase)
        )

    def _reward_single_support_balance_from_offsets(
        self,
        left_offset: jax.Array,
        right_offset: jax.Array,
        contact_target: jax.Array,
    ) -> jax.Array:
        target_sum = jp.maximum(jp.sum(contact_target), 1e-6)
        stance_weight = contact_target / target_sum
        left_xz = jp.array([left_offset[0], left_offset[2]])
        right_xz = jp.array([right_offset[0], right_offset[2]])
        stance_xz = stance_weight[0] * left_xz + stance_weight[1] * right_xz
        error = stance_xz - self._standing_support_offset_xz
        single_support_gate = jp.abs(stance_weight[0] - stance_weight[1])
        return single_support_gate * jp.exp(-8.0 * jp.sum(jp.square(error)))

    def _reward_foot_contact_balance(self, data: mjx.Data) -> jax.Array:
        left_contact, right_contact = self._foot_contact_scores(data)
        one_foot_stance = (
            left_contact + right_contact - 2.0 * left_contact * right_contact
        )
        any_contact = jp.clip(left_contact + right_contact, 0.0, 1.0)
        return 0.5 * any_contact + 0.5 * one_foot_stance

    def _reward_feet_air_time(
        self,
        feet_air_time: jax.Array,
        first_contact: jax.Array,
        command: jax.Array,
    ) -> jax.Array:
        moving = self._running_speed_gate(command) > 0.0
        air_time_reward = jp.sum((feet_air_time - 0.12) * first_contact)
        return moving * jp.clip(air_time_reward, 0.0, 0.5)

    def _cost_double_foot_contact(self, data: mjx.Data) -> jax.Array:
        left_contact, right_contact = self._foot_contact_scores(data)
        return left_contact * right_contact

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

    def _cost_contact_duty_error(
        self, data: mjx.Data, info: dict[str, Any]
    ) -> jax.Array:
        duty = self._updated_contact_duty(data, info)
        total = jp.sum(duty)
        asymmetry = jp.square(duty[0] - duty[1])
        support_error = jp.square(total - 1.0)
        return asymmetry + 0.25 * support_error

    def _anti_phase_score(
        self, left: jax.Array, right: jax.Array, epsilon: float = 0.02
    ) -> jax.Array:
        phase = -(left * right) / (jp.abs(left) * jp.abs(right) + epsilon)
        return jp.clip(phase, 0.0, 1.0)

    def _stride_gate(self, left_step: jax.Array, right_step: jax.Array) -> jax.Array:
        stride = 0.5 * (jp.abs(left_step) + jp.abs(right_step))
        return jp.clip((stride - 0.15) / 0.65, 0.0, 1.0)

    def _foot_contact_scores(self, data: mjx.Data) -> tuple[jax.Array, jax.Array]:
        left_clearance, right_clearance = self._foot_clearance_scores(data)
        return (
            self._height_contact_score(left_clearance),
            self._height_contact_score(right_clearance),
        )

    def _height_contact_score(self, clearance: jax.Array) -> jax.Array:
        contact_height = jp.maximum(self._config.foot_contact_height, 1e-6)
        return jp.clip(1.0 - clearance / contact_height, 0.0, 1.0)

    def _foot_ground_force(self, data: mjx.Data, foot_geom_ids: jax.Array) -> jax.Array:
        contact_geom = self._contact_geom(data)
        geom_a = contact_geom[:, 0]
        geom_b = contact_geom[:, 1]
        has_floor = (geom_a == self._floor_geom_id) | (geom_b == self._floor_geom_id)
        has_foot = jp.any(
            (geom_a[:, None] == foot_geom_ids[None, :])
            | (geom_b[:, None] == foot_geom_ids[None, :]),
            axis=1,
        )
        valid = self._contact_dim(data) > 0
        contact_force = self._contact_force(data)
        return jp.sum(jp.where(valid & has_floor & has_foot, contact_force, 0.0))

    def _contact_geom(self, data: mjx.Data) -> jax.Array:
        impl = data._impl
        if hasattr(impl, "contact__geom"):
            return impl.contact__geom
        return impl.contact.geom

    def _contact_dim(self, data: mjx.Data) -> jax.Array:
        impl = data._impl
        if hasattr(impl, "contact__dim"):
            return impl.contact__dim
        return impl.contact.dim

    def _contact_efc_address(self, data: mjx.Data) -> jax.Array:
        impl = data._impl
        if hasattr(impl, "contact__efc_address"):
            return impl.contact__efc_address
        return impl.contact.efc_address[:, None]

    def _efc_force(self, data: mjx.Data) -> jax.Array:
        impl = data._impl
        if hasattr(impl, "efc__force"):
            return impl.efc__force
        if hasattr(impl, "efc_force"):
            return impl.efc_force
        return data.efc_force

    def _contact_force(self, data: mjx.Data) -> jax.Array:
        address = self._contact_efc_address(data)
        valid_address = address >= 0
        safe_address = jp.maximum(address, 0)
        force = jp.take(self._efc_force(data), safe_address, mode="clip")
        force = jp.where(valid_address, jp.maximum(force, 0.0), 0.0)
        return jp.sum(force, axis=1)

    def _foot_clearance_scores(self, data: mjx.Data) -> tuple[jax.Array, jax.Array]:
        left_clearance = jp.min(
            jp.clip(self._geom_bottom(data, self._left_foot_geom_ids), 0.0, 0.3)
        )
        right_clearance = jp.min(
            jp.clip(self._geom_bottom(data, self._right_foot_geom_ids), 0.0, 0.3)
        )
        return left_clearance, right_clearance

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

    def _cost_hip_adduction_neutral(self, data: mjx.Data) -> jax.Array:
        hip_adduction = data.qpos[self._leg_qpos_ids[:2]]
        return jp.sum(jp.square(hip_adduction))

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
