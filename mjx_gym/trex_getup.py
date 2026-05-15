"""Getup-and-stand task for the simplified T-Rex MJX model."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env
from mjx_gym import trex_constants as consts


def _yaw_quat(yaw: jax.Array) -> jax.Array:
    return jp.array([jp.cos(yaw / 2.0), 0.0, 0.0, jp.sin(yaw / 2.0)])


def _roll_quat(roll: jax.Array) -> jax.Array:
    return jp.array([jp.cos(roll / 2.0), jp.sin(roll / 2.0), 0.0, 0.0])


def _quat_mul(left: jax.Array, right: jax.Array) -> jax.Array:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return jp.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ]
    )


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.004,
        Kp=200.0,
        passive_stiffness=1000.0,
        actuated_joint_passive_stiffness_scale=0.0,
        tail_joint_passive_stiffness_scale=1.0,
        leg_actuator_kp_scale=[1.0] * len(consts.LEG_JOINTS),
        passive_damping=80.0,
        armature=0.2,
        episode_length=750,
        action_repeat=1,
        action_scale=1.0,
        reset_xy_range=0.25,
        reset_yaw_range=3.141592653589793,
        reset_joint_noise=0.0,
        reset_qvel_noise=0.05,
        reset_height_noise=0.02,
        side_upright_roll_min=0.0,
        side_upright_roll_max=0.0,
        side_standing_joint_blend_min=0.0,
        side_standing_joint_blend_max=0.0,
        torso_height=2.5,
        clearance_height=0.12,
        reward_clip_min=-100.0,
        reward_clip_max=10000.0,
        reward_config=config_dict.create(
            scales=config_dict.create(
                orientation=1.0,
                torso_height=1.0,
                non_foot_clearance=1.0,
                foot_support=1.0,
                foot_balance=1.5,
                foot_placement=2.0,
                standing_pose=1.0,
                stand_still=0.25,
                action_rate=-1e-5,
                torques=-1e-9,
                dof_vel=-1e-6,
                root_vel=-1e-4,
                base_lin_vel=-5e-1,
                base_ang_vel=-5e-1,
            ),
        ),
        impl="jax",
        upright_gravity=consts.UPRIGHT_GRAVITY.tolist(),
        naconmax=16384,
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
            consts.trex_getup_xml(
                position_kp_per_row_sum=self._config.Kp,
                passive_stiffness_per_row_sum=self._config.passive_stiffness,
                actuated_joint_stiffness_scale=(
                    self._config.actuated_joint_passive_stiffness_scale
                ),
                tail_joint_stiffness_scale=(
                    self._config.tail_joint_passive_stiffness_scale
                ),
                actuator_kp_scale_overrides={
                    joint_name: scale
                    for joint_name, scale in zip(
                        consts.LEG_JOINTS, self._config.leg_actuator_kp_scale
                    )
                },
                passive_damping_per_row_sum=self._config.passive_damping,
                armature_per_row_sum=self._config.armature,
            )
        )
        self._mj_model.opt.timestep = self._config.sim_dt
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        self._xml_path = str(consts.URDF_PATH)
        self._imu_site_id = self._mj_model.site(consts.IMU_SITE).id
        self._action_actuator_ids = jp.array(
            [self._mj_model.actuator(name).id for name in consts.ACTION_ACTUATORS],
            dtype=jp.int32,
        )
        action_ctrlrange = self._mj_model.actuator_ctrlrange[
            np.array(self._action_actuator_ids)
        ]
        action_ctrl_low = action_ctrlrange[:, 0]
        action_ctrl_high = action_ctrlrange[:, 1]
        self._action_ctrl_neutral = jp.zeros(len(consts.ACTION_ACTUATORS))
        self._action_ctrl_negative_scale = jp.array(
            self._action_ctrl_neutral - action_ctrl_low
        )
        self._action_ctrl_positive_scale = jp.array(
            action_ctrl_high - self._action_ctrl_neutral
        )
        self._default_ctrl = jp.zeros(self._mj_model.nu)
        self._side_qpos = jp.array(consts.side_lying_qpos(self._mj_model))
        self._standing_qpos = jp.array(consts.standing_qpos(self._mj_model))
        self._target_torso_height = float(self._config.torso_height)
        self._leg_qpos_ids = jp.array(
            [
                self._mj_model.jnt_qposadr[self._mj_model.joint(joint_name).id]
                for joint_name in consts.LEG_JOINTS
            ],
            dtype=jp.int32,
        )
        self._standing_leg_qpos = self._standing_qpos[self._leg_qpos_ids]
        self._non_foot_geom_ids = jp.array(
            [
                geom_id
                for geom_id in range(self._mj_model.ngeom)
                if self._is_non_foot_contact_geom(geom_id)
            ],
            dtype=jp.int32,
        )
        self._left_foot_geom_ids = self._foot_geom_ids("left")
        self._right_foot_geom_ids = self._foot_geom_ids("right")
        self._standing_torso_to_support_xy = jp.array(
            self._torso_to_support_xy(np.array(self._standing_qpos))
        )
        standing_left, standing_right = self._foot_offsets_in_torso_frame(
            np.array(self._standing_qpos)
        )
        self._standing_left_foot_offset = jp.array(standing_left)
        self._standing_right_foot_offset = jp.array(standing_right)

    def reset(self, rng: jax.Array) -> mjx_env.State:
        yaw_rng, xy_rng, joint_rng, qvel_rng, height_rng, roll_rng, blend_rng = (
            jax.random.split(rng, 7)
        )
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
        roll = jax.random.uniform(
            roll_rng,
            (),
            minval=self._config.side_upright_roll_min,
            maxval=self._config.side_upright_roll_max,
        )
        standing_blend = jax.random.uniform(
            blend_rng,
            (),
            minval=self._config.side_standing_joint_blend_min,
            maxval=self._config.side_standing_joint_blend_max,
        )
        qpos = self._side_qpos.at[0:2].set(xy)
        qpos = qpos.at[2].set(
            (1.0 - standing_blend) * self._side_qpos[2]
            + standing_blend * self._standing_qpos[2]
            + height_noise
        )
        qpos = qpos.at[3:7].set(_quat_mul(_yaw_quat(yaw), _roll_quat(roll)))
        qpos = qpos.at[7:].set(
            standing_blend * self._standing_qpos[7:] + joint_noise
        )
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
        obs = self._get_obs(data, state.info)
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
        orientation = self._reward_orientation(gravity)
        clearance = self._reward_non_foot_clearance(data)
        stillness_gate = orientation * self._reward_height(torso_height)
        return {
            "orientation": orientation,
            "torso_height": orientation * self._reward_height(torso_height),
            "non_foot_clearance": orientation * clearance,
            "foot_support": orientation * self._reward_foot_support(data),
            "foot_balance": orientation * self._reward_foot_balance(data),
            "foot_placement": orientation * self._reward_foot_placement(data),
            "standing_pose": orientation * self._reward_standing_pose(data.qpos),
            "stand_still": self._reward_stand_still(action, gravity, torso_height),
            "action_rate": self._cost_action_rate(action, info),
            "torques": self._cost_torques(data.actuator_force),
            "dof_vel": self._cost_dof_vel(data.qvel[6:]),
            "root_vel": self._cost_root_vel(data.qvel[:6]),
            "base_lin_vel": stillness_gate * self._cost_base_lin_vel(data.qvel[:3]),
            "base_ang_vel": stillness_gate * self._cost_base_ang_vel(data.qvel[3:6]),
        }

    def _is_non_foot_contact_geom(self, geom_id: int) -> bool:
        if self._mj_model.geom_group[geom_id] != 2:
            return False
        name = mujoco.mj_id2name(self._mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if name is None:
            return False
        return "toe" not in name and "tarsometatarsus" not in name

    def _is_foot_contact_geom(self, geom_id: int, side: str) -> bool:
        if self._mj_model.geom_group[geom_id] != 2:
            return False
        name = mujoco.mj_id2name(self._mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if name is None or side not in name:
            return False
        return "toe" in name or "tarsometatarsus" in name

    def _foot_geom_ids(self, side: str) -> jax.Array:
        return jp.array(
            [
                geom_id
                for geom_id in range(self._mj_model.ngeom)
                if self._is_foot_contact_geom(geom_id, side)
            ],
            dtype=jp.int32,
        )

    def _reward_orientation(self, gravity: jax.Array) -> jax.Array:
        target = jp.array(self._config.upright_gravity)
        return jp.exp(-2.0 * jp.sum(jp.square(target - gravity)))

    def _reward_height(self, torso_height: jax.Array) -> jax.Array:
        height_error = jp.maximum(self._target_torso_height - torso_height, 0.0)
        return jp.exp(-2.0 * jp.square(height_error))

    def _reward_non_foot_clearance(self, data: mjx.Data) -> jax.Array:
        geom_bottom = self._geom_bottom(data, self._non_foot_geom_ids)
        clearance = jp.minimum(geom_bottom / self._config.clearance_height, 1.0)
        return jp.min(jp.clip(clearance, 0.0, 1.0))

    def _reward_foot_support(self, data: mjx.Data) -> jax.Array:
        left_height = jp.min(jp.abs(self._geom_bottom(data, self._left_foot_geom_ids)))
        right_height = jp.min(jp.abs(self._geom_bottom(data, self._right_foot_geom_ids)))
        left = jp.exp(-200.0 * jp.square(left_height))
        right = jp.exp(-200.0 * jp.square(right_height))
        return 0.5 * (left + right)

    def _reward_foot_balance(self, data: mjx.Data) -> jax.Array:
        left_center = jp.mean(data.geom_xpos[self._left_foot_geom_ids, :2], axis=0)
        right_center = jp.mean(data.geom_xpos[self._right_foot_geom_ids, :2], axis=0)
        support_center = 0.5 * (left_center + right_center)
        torso_xy = data.site_xpos[self._imu_site_id, :2]
        error = torso_xy - support_center - self._standing_torso_to_support_xy
        return jp.exp(-2.0 * jp.sum(jp.square(error)))

    def _reward_foot_placement(self, data: mjx.Data) -> jax.Array:
        left_offset, right_offset = self._mjx_foot_offsets_in_torso_frame(data)
        left_error = left_offset - self._standing_left_foot_offset
        right_error = right_offset - self._standing_right_foot_offset
        error = 0.5 * (
            jp.sum(jp.square(left_error)) + jp.sum(jp.square(right_error))
        )
        return jp.exp(-2.0 * error)

    def _mjx_foot_offsets_in_torso_frame(
        self, data: mjx.Data
    ) -> tuple[jax.Array, jax.Array]:
        torso_pos = data.site_xpos[self._imu_site_id]
        torso_xmat = data.site_xmat[self._imu_site_id].reshape((3, 3))
        left_center = jp.mean(data.geom_xpos[self._left_foot_geom_ids], axis=0)
        right_center = jp.mean(data.geom_xpos[self._right_foot_geom_ids], axis=0)
        return (
            torso_xmat.T @ (left_center - torso_pos),
            torso_xmat.T @ (right_center - torso_pos),
        )

    def _torso_to_support_xy(self, qpos: np.ndarray) -> np.ndarray:
        data = mujoco.MjData(self._mj_model)
        data.qpos[:] = qpos
        mujoco.mj_forward(self._mj_model, data)
        left_center = np.mean(
            data.geom_xpos[np.array(self._left_foot_geom_ids), :2], axis=0
        )
        right_center = np.mean(
            data.geom_xpos[np.array(self._right_foot_geom_ids), :2], axis=0
        )
        support_center = 0.5 * (left_center + right_center)
        return data.site_xpos[self._imu_site_id, :2] - support_center

    def _foot_offsets_in_torso_frame(
        self, qpos: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        data = mujoco.MjData(self._mj_model)
        data.qpos[:] = qpos
        mujoco.mj_forward(self._mj_model, data)
        torso_pos = data.site_xpos[self._imu_site_id]
        torso_xmat = data.site_xmat[self._imu_site_id].reshape((3, 3))
        left_center = np.mean(
            data.geom_xpos[np.array(self._left_foot_geom_ids)], axis=0
        )
        right_center = np.mean(
            data.geom_xpos[np.array(self._right_foot_geom_ids)], axis=0
        )
        return (
            torso_xmat.T @ (left_center - torso_pos),
            torso_xmat.T @ (right_center - torso_pos),
        )

    def _reward_standing_pose(self, qpos: jax.Array) -> jax.Array:
        error = qpos[self._leg_qpos_ids] - self._standing_leg_qpos
        return jp.exp(-0.5 * jp.sum(jp.square(error)))

    def _geom_bottom(self, data: mjx.Data, geom_ids: jax.Array) -> jax.Array:
        geom_xpos = data.geom_xpos[geom_ids]
        geom_xmat = data.geom_xmat[geom_ids].reshape((-1, 3, 3))
        geom_size = jp.array(self._mj_model.geom_size)[geom_ids]
        radius = geom_size[:, 0]
        half_length = geom_size[:, 1]
        local_z_vertical = jp.abs(geom_xmat[:, 2, 2])
        return geom_xpos[:, 2] - radius - half_length * local_z_vertical

    def _reward_stand_still(
        self, action: jax.Array, gravity: jax.Array, torso_height: jax.Array
    ) -> jax.Array:
        upright = self._reward_orientation(gravity) > 0.95
        high = torso_height > self._target_torso_height * 0.95
        return (upright * high) * jp.exp(-jp.sum(jp.square(action)))

    def _cost_action_rate(self, action: jax.Array, info: dict[str, Any]) -> jax.Array:
        first = jp.sum(jp.square(action - info["last_act"]))
        second = jp.sum(
            jp.square(action - 2 * info["last_act"] + info["last_last_act"])
        )
        return first + second

    def _cost_torques(self, torques: jax.Array) -> jax.Array:
        return jp.sqrt(jp.sum(jp.square(torques))) + jp.sum(jp.abs(torques))

    def _cost_dof_vel(self, qvel: jax.Array) -> jax.Array:
        max_velocity = 2.0 * jp.pi
        excess_velocity = jp.maximum(jp.abs(qvel) - max_velocity, 0.0)
        return jp.sum(jp.square(excess_velocity))

    def _cost_root_vel(self, qvel: jax.Array) -> jax.Array:
        return jp.sum(jp.square(qvel[:3])) + 0.25 * jp.sum(jp.square(qvel[3:6]))

    def _cost_base_lin_vel(self, qvel: jax.Array) -> jax.Array:
        return jp.sum(jp.square(qvel))

    def _cost_base_ang_vel(self, qvel: jax.Array) -> jax.Array:
        return jp.sum(jp.square(qvel))

    def get_gyro(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(self.mj_model, data, consts.GYRO_SENSOR)

    def get_gravity(self, data: mjx.Data) -> jax.Array:
        return data.site_xmat[self._imu_site_id].T @ jp.array([0.0, 0.0, -1.0])

    def render(self, trajectory, height=240, width=320, camera=None, **kwargs):
        if camera is None:
            camera = "track"
        return super().render(
            trajectory, height=height, width=width, camera=camera, **kwargs
        )

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
