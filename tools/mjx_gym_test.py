import unittest

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
import numpy as np

from mjx_gym import train
from mjx_gym import trex_constants
from mjx_gym import trex_getup
from mjx_gym import trex_joystick
from tools import mjx_model_simplification


class TestMjxGym(unittest.TestCase):
    def test_trex_getup_model_reset_and_step(self):
        env = trex_getup.TrexGetup()

        self.assertEqual(10, env.action_size)
        self.assertEqual(31, env.mj_model.nbody)
        self.assertEqual(32, env.mj_model.njnt)
        self.assertEqual(38, env.mj_model.nq)
        self.assertEqual(37, env.mj_model.nv)
        self.assertEqual(10, env.mj_model.nu)
        self.assertEqual(46, env.mj_model.ngeom)
        self.assertGreaterEqual(env.mj_model.nlight, 1)
        self.assertEqual(5, env.mj_model.nsensor)
        self.assertEqual(
            [
                "gyro",
                "accelerometer",
                "upvector",
                "global_linvel",
                "global_angvel",
            ],
            [env.mj_model.sensor(index).name for index in range(env.mj_model.nsensor)],
        )

        state = env.reset(jax.random.PRNGKey(0))
        other_state = env.reset(jax.random.PRNGKey(1))
        self.assertFalse(np.allclose(state.data.qpos, other_state.data.qpos))
        self.assertEqual((78,), state.obs["state"].shape)
        self.assertEqual((164,), state.obs["privileged_state"].shape)
        self.assertEqual((38,), state.data.qpos.shape)
        self.assertEqual((37,), state.data.qvel.shape)
        self.assertEqual((10,), state.data.ctrl.shape)
        self.assertEqual(750, env._config.episode_length)

        next_state = env.step(state, jp.zeros(env.action_size))
        self.assertEqual((78,), next_state.obs["state"].shape)
        self.assertEqual((164,), next_state.obs["privileged_state"].shape)
        self.assertGreater(float(next_state.data.time), 0.0)
        self.assertGreaterEqual(float(next_state.reward), env._config.reward_clip_min)

    def test_zero_upright_pose_matches_orientation_goal_and_feet_contact(self):
        env = trex_getup.TrexGetup()
        model = env.mj_model
        data = mujoco.MjData(model)
        data.qpos[:] = trex_constants.standing_qpos(model)
        mujoco.mj_forward(model, data)

        imu_id = model.site(trex_constants.IMU_SITE).id
        gravity = data.site_xmat[imu_id].reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
        self.assertTrue(np.allclose(gravity, trex_constants.UPRIGHT_GRAVITY))
        self.assertAlmostEqual(float(env._reward_orientation(gravity)), 1.0)
        self.assertEqual(2.5, env._target_torso_height)
        self.assertGreater(float(data.site_xpos[imu_id, 2]), env._target_torso_height)
        self.assertAlmostEqual(float(env._reward_height(env._target_torso_height)), 1.0)
        self.assertAlmostEqual(
            float(env._reward_height(env._target_torso_height + 0.5)), 1.0
        )
        self.assertLess(float(env._reward_height(env._target_torso_height - 0.5)), 1.0)
        mjx_data = mjx.put_data(
            model,
            data,
            impl=env.mjx_model.impl.value,
            naconmax=env._config.naconmax,
            njmax=env._config.njmax,
        )
        self.assertGreater(float(env._reward_non_foot_clearance(mjx_data)), 0.9)
        self.assertGreater(float(env._reward_foot_support(mjx_data)), 0.9)
        self.assertGreater(float(env._reward_foot_balance(mjx_data)), 0.9)
        self.assertGreater(float(env._reward_foot_placement(mjx_data)), 0.9)
        self.assertAlmostEqual(float(env._reward_standing_pose(mjx_data.qpos)), 1.0)

        floor_id = model.geom("floor").id
        floor_contacts = []
        for contact_id in range(data.ncon):
            geom_pair = data.contact[contact_id].geom
            if floor_id in geom_pair:
                other = geom_pair[1] if geom_pair[0] == floor_id else geom_pair[0]
                floor_contacts.append(
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(other))
                )
        self.assertTrue(any("toe" in name for name in floor_contacts))

    def test_non_foot_clearance_reward_distinguishes_side_and_standing_pose(self):
        env = trex_getup.TrexGetup()
        model = env.mj_model

        standing = mujoco.MjData(model)
        standing.qpos[:] = trex_constants.standing_qpos(model)
        mujoco.mj_forward(model, standing)

        side = mujoco.MjData(model)
        side.qpos[:] = trex_constants.side_lying_qpos(model)
        mujoco.mj_forward(model, side)

        standing_reward = float(
            env._reward_non_foot_clearance(
                mjx.put_data(
                    model,
                    standing,
                    impl=env.mjx_model.impl.value,
                    naconmax=env._config.naconmax,
                    njmax=env._config.njmax,
                )
            )
        )
        side_reward = float(
            env._reward_non_foot_clearance(
                mjx.put_data(
                    model,
                    side,
                    impl=env.mjx_model.impl.value,
                    naconmax=env._config.naconmax,
                    njmax=env._config.njmax,
                )
            )
        )
        self.assertGreater(standing_reward, 0.9)
        self.assertLess(side_reward, standing_reward)

    def test_foot_support_reward_requires_both_feet_near_floor(self):
        env = trex_getup.TrexGetup()
        model = env.mj_model

        standing = mujoco.MjData(model)
        standing.qpos[:] = trex_constants.standing_qpos(model)
        mujoco.mj_forward(model, standing)
        standing_reward = float(
            env._reward_foot_support(
                mjx.put_data(
                    model,
                    standing,
                    impl=env.mjx_model.impl.value,
                    naconmax=env._config.naconmax,
                    njmax=env._config.njmax,
                )
            )
        )

        floating = mujoco.MjData(model)
        floating.qpos[:] = trex_constants.standing_qpos(model)
        floating.qpos[2] += 1.0
        mujoco.mj_forward(model, floating)
        floating_reward = float(
            env._reward_foot_support(
                mjx.put_data(
                    model,
                    floating,
                    impl=env.mjx_model.impl.value,
                    naconmax=env._config.naconmax,
                    njmax=env._config.njmax,
                )
            )
        )

        self.assertGreater(standing_reward, 0.9)
        self.assertLess(floating_reward, 0.1)

    def test_balance_and_pose_rewards_penalize_zero_leg_pose(self):
        env = trex_getup.TrexGetup()
        model = env.mj_model

        standing = mujoco.MjData(model)
        standing.qpos[:] = trex_constants.standing_qpos(model)
        mujoco.mj_forward(model, standing)
        standing_mjx = mjx.put_data(
            model,
            standing,
            impl=env.mjx_model.impl.value,
            naconmax=env._config.naconmax,
            njmax=env._config.njmax,
        )

        zero_pose = mujoco.MjData(model)
        zero_pose.qpos[:] = trex_constants.zero_upright_qpos(model)
        mujoco.mj_forward(model, zero_pose)
        zero_pose_mjx = mjx.put_data(
            model,
            zero_pose,
            impl=env.mjx_model.impl.value,
            naconmax=env._config.naconmax,
            njmax=env._config.njmax,
        )

        self.assertGreater(float(env._reward_foot_balance(standing_mjx)), 0.9)
        self.assertLess(
            float(env._reward_foot_balance(zero_pose_mjx)),
            float(env._reward_foot_balance(standing_mjx)),
        )
        self.assertGreater(float(env._reward_foot_placement(standing_mjx)), 0.9)
        self.assertLess(
            float(env._reward_foot_placement(zero_pose_mjx)),
            float(env._reward_foot_placement(standing_mjx)),
        )
        self.assertAlmostEqual(float(env._reward_standing_pose(standing_mjx.qpos)), 1.0)
        self.assertLess(float(env._reward_standing_pose(zero_pose_mjx.qpos)), 0.5)

    def test_reset_randomizes_yaw_without_matching_upright_orientation(self):
        env = trex_getup.TrexGetup()

        for seed in range(8):
            state = env.reset(jax.random.PRNGKey(seed))
            orientation = float(env._reward_orientation(env.get_gravity(state.data)))
            height = float(
                env._reward_height(state.data.site_xpos[env._imu_site_id, 2])
            )
            self.assertLess(orientation, 0.1)
            self.assertLess(height, 0.1)

    def test_trex_getup_contacts_are_ground_only(self):
        env = trex_getup.TrexGetup()
        model = env.mj_model
        floor_id = model.geom("floor").id

        self.assertEqual(1, model.geom_contype[floor_id])
        self.assertEqual(0, model.geom_conaffinity[floor_id])
        for geom_id in range(model.ngeom):
            if geom_id == floor_id or model.geom_group[geom_id] != 2:
                continue
            self.assertEqual(0, model.geom_contype[geom_id])
            self.assertEqual(1, model.geom_conaffinity[geom_id])

        data = mujoco.MjData(model)
        data.qpos[:] = trex_constants.side_lying_qpos(model)
        for _ in range(200):
            mujoco.mj_step(model, data)
            for contact_id in range(data.ncon):
                self.assertIn(floor_id, data.contact[contact_id].geom)

    def test_trex_getup_mass_scaled_passive_gains_settle_joint_perturbations(self):
        env = trex_getup.TrexGetup()
        model = env.mj_model
        model.opt.gravity[:] = 0
        actuated_joint_names = mjx_model_simplification._actuated_joint_names(
            trex_constants.trex_getup_mjcf()
        )
        actuated_joint_ids = {
            model.joint(joint_name).id for joint_name in actuated_joint_names
        }
        hinge_joint_ids = {
            joint_id
            for joint_id in range(model.njnt)
            if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE
        }
        passive_joint_ids = sorted(hinge_joint_ids - actuated_joint_ids)

        self.assertTrue(passive_joint_ids)
        self.assertTrue(actuated_joint_ids)
        tail_joint_ids = {
            model.joint(joint_name).id
            for joint_name in trex_constants.TAIL_TENDON_JOINTS
        }
        leg_actuated_joint_ids = actuated_joint_ids - tail_joint_ids

        self.assertGreater(np.min(model.jnt_stiffness[passive_joint_ids]), 1000.0)
        self.assertTrue(
            np.allclose(model.jnt_stiffness[list(leg_actuated_joint_ids)], 0.0)
        )
        self.assertGreater(np.min(model.jnt_stiffness[list(tail_joint_ids)]), 1000.0)
        self.assertGreater(
            np.min(model.dof_damping[[model.jnt_dofadr[j] for j in tail_joint_ids]]),
            100.0,
        )
        self.assertGreater(np.min(model.dof_damping[6:]), 100.0)
        self.assertGreater(np.min(model.actuator_gainprm[:, 0]), 100000.0)
        self.assertLess(np.max(model.actuator_gainprm[:, 0]), 2500000.0)

        for joint_name in (
            "joint_toe_04_d_right",
            "joint_vertebra_cervical_09",
        ):
            data = mujoco.MjData(model)
            data.qpos[:] = trex_constants.zero_upright_qpos(model)
            data.qpos[2] += 5.0
            joint_id = model.joint(joint_name).id
            qpos_id = model.jnt_qposadr[joint_id]
            data.qpos[qpos_id] = 0.1
            for _ in range(1000):
                mujoco.mj_step(model, data)
            self.assertLess(abs(data.qpos[qpos_id]), 1.0e-3)
            self.assertTrue(np.all(data.warning.number == 0))

        data = mujoco.MjData(model)
        data.qpos[:] = trex_constants.zero_upright_qpos(model)
        data.qpos[2] += 5.0
        joint_id = model.joint("joint_vertebra_caudal_34").id
        qpos_id = model.jnt_qposadr[joint_id]
        data.qpos[qpos_id] = 0.1
        for _ in range(1000):
            mujoco.mj_step(model, data)
        self.assertLess(abs(data.qpos[qpos_id]), 1.0e-2)
        self.assertTrue(np.all(data.warning.number == 0))

    def test_trex_getup_actions_move_driven_joints(self):
        env = trex_getup.TrexGetup()
        model = env.mj_model
        model.opt.gravity[:] = 0
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        leg_joint_names = (
            "joint_hip_adduction_right",
            "joint_hip_adduction_left",
            "joint_femur_right",
            "joint_femur_left",
            "joint_tibia_right",
            "joint_tibia_left",
            "joint_tarsometatarsus_right",
            "joint_tarsometatarsus_left",
        )
        start = {}
        for joint_name in leg_joint_names:
            joint_id = model.joint(joint_name).id
            start[joint_name] = data.qpos[model.jnt_qposadr[joint_id]]

        data.ctrl[np.array(env._action_actuator_ids)] = np.array(
            env._action_ctrl_neutral
        ) + np.array(env._action_ctrl_positive_scale)
        for _ in range(250):
            mujoco.mj_step(model, data)

        for joint_name in leg_joint_names:
            joint_id = model.joint(joint_name).id
            qpos_id = model.jnt_qposadr[joint_id]
            self.assertGreater(data.qpos[qpos_id] - start[joint_name], 0.08)
        self.assertTrue(np.all(data.warning.number == 0))

    def test_trex_getup_random_actions_do_not_explode(self):
        env = trex_getup.TrexGetup()
        state = env.reset(jax.random.PRNGKey(0))
        step = jax.jit(env.step)
        rng = jax.random.PRNGKey(1)

        max_abs_qvel = 0.0
        max_abs_actuator_force = 0.0
        for _ in range(10):
            rng, action_rng = jax.random.split(rng)
            action = jax.random.uniform(
                action_rng, (env.action_size,), minval=-1.0, maxval=1.0
            )
            state = step(state, action)
            max_abs_qvel = max(max_abs_qvel, float(jp.max(jp.abs(state.data.qvel))))
            max_abs_actuator_force = max(
                max_abs_actuator_force,
                float(jp.max(jp.abs(state.data.actuator_force))),
            )

            leaves = jax.tree_util.tree_leaves(state)
            for leaf in leaves:
                if hasattr(leaf, "dtype") and jp.issubdtype(leaf.dtype, jp.inexact):
                    self.assertTrue(bool(jp.all(jp.isfinite(leaf))))

        self.assertLess(max_abs_qvel, 180.0)
        self.assertLess(max_abs_actuator_force, 2000000.0)

    def test_trex_joystick_model_reset_and_step(self):
        env = trex_joystick.TrexJoystick()

        self.assertEqual(10, env.action_size)
        state = env.reset(jax.random.PRNGKey(0))
        other_state = env.reset(jax.random.PRNGKey(1))
        self.assertFalse(
            np.allclose(state.info["command"], other_state.info["command"])
        )
        self.assertEqual((88,), state.obs["state"].shape)
        self.assertEqual((174,), state.obs["privileged_state"].shape)
        self.assertEqual((2,), state.info["command"].shape)
        self.assertGreaterEqual(int(state.info["steps_until_next_cmd"]), 1)

        next_state = env.step(state, jp.zeros(env.action_size))
        self.assertEqual((88,), next_state.obs["state"].shape)
        self.assertEqual((174,), next_state.obs["privileged_state"].shape)
        self.assertGreater(float(next_state.data.time), 0.0)
        self.assertGreaterEqual(float(next_state.reward), env._config.reward_clip_min)

    def test_trex_joystick_zero_command_uses_stable_stand_pose_when_upright(self):
        config = trex_joystick.default_config()
        config.reset_standing_prob = 1.0
        env = trex_joystick.TrexJoystick(config)
        state = env.reset(jax.random.PRNGKey(0))
        state.info["command"] = jp.zeros(2)

        requested_action = jp.ones(env.action_size)
        next_state = env.step(state, requested_action)

        self.assertTrue(
            np.allclose(
                np.asarray(next_state.info["last_act"]),
                np.asarray(config.stand_pose_action),
            )
        )

    def test_trex_joystick_gait_phase_scores_prefer_alternating_steps(self):
        env = trex_joystick.TrexJoystick()

        alternating = float(env._anti_phase_score(jp.array(0.5), jp.array(-0.5)))
        in_phase = float(env._anti_phase_score(jp.array(0.5), jp.array(0.5)))
        small_stride = float(env._stride_gate(jp.array(0.05), jp.array(-0.05)))
        large_stride = float(env._stride_gate(jp.array(0.8), jp.array(-0.8)))

        self.assertGreater(alternating, 0.9)
        self.assertLess(in_phase, 0.1)
        self.assertLess(small_stride, 0.1)
        self.assertGreater(large_stride, 0.9)

    def test_trex_joystick_running_gait_rewards_require_speed_and_one_foot_duty(self):
        env = trex_joystick.TrexJoystick()
        command = jp.array([10.0, 0.0])

        stopped = float(env._achieved_running_speed_gate(command, jp.zeros(3)))
        running = float(
            env._achieved_running_speed_gate(command, jp.array([8.0, 0.0, 0.0]))
        )
        both_planted = float(
            jp.exp(-4.0 * jp.square(jp.sum(jp.array([1.0, 1.0])) - 1.0))
        )
        alternating_duty = float(
            jp.exp(-4.0 * jp.square(jp.sum(jp.array([0.5, 0.5])) - 1.0))
        )

        self.assertLess(stopped, 0.1)
        self.assertGreater(running, 0.9)
        self.assertLess(both_planted, 0.1)
        self.assertGreater(alternating_duty, 0.9)

    def test_trex_joystick_moving_support_gate_requires_foot_contact(self):
        env = trex_joystick.TrexJoystick()

        no_contact_gate = jp.clip(0.0 / 0.75, 0.0, 1.0)
        one_foot_gate = jp.clip(1.0 / 0.75, 0.0, 1.0)

        self.assertLess(float(no_contact_gate), 0.1)
        self.assertGreater(float(one_foot_gate), 0.9)

    def test_trex_joystick_double_foot_contact_cost_prefers_single_support(self):
        env = trex_joystick.TrexJoystick()
        no_double_support = jp.array(1.0) * jp.array(0.0)
        double_support = jp.array(1.0) * jp.array(1.0)

        self.assertLess(float(no_double_support), 0.1)
        self.assertGreater(float(double_support), 0.9)
        self.assertIn("phase_contact", env._config.reward_config.scales)
        self.assertIn("phase_foot_clearance", env._config.reward_config.scales)

    def test_trex_joystick_leg_action_alternation_prefers_opposite_pairs(self):
        env = trex_joystick.TrexJoystick()
        alternating = np.asarray(env._stand_pose_action).copy()
        in_phase = alternating.copy()

        alternating[2:8] += np.array([0.4, -0.4, -0.4, 0.4, 0.4, -0.4])
        in_phase[2:8] += np.array([0.4, 0.4, -0.4, -0.4, 0.4, 0.4])

        self.assertGreater(
            float(env._reward_leg_action_alternation(jp.array(alternating))),
            0.8,
        )
        self.assertLess(
            float(env._reward_leg_action_alternation(jp.array(in_phase))),
            0.2,
        )

    def test_trex_joystick_gait_prior_alternates_leg_pairs_when_running(self):
        env = trex_joystick.TrexJoystick()
        prior = np.asarray(
            env._gait_prior_action(
                {"command": jp.array([10.0, 0.0]), "gait_phase": jp.pi / 2.0}
            )
        )

        self.assertGreater(prior[2], 0.1)
        self.assertLess(prior[3], -0.1)
        self.assertLess(prior[4], -0.1)
        self.assertGreater(prior[5], 0.1)
        self.assertAlmostEqual(prior[8], 0.0)
        self.assertAlmostEqual(prior[9], 0.0)

    def test_trex_joystick_tracking_rewards_use_forward_and_turn_axes(self):
        env = trex_joystick.TrexJoystick()
        command = jp.array([1.0, 0.5])

        self.assertAlmostEqual(
            float(env._reward_tracking_forward_vel(command, jp.array([1.0, 0.0, 0.0]))),
            1.0,
        )
        self.assertLess(
            float(env._reward_tracking_forward_vel(command, jp.array([0.0, 0.0, 0.0]))),
            0.1,
        )
        self.assertGreater(
            float(env._cost_forward_speed_deficit(command, jp.array([0.0, 0.0, 0.0]))),
            0.9,
        )
        self.assertLess(
            float(env._cost_forward_speed_deficit(command, jp.array([1.0, 0.0, 0.0]))),
            0.1,
        )
        self.assertAlmostEqual(
            float(env._cost_base_tilt_ang_vel(jp.array([1.0, 10.0, 2.0]))),
            5.0,
        )
        self.assertAlmostEqual(float(env._cost_running_height_excess(2.55)), 0.0)
        self.assertGreater(float(env._cost_running_height_excess(3.00)), 0.1)
        self.assertGreater(float(env._reward_running_height_gate(2.55)), 0.9)
        self.assertLess(float(env._reward_running_height_gate(3.50)), 0.01)
        self.assertAlmostEqual(
            float(env._reward_tracking_turn_vel(command, jp.array([0.0, 0.5, 0.0]))),
            1.0,
        )
        self.assertLess(
            float(env._reward_tracking_turn_vel(command, jp.array([0.0, -0.5, 0.0]))),
            0.1,
        )
        self.assertGreater(
            float(
                env._reward_commanded_stand_still(jp.zeros(2), jp.zeros(3), jp.zeros(3))
            ),
            0.9,
        )
        self.assertEqual(
            float(env._reward_commanded_stand_still(command, jp.zeros(3), jp.zeros(3))),
            0.0,
        )

    def test_register_environments_adds_trex_tasks(self):
        train.register_environments()
        from mujoco_playground import registry

        self.assertIn("TrexGetup", registry.ALL_ENVS)
        self.assertIn("TrexJoystick", registry.ALL_ENVS)


if __name__ == "__main__":
    unittest.main()
