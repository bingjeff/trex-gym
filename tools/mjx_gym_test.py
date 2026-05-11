import unittest

import jax
import jax.numpy as jp
import mujoco
import numpy as np

from mjx_gym import train
from mjx_gym import trex_constants
from mjx_gym import trex_getup


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
        self.assertEqual((78,), state.obs["state"].shape)
        self.assertEqual((164,), state.obs["privileged_state"].shape)
        self.assertEqual((38,), state.data.qpos.shape)
        self.assertEqual((37,), state.data.qvel.shape)
        self.assertEqual((10,), state.data.ctrl.shape)

        next_state = env.step(state, jp.zeros(env.action_size))
        self.assertEqual((78,), next_state.obs["state"].shape)
        self.assertEqual((164,), next_state.obs["privileged_state"].shape)
        self.assertGreater(float(next_state.data.time), 0.0)

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

    def test_trex_getup_mass_scaled_gains_settle_joint_perturbations(self):
        env = trex_getup.TrexGetup()
        model = env.mj_model
        self.assertGreater(np.min(model.jnt_stiffness[1:]), 1000.0)
        self.assertGreater(np.min(model.dof_damping[6:]), 100.0)
        self.assertGreater(np.min(model.actuator_gainprm[:, 0]), 1000.0)

        for joint_name in (
            "joint_vertebra_caudal_24",
            "joint_toe_04_d_right",
            "joint_femur_right",
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

    def test_register_environments_adds_trex_getup(self):
        train.register_environments()
        from mujoco_playground import registry

        self.assertIn("TrexGetup", registry.ALL_ENVS)


if __name__ == "__main__":
    unittest.main()
