import unittest

import jax
import jax.numpy as jp

from mjx_gym import train
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

    def test_register_environments_adds_trex_getup(self):
        train.register_environments()
        from mujoco_playground import registry

        self.assertIn("TrexGetup", registry.ALL_ENVS)


if __name__ == "__main__":
    unittest.main()
