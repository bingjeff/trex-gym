#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import trex_env

from absl import app
from absl import flags

from baselines import bench
from baselines import logger
from baselines.common import misc_util
from baselines.common.vec_env import vec_normalize
from baselines.ppo2 import ppo2
from baselines.ppo2 import policies
from baselines.common.vec_env import dummy_vec_env

FLAGS = flags.FLAGS

flags.DEFINE_bool('train', 'Whether to start training.', True)
flags.DEFINE_bool('play', 'Whether to run the policy after training.', False)
flags.DEFINE_bool('debug_render', 'Whether to show the bullet debug render.', False)
flags.DEFINE_integer('num_timesteps', 'Number of time step iterations.', 1e6)
flags.DEFINE_integer('random_seed', 'Seed to use for random initialization.', 0)

_NUM_CPUS = 1
_URDF_PATH = './assets/trex.urdf'


def train(training_env, num_timesteps, seed):
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=_NUM_CPUS,
                            inter_op_parallelism_threads=_NUM_CPUS)
    tf.Session(config=config).__enter__()

    def make_env():
        return bench.Monitor(training_env, logger.get_dir(), allow_early_resets=True)

    env = dummy_vec_env.DummyVecEnv([make_env])
    env = vec_normalize.VecNormalize(env)

    misc_util.set_global_seeds(seed)
    policy = policies.MlpPolicy
    model = ppo2.learn(policy=policy,
                       env=env,
                       nsteps=2048,
                       nminibatches=32,
                       lam=0.95,
                       gamma=0.99,
                       noptepochs=10,
                       log_interval=1,
                       ent_coef=0.0,
                       lr=3e-4,
                       cliprange=0.2,
                       total_timesteps=num_timesteps)

    return model, env


def main(argv):
    del argv  # Unused.

    logger.configure()

    if FLAGS.train:
        logger.log("Training model.")
        training_env = trex_env.TrexBulletEnv(_URDF_PATH,
                                              action_repeat=1,
                                              distance_weight=1.0,
                                              energy_weight=0.005,
                                              drift_weight=0.002,
                                              render=FLAGS.debug_render)
        model, env = train(training_env, num_timesteps=FLAGS.num_timesteps, seed=FLAGS.random_seed)

    if FLAGS.train and FLAGS.play:
        logger.log("Running trained model.")
        obs = np.zeros((env.num_envs,) + env.observation_space.shape)
        obs[:] = env.reset()
        while True:
            actions = model.step(obs)[0]
            obs[:] = env.step(actions)[0]
            env.render()


if __name__ == '__main__':
    app.run(main)
