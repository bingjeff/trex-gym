#!/usr/bin/env python3
import joblib
import numpy as np
import os
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

from PIL import Image

FLAGS = flags.FLAGS

flags.DEFINE_bool('train', True, 'Whether to start training.')
flags.DEFINE_bool('play', True, 'Whether to run the policy after training.')
flags.DEFINE_bool('debug_render', False, 'Whether to show the bullet debug render.')
flags.DEFINE_integer('num_timesteps', int(1e6), 'Number of time step iterations.')
flags.DEFINE_integer('num_play_timesteps', int(1e3), 'Number of time step iterations to render.')
flags.DEFINE_integer('random_seed', 0, 'Seed to use for random initialization.')

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
                       total_timesteps=num_timesteps,
                       save_interval=10)

    return model, env


def build_environment(action_repeat=1, distance_weight=1.0e3, energy_weight=1.0e-6, drift_weight=1.0e-2, render=False):
    return trex_env.TrexBulletEnv(_URDF_PATH,
                                  action_repeat=action_repeat,
                                  distance_weight=distance_weight,
                                  energy_weight=energy_weight,
                                  drift_weight=drift_weight,
                                  render=render)


def replay(load_path, num_time_steps, render=False):
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=_NUM_CPUS,
                            inter_op_parallelism_threads=_NUM_CPUS)
    tf.Session(config=config).__enter__()

    sess = tf.get_default_session()

    training_env = build_environment(render=render)
    env = dummy_vec_env.DummyVecEnv([lambda: training_env])
    env = vec_normalize.VecNormalize(env)

    policy = policies.MlpPolicy
    ob_space = env.observation_space
    ac_space = env.action_space
    num_batch_ac = env.num_envs
    model = policy(sess, ob_space, ac_space, num_batch_ac, 1, reuse=False)

    with tf.variable_scope('model'):
        trained_vars = tf.trainable_variables()
        loaded_vars = joblib.load(load_path)
        restore_ops = []
        for trained, loaded in zip(trained_vars, loaded_vars):
            restore_ops.append(trained.assign(loaded))
        sess.run(restore_ops)

    observations = env.reset()
    episode_reward = 0.0
    states = model.initial_state
    dones = [False for _ in range(env.num_envs)]
    for _ in range(num_time_steps):
        env.render()
        actions, values, states, neglogpacs = model.step(observations, states, dones)
        observations, reward, done, info = env.step(actions)
        episode_reward += reward
        print("Episode reward: {}".format(episode_reward))


def main(argv):
    del argv  # Unused.

    logger.configure()

    if FLAGS.train:
        logger.log("Training model.")
        training_env = build_environment(render=FLAGS.debug_render)
        logger.log("--num actions: {}".format(len(training_env.model.get_action_limits()[0])))
        logger.log("--num joints: {}".format(len(training_env.model._revolute_joint_indices)))
        logger.log("--total mass: {}".format(training_env.model._total_mass))
        model, env = train(training_env, num_timesteps=FLAGS.num_timesteps, seed=FLAGS.random_seed)

    if FLAGS.train and FLAGS.play:
        logger.log("Running trained model.")
        obs = np.zeros((env.num_envs,) + env.observation_space.shape)
        obs[:] = env.reset()
        for frame_idx in range(FLAGS.num_play_timesteps):
            actions = model.step(obs)[0]
            obs[:] = env.step(actions)[0]
            rgb_img = env.render(mode='rgb_array')[0]
            im = Image.fromarray(rgb_img.astype(np.uint8))
            im.save(os.path.join(logger.get_dir(), '{:05d}-of-{:05d}.png'.format(frame_idx, FLAGS.num_play_timesteps)))



if __name__ == '__main__':
    app.run(main)
