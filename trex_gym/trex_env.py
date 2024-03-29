"""This implements the gym for a t-rex model.

"""

import trex_robot
from pybullet_envs.bullet import bullet_client

from baselines import logger

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import os
import pybullet


NUM_SUBSTEPS = 5
FLOOR_URDF_FILENAME = 'floor.urdf'
EARTH_GRAVITATIONAL_CONSTANT = 9.81
RENDER_HEIGHT = 720
RENDER_WIDTH = 960


class TrexBulletEnv(gym.Env):
    """The gym environment for the T-rex model.

    This simulates the standing of a Tyrannosaurus model.
    The observation space is expected to be the joint angles, velocities and torques.
    The action space is a desired joint angle and stiffness.
    The cost is formulated on raising the center of mass, maintaining it and utilizing minimum energy.

    """
    metadata = {
        "render.modes": ["human", "rgb_array"],
        "video.frames_per_second": 50
    }

    def __init__(self,
                 urdf_path,
                 action_repeat=1,
                 distance_weight=1.0,
                 energy_weight=0.005,
                 drift_weight=0.002,
                 render=False):
        """Environment for the T-rex model.

        :param urdf_path: path to the urdf data folder.
        :param action_repeat: number of simulation steps before actions are applied.
        :param distance_weight: weight of the distance term in the reward.
        :param energy_weight: weight of the energy term in the reward.
        :param render: whether to render the simulation.
        """
        self._time_step = 0.01
        self._urdf_path = urdf_path
        self._action_repeat = action_repeat
        self._num_bullet_solver_iterations = 300
        self._observation = []
        self._env_step_counter = 0
        self._is_render = render
        self._last_base_position = [0.] * 3
        self._weight_distance = distance_weight
        self._weight_energy = energy_weight
        self._weight_drift = drift_weight
        self._action_bound = 1
        self._cam_dist = 10.0
        self._cam_yaw = 90.0
        self._cam_pitch = -30.0
        self._last_frame_time = 0.0
        # PD control needs smaller time step for stability.
        self._time_step /= NUM_SUBSTEPS
        self._num_bullet_solver_iterations /= NUM_SUBSTEPS
        self._action_repeat *= NUM_SUBSTEPS

        connection_mode = pybullet.DIRECT
        if self._is_render:
            connection_mode = pybullet.GUI
        self._pybullet_client = bullet_client.BulletClient(
                connection_mode=connection_mode)

        self._starting_configuration = {'femur_L_joint': -0.6,
                                        'tibia_L_joint': 0.4,
                                        'tarsometatarsus_L_joint': -1.2,
                                        'femur_R_joint': -0.6,
                                        'tibia_R_joint': 0.4,
                                        'tarsometatarsus_R_joint': -1.2
                                        }

        self.model = None
        self.np_random = None
        self.seed()
        self.reset()
        action_low, action_high = self.model.get_action_limits()
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)
        observation_low, observation_high = self.model.get_observation_limits()
        self.observation_space = spaces.Box(low=observation_low, high=observation_high, dtype=np.float32)

    def reset(self):
        if self.model:
            self.model.reset()
        else:
            self._pybullet_client.resetSimulation()
            plane = self._pybullet_client.loadURDF(os.path.join(os.path.dirname(self._urdf_path), FLOOR_URDF_FILENAME))
            self._pybullet_client.changeVisualShape(plane, -1, rgbaColor=[1, 1, 1, 0.9])
            origin_xyz = [0., 0., 3.]
            origin_rpy = [0., 0., 0.]
            self.model = trex_robot.TrexRobot(self._pybullet_client, self._urdf_path,
                                              world_origin_xyz=origin_xyz, world_origin_rpy=origin_rpy,
                                              starting_configuration=self._starting_configuration)
            self.model.reset()
        if self._is_render:
            self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_PLANAR_REFLECTION, 0)
            self._pybullet_client.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw, self._cam_pitch,
                                                             self.model.get_base_position())
        self._pybullet_client.setPhysicsEngineParameter(numSolverIterations=int(self._num_bullet_solver_iterations))
        self._pybullet_client.setTimeStep(self._time_step)
        self._pybullet_client.setGravity(0, 0, -EARTH_GRAVITATIONAL_CONSTANT)
        self._env_step_counter = 0
        self._last_base_position = self.model.get_base_position()
        self._pybullet_client.stepSimulation()
        print(self._pybullet_client.getPhysicsEngineParameters())
        return self.model.get_observations()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """Step forward the simulation, given the action.

        Args:
          action: A list of desired joint angles and stiffnesses.

        Returns:
          observations: The angles, velocities and torques of all revolute joints.
          reward: The reward for the current state-action pair.
          done: Whether the episode has ended.
          info: A dictionary that stores diagnostic information.

        Raises:
          ValueError: The action dimension is not the same as the number of motors.
          ValueError: The magnitude of actions is out of bounds.
        """
        # Clipping the action here is a hack, because PPO2 does not believe in using the action space :(.
        action_scaled = action
        # action_scaled[(len(action)//2):] *= 1.0e-3
        clipped_action = np.clip(action_scaled, self.action_space.low, self.action_space.high)
        for _ in range(self._action_repeat):
            self.model.set_actions(clipped_action)
            self._pybullet_client.stepSimulation()

        self._env_step_counter += 1
        self._observation = self.model.get_observations()
        return self._observation, self.compute_reward(), self.should_terminate(), {}

    def render(self, mode='rgb_array', close=False):
        base_position = self.model.get_base_position()
        if mode is 'human':
            distance = self._cam_dist
            yaw = self._cam_yaw
            pitch = self._cam_pitch
            self._pybullet_client.resetDebugVisualizerCamera(distance, yaw, pitch, base_position)
        elif mode is 'rgb_array':
            view_matrix = self._pybullet_client.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=base_position,
                distance=self._cam_dist,
                yaw=self._cam_yaw,
                pitch=self._cam_pitch,
                roll=0,
                upAxisIndex=2)
            proj_matrix = self._pybullet_client.computeProjectionMatrixFOV(
                fov=60, aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
                nearVal=0.1, farVal=100.0)
            (_, _, px, _, _) = self._pybullet_client.getCameraImage(
                width=RENDER_WIDTH, height=RENDER_HEIGHT, viewMatrix=view_matrix,
                projectionMatrix=proj_matrix, renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
            rgb_array = np.array(px)
            rgb_array = np.reshape(rgb_array, (RENDER_HEIGHT, RENDER_WIDTH, 4))
            rgb_array = rgb_array[:, :, :3]
            return rgb_array
        return np.array([])

    def should_terminate(self):
        return False

    def compute_reward(self):
        current_base_position = self.model.get_head_position()
        station_keeping_penalty = self._weight_drift * (current_base_position[0]**2 + current_base_position[1]**2)
        lifting_com_penalty = self._weight_distance * (2.5 - current_base_position[2])**2
        energy_penalty = self._weight_energy * self.model.get_total_joint_power()
        self._last_base_position = current_base_position
        reward = -lifting_com_penalty - station_keeping_penalty - energy_penalty
        logger.logkv('penalty_lifting_com', lifting_com_penalty)
        logger.logkv('penalty_station_keeping', station_keeping_penalty)
        logger.logkv('penalty_energy', energy_penalty)
        return reward
