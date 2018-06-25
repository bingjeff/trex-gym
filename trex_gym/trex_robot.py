import numpy as np
import pybullet


class BulletRobotBase(object):
    """Base class for Bullet robots.

    Provides basic functionality and methods for setting up a robot actor and getting/setting commonly used properties.

    Classes that inherit from this base class should override the following methods:

    * reset_configuration
    * get_observations
    * set_actions

    """

    def __init__(self, pybullet_client, urdf_path, world_origin_xyz=None, world_origin_rpy=None, self_collision=False,
                 fixed_base=False):
        """Holds the bullet body handle.

        :param pybullet_client: BulletClient object from pybullet_envs.bullet.bullet_client.
        :param urdf_path: path to the URDF to load.
        :param world_origin_xyz: a list of coordinates [x, y, z] to set the location of the model. If none,
         defaults to the origin.
        :param world_origin_rpy: a list of euler angles [r, p, y] to set the orientation of the model. If none,
         defaults to identity.
        :param self_collision: whether the model should be allowed to collide with itself. Defaults to False.
        :param fixed_base: whether the model should be imported with a fixed base. Defaults to False.
        """
        self.client = pybullet_client
        self.urdf_path = urdf_path
        self.world_origin_xyz = world_origin_xyz
        self.world_origin_rpy = world_origin_rpy
        self.self_collision = self_collision
        self.fixed_base = fixed_base
        self.body_handle = None

    def reset(self, reload_urdf=False):
        """Returns the model to its initial state.

        This provides a method to initialize and reset the model, which can also reload the model from disk.

        :param reload_urdf: whether to load the model from disk on reset. Defaults to False.
        """
        zero_vec = [0., 0., 0.]
        if reload_urdf or self.body_handle is None:
            arguments = {}
            if self.fixed_base:
                arguments['useFixedBase'] = 1
            if self.self_collision:
                arguments['flags'] = self.client.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
            if self.body_handle:
                self.client.removeBody(self.body_handle)
            print(arguments)
            self.body_handle = self.client.loadURDF(self.urdf_path, **arguments)
        position = zero_vec
        if self.world_origin_xyz:
            position = self.world_origin_xyz
        orientation = [0., 0., 0., 1.]
        if self.world_origin_rpy:
            orientation = self.client.getQuaternionFromEuler(self.world_origin_rpy)
        self.client.resetBasePositionAndOrientation(self.body_handle, position, orientation)
        self.client.resetBaseVelocity(self.body_handle, zero_vec, zero_vec)
        self.reset_configuration()

    def reset_configuration(self):
        """Overridable function to set the configuration on model reset.

        This method should be overridden for specialization. The base method zeros the joint state and removes joint
        control.
        """
        self.zero_joint_state()
        self.remove_joint_control()

    def get_observations(self):
        """Overridable function to return the appropriate observations from the model

        :return: a list of observation values.
        """
        return []

    def set_actions(self, actions):
        """Overridable function to set the appropriate actions for the model.

        :param actions: a list of actions to set.
        """
        pass

    def get_joint_info(self, joint_index):
        """Get the stored info for a specific joint.

        This is a thin wrapper of the pybullet getJointInfo command.

        :param joint_index: the index bullet gives for a joint.
        :return: a dictionary with the resulting joint information.
        """
        result = self.client.getJointInfo(self.body_handle, joint_index)
        names = [
            'jointIndex',
            'jointName',
            'jointType',
            'qIndex',
            'uIndex',
            'flags',
            'jointDamping',
            'jointFriction',
            'jointLowerLimit',
            'jointUpperLimit',
            'jointMaxForce',
            'jointMaxVelocity',
            'linkName',
            'jointAxis',
            'parentFramePos',
            'parentFrameOrn',
            'parentIndex',
        ]
        return {key: value for key, value in zip(names, result)}

    def _get_joint_state_names(self):
        return [
            'jointPosition',
            'jointVelocity',
            'jointReactionForces',
            'appliedJointMotorTorque',
        ]

    def get_joint_state(self, joint_index):
        """Get the joint state.

        This is a thin wrapper of the pybullet getJointState command.

        :param joint_index: the index bullet gives for a joint.
        :return: a dictionary with the resulting joint state.
        """
        result = self.client.getJointState(self.body_handle, joint_index)
        names = self._get_joint_state_names()
        return {key: value for key, value in zip(names, result)}

    def get_joint_states(self, joint_index_list):
        """Get the joint states.

        This is a thin wrapper of the pybullet getJointStates command.

        :param joint_index_list: a list of indices based on the index that bullet gives a joint.
        :return: a dictionary with the resulting joint states
        """
        results = self.client.getJointStates(self.body_handle, joint_index_list)
        names = self._get_joint_state_names()
        return {name: [result[c] for result in results] for c, name in enumerate(names)}

    def get_joint_map(self):
        """Get a map between the name of a joint and the corresponding joint index.

        :return: a dictionary with key joint name and value joint index.
        """
        result = {}
        for joint_index in range(self.client.getNumJoints(self.body_handle)):
            joint_info = self.get_joint_info(joint_index)
            result[joint_info['jointName']] = joint_info['jointIndex']
        return result

    def get_link_info(self, link_index):
        """Get the dynamics info for a specified link.

        This is a thin wrapper around getDynamicsInfo.

        :param link_index: a bullet index corresponding to the desired link.
        :return: a dictionary with the resulting link information.
        """
        result = self.client.getDynamicsInfo(self.body_handle, link_index)
        names = [
            'mass',
            'lateral_friction',
            'local_inertia_diagonal',
            'local_inertia_position',
            'local_inertia_orientation',
            'restitution',
            'rolling_friction',
            'spinning_friction',
            'contact_damping',
            'contact_stiffness',
        ]
        return {key: value for key, value in zip(names, result)}

    def get_link_state(self, link_index):
        """Get the link state.

        This is a thin wrapper of the pybullet getLinkState command, but with all kinematics calc turned on.

        :param link_index: the index bullet gives for a link.
        :return: a dictionary with the resulting link state.
        """
        result = self.client.getLinkState(self.body_handle, link_index, computeLinkVelocity=1,
                                          computeForwardKinematics=1)
        names = [
            'linkWorldPosition',
            'linkWorldOrientation',
            'localInertialFramePosition',
            'localInertialFrameOrientation',
            'worldLinkFramePosition',
            'worldLinkFrameOrientation',
            'worldLinkLinearVelocity',
            'worldLinkAngularVelocity',
        ]
        return {key: value for key, value in zip(names, result)}

    def get_link_map(self):
        """Get a map between the name of a link and the corresponding link index.

        :return: a dictionary with key link name and value link index.
        """
        result = {}
        for joint_index in range(self.client.getNumJoints(self.body_handle)):
            joint_info = self.get_joint_info(joint_index)
            result[joint_info['linkName']] = joint_info['jointIndex']
        return result

    def set_joint_state(self, joint_index, position, velocity=0.0):
        """Set the joint state.

        This is a thin wrapper around resetJointState.

        :param joint_index: the index bullet gives for a joint.
        :param position: the value in radians to set the joint position.
        :param velocity: the value in radians / sec to set the joint velocity.
        """
        self.client.resetJointState(self.body_handle, joint_index, targetValue=position, targetVelocity=velocity)

    def remove_joint_control(self):
        """Remove motor control for all joints.

        This sets the joint control to position mode with zero gains for all joints.
        """
        num_joints = self.client.getNumJoints(self.body_handle)
        zero_vec = [0.0] * num_joints
        joint_indices = range(num_joints)
        control_mode = self.client.POSITION_CONTROL
        self.client.setJointMotorControlArray(self.body_handle, joint_indices, control_mode, targetPositions=zero_vec,
                                              targetVelocities=zero_vec, forces=zero_vec, positionGains=zero_vec,
                                              velocityGains=zero_vec)

    def zero_joint_state(self):
        """Set the joint state to zero.

        This sets the joint position and velocity states to zero for all joints.
        """
        for joint_index in range(self.client.getNumJoints(self.body_handle)):
            self.set_joint_state(joint_index, 0.0, velocity=0.0)


class TrexRobot(BulletRobotBase):
    """Class to hold T-rex specific models.
    """

    _MAX_JOINT_TORQUE_IN_NM = 300000.0

    def __init__(self, pybullet_client, urdf_path, world_origin_xyz=None, world_origin_rpy=None, starting_configuration=None):
        """Holds the bullet body handle.

        :param pybullet_client: BulletClient object from pybullet_envs.bullet.bullet_client.
        :param urdf_path: path to the URDF to load.
        :param world_origin_xyz: a list of coordinates [x, y, z] to set the location of the model. If none,
         defaults to the origin.
        :param world_origin_rpy: a list of euler angles [r, p, y] to set the orientation of the model. If none,
         defaults to identity.
        :param starting_configuration: a dictionary with key a string identifying the joint and value the angle to set
         on configuration reset. Key/value only needs to be set for non-zero configuration joints.
        """
        self_collision = False
        fixed_base = False
        super(TrexRobot, self).__init__(pybullet_client, urdf_path, world_origin_xyz=world_origin_xyz,
                                        world_origin_rpy=world_origin_rpy, self_collision=self_collision,
                                        fixed_base=fixed_base)
        self._revolute_joint_indices = []
        if starting_configuration:
            self._starting_configuration = starting_configuration
        else:
            self._starting_configuration = {}
        self._total_mass = 0.0

    def _get_joints_by_type(self, joint_type):
        """Internal method to return only joints of a specific type.

        :param joint_type: bullet enum describing joint type, e.g. JOINT_REVOLUTE.
        :return: a dictionary with key joint name and value joint index.
        """
        result = {}
        for joint_index in range(self.client.getNumJoints(self.body_handle)):
            joint_info = self.get_joint_info(joint_index)
            if joint_info['jointType'] == joint_type:
                result[joint_info['jointName']] = joint_info['jointIndex']
        return result

    def reset_configuration(self):
        """Set the configuration on model reset.
        """
        # Set the joints to their zero pose.
        self.zero_joint_state()
        joint_map = self.get_joint_map()
        for joint_name, joint_position in self._starting_configuration.items():
            joint_index = joint_map[joint_name]
            self.set_joint_state(joint_index, joint_position)
        self.remove_joint_control()
        # Grab the revolute joints and add them to the cache.
        joints = self._get_joints_by_type(pybullet.JOINT_REVOLUTE)
        joint_names = joints.keys()
        joint_names.sort()
        self._revolute_joint_indices = [joints[name] for name in joint_names]
        # Calculate the model mass.
        link_map = self.get_link_map()
        link_masses = [self.get_link_info(l)['mass'] for l in link_map.values()]
        self._total_mass = np.sum(link_masses)

    def get_base_position(self):
        """Get the location of the base link.

        :return: a list of [x, y, z] to the CoM of the base link in world coordinates.
        """
        position, _ = self.client.getBasePositionAndOrientation(self.body_handle)
        return position

    def _get_joint_limits(self):
        """Internal method to get the joint limits.
        """
        lower_limit = []
        upper_limit = []
        for index in self._revolute_joint_indices:
            info = self.get_joint_info(index)
            lower_limit.append(info['jointLowerLimit'])
            upper_limit.append(info['jointUpperLimit'])
        return lower_limit, upper_limit

    def get_observation_limits(self):
        """Get the observation limits.

        :return: a tuple of np.arrays (lower_limit, upper_limit) giving the acceptable observation limits.
        """
        num_joints = len(self._revolute_joint_indices)
        lower_limit, upper_limit = self._get_joint_limits()
        lower_limit.extend([-np.inf] * 2 * num_joints)
        upper_limit.extend([np.inf] * 2 * num_joints)
        return np.array(lower_limit), np.array(upper_limit)

    def get_observations(self):
        """Get the appropriate observations from the model.

        :return: a list of observation values.
        """
        joint_states = self.get_joint_states(self._revolute_joint_indices)
        return joint_states['jointPosition'] + joint_states['jointVelocity'] + joint_states['appliedJointMotorTorque']

    def get_total_joint_power(self):
        """Get the instantaneous total power of the joints.

        Calculates the sum of the absolute value of joint torque times joint velocity, e.g. sum_i abs(q_i * tau_i).

        :return: a float representing the total power in Joules (N-m/s).
        """
        joint_states = self.get_joint_states(self._revolute_joint_indices)
        return np.sum(np.fabs(np.multiply(joint_states['jointVelocity'], joint_states['appliedJointMotorTorque'])))

    def _set_joint_tracking(self, theta, kp):
        """Internal method for setting the joint controller.

        For now the damping is set automatically based on a minimum rise-time damping (0.707) and assuming worst case
        mass. This may be excessively damped for many cases.

        xdd - b/m * xd + k/m * x
        s^2 + 2*z*w * s + w^2
        w = sqrt(k/m)
        b = 2 * z * sqrt(m * k)

        b = 2 * sqrt(2)/2 * sqrt(m*k) = sqrt(2*m*k)

        Model currently weighs ~4800 kg and is 5 m tall. That means that the ankle joint may need to take on the order
        of 4800 * 10 * 5 = 240,000 N-m just for static loading. If the ankle should accelerate at some fraction of G
        then that number should go up. For now assume max torque to be 300,000 N-m.

        :param theta: desired tracking position.
        :param kp: desired tracking stiffness.
        """
        num_joints = len(self._revolute_joint_indices)
        max_torque = [self._MAX_JOINT_TORQUE_IN_NM] * num_joints
        damping_gain = np.sqrt(2.0 * self._total_mass * np.array(kp))
        vec_zero = [0.0] * num_joints
        control_mode = self.client.POSITION_CONTROL
        self.client.setJointMotorControlArray(self.body_handle,
                                              self._revolute_joint_indices,
                                              control_mode,
                                              targetPositions=theta,
                                              targetVelocities=vec_zero,
                                              forces=max_torque,
                                              positionGains=kp,
                                              velocityGains=damping_gain.tolist())

    def set_actions(self, actions):
        """Set the appropriate actions for the model.

        :param actions: a list of actions to set.
        """
        num_joints = len(self._revolute_joint_indices)
        theta = actions[:num_joints]
        kp = actions[num_joints:]
        self._set_joint_tracking(theta, kp)

    def get_action_limits(self):
        """Get the limits that are associated with the available actions.

        :return: a tuple of np.arrays (lower_limit, upper_limit) giving the acceptable action limits.
        """
        num_joints = len(self._revolute_joint_indices)
        lower_limit, upper_limit = self._get_joint_limits()
        lower_limit.extend([0.0] * num_joints)
        upper_limit.extend([np.inf] * num_joints)
        return np.array(lower_limit), np.array(upper_limit)
