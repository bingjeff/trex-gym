import unittest

import numpy as np

from tools import urdf_parsing

_TEST_URDF = """<?xml version="1.0" encoding="utf-8"?>
<robot name="stan_t_rex">
  <joint name="joint_femur_right" type="revolute">
    <origin rpy="-3.141592502593994 -0.0 0.27925267815589905"
      xyz="0.0171966552734375 -0.2207697629928589 0.2492464929819107" />
    <parent link="link_vertebrae_sacral" />
    <child link="link_femur_right" />
    <axis xyz="0.0 0.0 1.0" />
    <dynamics damping="1.0" />
    <limit effort="100.0" lower="-1.5707963267948966" upper="1.5707963267948966" velocity="1.0" />
  </joint>
  <joint name="joint_tibia_right" type="revolute">
    <origin rpy="0.0 -0.0 0.593411922454834"
      xyz="0.0002651557151693851 1.0657678842544556 -0.14526696503162384" />
    <parent link="link_femur_right" />
    <child link="link_tibia_right" />
    <axis xyz="0.0 0.0 1.0" />
    <dynamics damping="1.0" />
    <limit effort="100.0" lower="-0.5934119456780721" upper="1.5707963267948966" velocity="1.0" />
  </joint>
    <link name="link_femur_right">
    <inertial>
      <origin rpy="0.8730872273445129 1.5707982778549194 -0.27938660979270935"
        xyz="0.059595704078674316 0.5649366974830627 -0.1708899438381195" />
      <mass value="390.7456359863281" />
      <inertia ixx="68.57667541503906" ixy="0.0" ixz="0.0" iyy="65.79754638671875" iyz="0.0"
        izz="19.348051071166992" />
    </inertial>
    <visual>
      <origin rpy="1.291545033454895 1.5707961320877075 0.0"
        xyz="0.07209686934947968 0.555097222328186 -0.18508979678153992" />
      <geometry>
        <mesh filename="meshes/femur_right.obj" scale="1.0 1.0 1.0" />
      </geometry>
    </visual>
  </link>
  <link name="link_tibia_right">
    <inertial>
      <origin rpy="0.6178942918777466 1.570798397064209 -1.2070443630218506"
        xyz="0.07327674329280853 0.6192649602890015 -0.07032167166471481" />
      <mass value="219.72840881347656" />
      <inertia ixx="27.967979431152344" ixy="0.0" ixz="0.0" iyy="22.775781631469727" iyz="0.0"
        izz="9.164529800415039" />
    </inertial>
    <visual>
      <origin rpy="1.8849579095840454 1.5707961320877075 0.0"
        xyz="0.002119977492839098 0.6098067164421082 -0.05006600543856621" />
      <geometry>
        <mesh filename="meshes/tibia_right.obj" scale="1.0 1.0 1.0" />
      </geometry>
    </visual>
    <visual>
      <origin rpy="1.8849579095840454 1.5707961320877075 0.0"
        xyz="0.0012490012450143695 0.4370576739311218 -0.19665572047233582" />
      <geometry>
        <mesh filename="meshes/fibula_right.obj" scale="1.0 1.0 1.0" />
      </geometry>
    </visual>
  </link>
</robot>
"""


class TestUrdfParsing(unittest.TestCase):
    def test_urdf_round_trip(self):
        urdf_original = urdf_parsing.Urdf.from_string(_TEST_URDF)
        urdf_string = urdf_original.to_string()
        urdf_roundtrip = urdf_parsing.Urdf.from_string(urdf_string)
        self.assertEqual(urdf_original.name, urdf_roundtrip.name)
        self.assertEqual(len(urdf_original.joints), len(urdf_roundtrip.joints))
        for name in urdf_original.joints:
            o = urdf_original.joints[name]
            r = urdf_roundtrip.joints[name]
            self.assertEqual(o.child_name, r.child_name)
            self.assertEqual(o.parent_name, r.parent_name)
            self.assertEqual(o.type, r.type)
            np.testing.assert_allclose(o.axis, r.axis)
            np.testing.assert_allclose(o.limits.position, r.limits.position)
            np.testing.assert_allclose(
                o.origin.translation, r.origin.translation
            )
            np.testing.assert_allclose(
                o.origin.rotation.as_rotvec(), r.origin.rotation.as_rotvec()
            )
        for name in urdf_original.links:
            o = urdf_original.links[name]
            r = urdf_roundtrip.links[name]
            self.assertEqual(o.name, r.name)
            self.assertAlmostEqual(o.inertia.mass, r.inertia.mass)
            np.testing.assert_allclose(o.inertia.inertia, r.inertia.inertia)
            np.testing.assert_allclose(
                o.inertia.origin.translation, r.inertia.origin.translation
            )
            np.testing.assert_allclose(
                o.inertia.origin.rotation.as_rotvec(),
                r.inertia.origin.rotation.as_rotvec(),
            )
            for os, rs in zip(o.visual_shapes, r.visual_shapes):
                self.assertEqual(type(os), type(rs))
                np.testing.assert_allclose(
                    os.origin.translation, rs.origin.translation
                )
                np.testing.assert_allclose(
                    os.origin.rotation.as_rotvec(),
                    rs.origin.rotation.as_rotvec(),
                )
            for os, rs in zip(o.collision_shapes, r.collision_shapes):
                self.assertEqual(type(os), type(rs))
                np.testing.assert_allclose(
                    os.origin.translation, rs.origin.translation
                )
                np.testing.assert_allclose(
                    os.origin.rotation.as_rotvec(),
                    rs.origin.rotation.as_rotvec(),
                )
