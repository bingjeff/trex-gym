import unittest
from pathlib import Path

import mujoco
from mujoco import mjx
import numpy as np

from tools import mjx_model_simplification
from tools import mujoco_parsing
from tools import urdf_parsing

_ASSET_DIR = Path(__file__).resolve().parents[1] / "assets"

_FIXED_CHAIN_URDF = """<?xml version="1.0" encoding="utf-8"?>
<robot name="fixed_chain">
  <joint name="joint_root_to_fixed" type="fixed">
    <origin rpy="0.0 0.0 0.0" xyz="1.0 0.0 0.0" />
    <parent link="root" />
    <child link="fixed_child" />
  </joint>
  <joint name="joint_fixed_to_hinge" type="revolute">
    <origin rpy="0.0 0.0 0.0" xyz="0.0 2.0 0.0" />
    <parent link="fixed_child" />
    <child link="hinge_child" />
    <axis xyz="0.0 0.0 1.0" />
    <limit lower="-1.0" upper="1.0" />
  </joint>
  <link name="root">
    <inertial>
      <origin rpy="0.0 0.0 0.0" xyz="0.0 0.0 0.0" />
      <mass value="2.0" />
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0" />
    </inertial>
    <visual>
      <origin rpy="0.0 0.0 0.0" xyz="0.0 0.0 0.0" />
      <geometry>
        <mesh filename="meshes/root.obj" />
      </geometry>
    </visual>
  </link>
  <link name="fixed_child">
    <inertial>
      <origin rpy="0.0 0.0 0.0" xyz="0.0 0.0 0.0" />
      <mass value="3.0" />
      <inertia ixx="2.0" ixy="0.0" ixz="0.0" iyy="2.0" iyz="0.0" izz="2.0" />
    </inertial>
    <collision>
      <origin rpy="0.0 0.0 0.0" xyz="0.5 0.0 0.0" />
      <geometry>
        <capsule radius="0.1" length="0.2" />
      </geometry>
    </collision>
  </link>
  <link name="hinge_child">
    <inertial>
      <origin rpy="0.0 0.0 0.0" xyz="0.0 0.0 0.0" />
      <mass value="1.0" />
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0" />
    </inertial>
  </link>
</robot>
"""


class TestMjxModelSimplification(unittest.TestCase):
    def test_fixed_joint_links_are_fused_in_urdf_space(self):
        urdf = urdf_parsing.Urdf.from_string(_FIXED_CHAIN_URDF)
        simplified = mjx_model_simplification.simplify_urdf_for_mjx(urdf)

        self.assertEqual({"root", "hinge_child"}, set(simplified.links))
        self.assertEqual(["joint_fixed_to_hinge"], list(simplified.joints))
        self.assertFalse(simplified.links["root"].visual_shapes)
        self.assertEqual(1, len(simplified.links["root"].collision_shapes))
        np.testing.assert_allclose(
            [1.5, 0.0, 0.0],
            simplified.links["root"].collision_shapes[0].origin.translation,
        )

        joint = simplified.joints["joint_fixed_to_hinge"]
        self.assertEqual("root", joint.parent_name)
        self.assertEqual("hinge_child", joint.child_name)
        np.testing.assert_allclose([1.0, 2.0, 0.0], joint.origin.translation)

        inertial = simplified.links["root"].inertia
        self.assertAlmostEqual(5.0, inertial.mass)
        np.testing.assert_allclose([0.6, 0.0, 0.0], inertial.origin.translation)

    def test_trex_simplified_mjcf_has_no_visual_meshes_and_loads_in_mjx(self):
        urdf = urdf_parsing.Urdf.from_element(
            urdf_parsing.read_root_node_from_urdf(str(_ASSET_DIR / "trex.urdf"))
        )
        simplified = mjx_model_simplification.simplify_urdf_for_mjx(urdf)

        self.assertEqual(
            len([joint for joint in urdf.joints.values() if joint.type != "fixed"]),
            len(simplified.joints),
        )
        self.assertTrue(
            all(joint.type != "fixed" for joint in simplified.joints.values())
        )
        self.assertTrue(
            all(not link.visual_shapes for link in simplified.links.values())
        )
        self.assertLess(len(simplified.links), len(urdf.links))
        self.assertEqual(
            sum(len(link.collision_shapes) for link in urdf.links.values()),
            sum(len(link.collision_shapes) for link in simplified.links.values()),
        )
        self.assertAlmostEqual(
            sum(link.inertia.mass for link in urdf.links.values()),
            sum(link.inertia.mass for link in simplified.links.values()),
        )

        full_mjcf = mujoco_parsing.urdf_to_mujoco(urdf)
        simplified_mjcf = mjx_model_simplification.urdf_to_mjx_mujoco(urdf)
        self.assertLess(
            len(simplified_mjcf.findall(".//body")),
            len(full_mjcf.findall(".//body")),
        )
        self.assertFalse(simplified_mjcf.findall("./asset/mesh"))
        self.assertFalse(
            [
                geom
                for geom in simplified_mjcf.findall(".//geom")
                if geom.get("class") == "visual" or geom.get("type") == "mesh"
            ]
        )
        self.assertFalse(
            [
                body.get("name")
                for body in simplified_mjcf.findall(".//body")
                if not body.findall("joint") and not body.findall("freejoint")
            ]
        )
        self.assertEqual(
            {"position"},
            {actuator.tag for actuator in simplified_mjcf.findall("./actuator/*")},
        )
        self.assertTrue(
            all(
                float(actuator.get("kp"))
                == mjx_model_simplification.DEFAULT_POSITION_KP
                for actuator in simplified_mjcf.findall("./actuator/*")
            )
        )

        model = mujoco.MjModel.from_xml_string(
            mujoco_parsing.to_string(simplified_mjcf)
        )
        summary = mjx_model_simplification.summarize_mujoco_model(model)
        self.assertEqual(31, summary.bodies)
        self.assertEqual(32, summary.joints)
        self.assertEqual(38, summary.qpos)
        self.assertEqual(37, summary.qvel)
        self.assertEqual(10, summary.actuators)
        self.assertEqual(2, summary.tendons)
        self.assertEqual(45, summary.geoms)
        self.assertEqual(0, summary.visual_geoms)
        self.assertEqual(45, summary.contact_geoms)
        self.assertEqual(0, summary.mesh_assets)

        mjx_model = mjx.put_model(model)
        mjx_data = mjx.make_data(mjx_model)
        self.assertEqual((38,), mjx_data.qpos.shape)
        self.assertEqual((37,), mjx_data.qvel.shape)

    def test_trex_complexity_comparison_reports_expected_reduction(self):
        urdf = urdf_parsing.Urdf.from_element(
            urdf_parsing.read_root_node_from_urdf(str(_ASSET_DIR / "trex.urdf"))
        )
        comparison = mjx_model_simplification.compare_full_and_simplified(
            urdf, asset_dir=_ASSET_DIR
        )

        self.assertEqual(134, comparison.full.bodies)
        self.assertEqual(31, comparison.simplified.bodies)
        self.assertEqual(297, comparison.full.geoms)
        self.assertEqual(45, comparison.simplified.geoms)
        self.assertEqual(252, comparison.full.visual_geoms)
        self.assertEqual(0, comparison.simplified.visual_geoms)
        self.assertEqual(45, comparison.full.contact_geoms)
        self.assertEqual(45, comparison.simplified.contact_geoms)
        self.assertEqual(252, comparison.full.mesh_assets)
        self.assertEqual(0, comparison.simplified.mesh_assets)
        self.assertAlmostEqual(
            comparison.full.total_mass, comparison.simplified.total_mass
        )

        table = mjx_model_simplification.comparison_markdown(comparison)
        self.assertIn("| Bodies | 134 | 31 |", table)
        self.assertIn("| Mesh assets | 252 | 0 |", table)


if __name__ == "__main__":
    unittest.main()
