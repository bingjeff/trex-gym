import unittest
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from tools import mujoco_parsing
from tools import generate_collision_capsules
from tools import geometry
from tools import urdf_parsing

_NP_TOLERANCE = 1.0e-10
_ASSET_DIR = Path(__file__).resolve().parents[1] / "assets"


def _test_joint_positions(urdf: urdf_parsing.Urdf) -> dict[str, float]:
    output = {}
    for index, joint in enumerate(urdf.joints.values()):
        if joint.type not in ("revolute", "continuous"):
            continue
        lower, upper = joint.limits.position
        if np.isfinite(lower) and np.isfinite(upper):
            midpoint = 0.5 * (lower + upper)
            half_range = 0.5 * (upper - lower)
            output[joint.name] = midpoint + 0.25 * half_range * np.sin(index)
        else:
            output[joint.name] = 0.1 * np.sin(index)
    return output


class TestMujocoParsing(unittest.TestCase):
    def test_trex_urdf_to_mujoco_kinematics_round_trip(self):
        urdf = urdf_parsing.Urdf.from_element(
            urdf_parsing.read_root_node_from_urdf(str(_ASSET_DIR / "trex.urdf"))
        )
        collision_shapes = [
            shape for link in urdf.links.values() for shape in link.collision_shapes
        ]
        self.assertEqual(
            len(generate_collision_capsules.CAPSULE_FIT_SPECS),
            len(collision_shapes),
        )
        self.assertTrue(
            all(
                isinstance(shape, geometry.GeometryCapsule)
                for shape in collision_shapes
            )
        )
        self.assertEqual(
            3, len(urdf.links["link_tarsometatarsus_right"].collision_shapes)
        )
        self.assertEqual(
            3, len(urdf.links["link_tarsometatarsus_left"].collision_shapes)
        )
        self.assertEqual(0, len(urdf.links["link_vertebrae_sacral"].collision_shapes))
        self.assertEqual(1, len(urdf.links["link_femur_right"].collision_shapes))
        self.assertEqual(1, len(urdf.links["link_femur_left"].collision_shapes))
        linked_dof_links = set()
        for side in ("right", "left"):
            adduction = urdf.joints[f"joint_hip_adduction_{side}"]
            flexion = urdf.joints[f"joint_femur_{side}"]
            self.assertIn(f"link_hip_adduction_{side}", urdf.links)
            self.assertEqual("link_vertebrae_sacral", adduction.parent_name)
            self.assertEqual(f"link_hip_adduction_{side}", adduction.child_name)
            self.assertEqual(f"link_femur_{side}", adduction.linked_dof_body)
            np.testing.assert_allclose([1.0, 0.0, 0.0], adduction.axis)
            np.testing.assert_allclose(
                [-0.7853981633974483, 0.7853981633974483],
                adduction.limits.position,
            )
            self.assertEqual(f"link_hip_adduction_{side}", flexion.parent_name)
            self.assertEqual(f"link_femur_{side}", flexion.child_name)
            np.testing.assert_allclose([0.0, 0.0, 1.0], flexion.axis)
            linked_dof_links.add(adduction.child_name)
        mujoco_xml = mujoco_parsing.to_string(mujoco_parsing.urdf_to_mujoco(urdf))
        mujoco_node = ElementTree.fromstring(mujoco_xml)
        mujoco = mujoco_parsing.parse_mujoco(mujoco_node)
        joint_positions = _test_joint_positions(urdf)

        urdf_poses = mujoco_parsing.urdf_forward_kinematics(urdf, joint_positions)
        mujoco_poses = mujoco_parsing.mujoco_forward_kinematics(mujoco, joint_positions)

        self.assertEqual(["world"], urdf.root_link_names)
        self.assertEqual(
            set(urdf.links) - {"world"} - linked_dof_links, set(mujoco_poses)
        )
        self.assertNotIn("world", mujoco.body_map)
        self.assertTrue(linked_dof_links.isdisjoint(mujoco.body_map))
        for side in ("right", "left"):
            femur_joints = mujoco.body_map[f"link_femur_{side}"].joints
            self.assertEqual(
                [f"joint_hip_adduction_{side}", f"joint_femur_{side}"],
                [joint.name for joint in femur_joints],
            )
            np.testing.assert_allclose([1.0, 0.0, 0.0], femur_joints[0].axis)
            np.testing.assert_allclose(
                [-0.7853981633974483, 0.7853981633974483],
                femur_joints[0].limits.position,
            )
            np.testing.assert_allclose([0.0, 0.0, 1.0], femur_joints[1].axis)
        self.assertEqual(
            "joint_world_to_sacrum",
            mujoco_node.find("./worldbody/body/freejoint").get("name"),
        )
        self.assertEqual("free", mujoco.body_map["link_vertebrae_sacral"].joint.type)
        sacrum = urdf.links["link_vertebrae_sacral"]
        sacrum_inertial = mujoco_node.find(
            "./worldbody/body[@name='link_vertebrae_sacral']/inertial"
        )
        self.assertIsNotNone(sacrum_inertial)
        self.assertAlmostEqual(sacrum.inertia.mass, float(sacrum_inertial.get("mass")))
        np.testing.assert_allclose(
            sacrum.inertia.origin.translation,
            np.fromstring(sacrum_inertial.get("pos"), sep=" "),
            atol=_NP_TOLERANCE,
        )
        np.testing.assert_allclose(
            [
                sacrum.inertia.inertia[0, 0],
                sacrum.inertia.inertia[1, 1],
                sacrum.inertia.inertia[2, 2],
            ],
            np.fromstring(sacrum_inertial.get("diaginertia"), sep=" "),
            atol=_NP_TOLERANCE,
        )
        visual_default = mujoco_node.find("./default/default[@class='visual']/geom")
        self.assertIsNotNone(visual_default)
        self.assertEqual("0", visual_default.get("contype"))
        self.assertEqual("0", visual_default.get("conaffinity"))
        self.assertEqual("1", visual_default.get("group"))
        contact_default = mujoco_node.find("./default/default[@class='contact']/geom")
        self.assertIsNotNone(contact_default)
        self.assertEqual("1", contact_default.get("contype"))
        self.assertEqual("1", contact_default.get("conaffinity"))
        self.assertEqual("2", contact_default.get("group"))
        self.assertEqual("0.5 0.8 1.0 0.5", contact_default.get("rgba"))
        self.assertTrue(
            all(
                geom.get("class") == "visual"
                for geom in mujoco_node.findall(".//geom")
                if "_visual_" in geom.get("name", "")
            )
        )
        collision_geoms = [
            geom
            for geom in mujoco_node.findall(".//geom")
            if geom.get("name", "").startswith("link_")
            and "_collision_" in geom.get("name", "")
        ]
        self.assertEqual(len(collision_shapes), len(collision_geoms))
        self.assertTrue(all(geom.get("class") == "contact" for geom in collision_geoms))
        self.assertTrue(all(geom.get("type") == "capsule" for geom in collision_geoms))
        self.assertTrue(all(geom.get("mesh") is None for geom in collision_geoms))

        for link_name, urdf_pose in urdf_poses.items():
            if link_name == "world" or link_name in linked_dof_links:
                continue
            mujoco_pose = mujoco_poses[link_name]
            np.testing.assert_allclose(
                urdf_pose.translation,
                mujoco_pose.translation,
                atol=_NP_TOLERANCE,
            )
            np.testing.assert_allclose(
                urdf_pose.rotation.as_matrix(),
                mujoco_pose.rotation.as_matrix(),
                atol=_NP_TOLERANCE,
            )

    def test_generated_capsules_enclose_source_mesh_vertices(self):
        fits = generate_collision_capsules.generate_capsule_fits(
            _ASSET_DIR / "trex.urdf"
        )
        self.assertEqual(len(generate_collision_capsules.CAPSULE_FIT_SPECS), len(fits))
        for fit in fits:
            self.assertLessEqual(fit.max_outside_distance, _NP_TOLERANCE)


if __name__ == "__main__":
    unittest.main()
