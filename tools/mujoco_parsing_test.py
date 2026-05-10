import unittest
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from tools import mujoco_parsing
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
        mujoco_xml = mujoco_parsing.to_string(mujoco_parsing.urdf_to_mujoco(urdf))
        mujoco_node = ElementTree.fromstring(mujoco_xml)
        mujoco = mujoco_parsing.parse_mujoco(mujoco_node)
        joint_positions = _test_joint_positions(urdf)

        urdf_poses = mujoco_parsing.urdf_forward_kinematics(urdf, joint_positions)
        mujoco_poses = mujoco_parsing.mujoco_forward_kinematics(mujoco, joint_positions)

        self.assertEqual(["world"], urdf.root_link_names)
        self.assertEqual(set(urdf.links) - {"world"}, set(mujoco_poses))
        self.assertNotIn("world", mujoco.body_map)
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
        self.assertTrue(
            all(
                geom.get("class") == "visual"
                for geom in mujoco_node.findall(".//geom")
                if "_visual_" in geom.get("name", "")
            )
        )

        for link_name, urdf_pose in urdf_poses.items():
            if link_name == "world":
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


if __name__ == "__main__":
    unittest.main()
