"""URDF to MuJoCo XML conversion helpers.

The converter keeps URDF link frames as MuJoCo body frames. A URDF joint is
represented as the pose of the child body relative to its parent body, with a
MuJoCo hinge joint added for revolute/continuous joints.
"""

import dataclasses
import re
from xml.dom import minidom
from xml.etree import ElementTree

import numpy as np
from scipy.spatial import transform

from . import geometry
from . import urdf_parsing


@dataclasses.dataclass
class MujocoJoint:
    name: str
    type: str
    axis: np.ndarray
    limits: geometry.MotionLimits = dataclasses.field(
        default_factory=geometry.MotionLimits
    )


@dataclasses.dataclass
class MujocoBody:
    name: str
    transform: geometry.Transform = dataclasses.field(
        default_factory=geometry.Transform
    )
    joint: MujocoJoint | None = None
    children: list["MujocoBody"] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class MujocoModel:
    name: str
    bodies: list[MujocoBody] = dataclasses.field(default_factory=list)

    @property
    def body_map(self) -> dict[str, MujocoBody]:
        output = {}

        def visit(body: MujocoBody):
            output[body.name] = body
            for child in body.children:
                visit(child)

        for body in self.bodies:
            visit(body)
        return output


def urdf_to_mujoco(urdf: urdf_parsing.Urdf) -> ElementTree.Element:
    """Converts a parsed URDF model into an MJCF XML element."""
    mujoco = ElementTree.Element("mujoco", {"model": urdf.name})
    mujoco.extend(
        [
            ElementTree.Element("compiler", {"angle": "radian", "coordinate": "local"}),
            ElementTree.Element("option", {"timestep": "0.001"}),
        ]
    )
    asset = _asset_node(urdf)
    if len(asset):
        mujoco.append(asset)
    worldbody = ElementTree.Element("worldbody")
    for root_name in sorted(urdf.root_link_names):
        worldbody.append(_link_to_body(root_name, urdf, None))
    mujoco.append(worldbody)
    return mujoco


def urdf_path_to_mujoco_xml(urdf_path: str) -> str:
    """Reads a URDF file and returns a pretty-printed MJCF XML string."""
    urdf = urdf_parsing.Urdf.from_element(
        urdf_parsing.read_root_node_from_urdf(urdf_path)
    )
    return to_string(urdf_to_mujoco(urdf))


def to_string(node: ElementTree.Element) -> str:
    return minidom.parseString(
        ElementTree.tostring(node, encoding="utf-8")
    ).toprettyxml(indent="  ")


def parse_mujoco(node: ElementTree.Element) -> MujocoModel:
    """Parses the MJCF subset emitted by this module."""
    worldbody = node.find("worldbody")
    bodies = []
    if worldbody is not None:
        bodies = [_body_from_element(body) for body in worldbody.findall("body")]
    return MujocoModel(name=node.get("model", ""), bodies=bodies)


def urdf_forward_kinematics(
    urdf: urdf_parsing.Urdf, joint_positions: dict[str, float] | None = None
) -> dict[str, geometry.Transform]:
    """Computes link poses from URDF kinematics."""
    joint_positions = joint_positions or {}
    poses = {}
    child_joints = urdf.parent_link_name_to_joint

    def visit(link_name: str, parent_pose: geometry.Transform):
        poses[link_name] = parent_pose
        for joint in child_joints.get(link_name, []):
            child_pose = parent_pose * joint.origin
            if joint.type in ("revolute", "continuous"):
                child_pose = child_pose * _axis_rotation(
                    joint.axis, joint_positions.get(joint.name, 0.0)
                )
            visit(joint.child_name, child_pose)

    for root_name in urdf.root_link_names:
        visit(root_name, geometry.Transform())
    return poses


def mujoco_forward_kinematics(
    model: MujocoModel, joint_positions: dict[str, float] | None = None
) -> dict[str, geometry.Transform]:
    """Computes body poses for the MJCF subset emitted by this module."""
    joint_positions = joint_positions or {}
    poses = {}

    def visit(body: MujocoBody, parent_pose: geometry.Transform):
        body_pose = parent_pose * body.transform
        if body.joint and body.joint.type == "hinge":
            body_pose = body_pose * _axis_rotation(
                body.joint.axis, joint_positions.get(body.joint.name, 0.0)
            )
        poses[body.name] = body_pose
        for child in body.children:
            visit(child, body_pose)

    for body in model.bodies:
        visit(body, geometry.Transform())
    return poses


def _link_to_body(
    link_name: str,
    urdf: urdf_parsing.Urdf,
    parent_joint: urdf_parsing.UrdfJoint | None,
) -> ElementTree.Element:
    link = urdf.links[link_name]
    body = ElementTree.Element("body", {"name": link.name})
    if parent_joint is not None:
        _set_transform_attributes(body, parent_joint.origin)
        _maybe_append(body, _joint_to_element(parent_joint))
    _maybe_append(body, _inertial_to_element(link.inertia))
    for shape_index, shape in enumerate(link.visual_shapes):
        _maybe_append(
            body,
            _shape_to_geom(
                shape=shape,
                name=f"{link.name}_visual_{shape_index:02d}",
                visual=True,
            ),
        )
    for shape_index, shape in enumerate(link.collision_shapes):
        _maybe_append(
            body,
            _shape_to_geom(
                shape=shape,
                name=f"{link.name}_collision_{shape_index:02d}",
                visual=False,
            ),
        )
    for child_joint in urdf.parent_link_name_to_joint.get(link_name, []):
        body.append(_link_to_body(child_joint.child_name, urdf, child_joint))
    return body


def _asset_node(urdf: urdf_parsing.Urdf) -> ElementTree.Element:
    asset = ElementTree.Element("asset")
    seen = set()
    for link in urdf.links.values():
        for shape in link.visual_shapes + link.collision_shapes:
            if not isinstance(shape, geometry.GeometryMesh):
                continue
            name = _mesh_name(shape.filename)
            if name in seen:
                continue
            seen.add(name)
            asset.append(
                ElementTree.Element("mesh", {"name": name, "file": shape.filename})
            )
    return asset


def _joint_to_element(
    joint: urdf_parsing.UrdfJoint,
) -> ElementTree.Element | None:
    if joint.type not in ("revolute", "continuous"):
        return None
    node = ElementTree.Element(
        "joint",
        {
            "name": joint.name,
            "type": "hinge",
            "axis": _to_vec3(joint.axis),
        },
    )
    if np.all(np.isfinite(joint.limits.position)):
        node.set("limited", "true")
        node.set("range", _to_vec2(joint.limits.position))
    return node


def _inertial_to_element(
    inertial: urdf_parsing.UrdfInertial,
) -> ElementTree.Element | None:
    if inertial.mass <= 0:
        return None
    node = ElementTree.Element(
        "inertial",
        {
            "mass": _format_float(inertial.mass),
            "pos": _to_vec3(inertial.origin.translation),
            "quat": _to_quat(inertial.origin.rotation),
            "fullinertia": " ".join(
                _format_float(v)
                for v in (
                    inertial.inertia[0, 0],
                    inertial.inertia[1, 1],
                    inertial.inertia[2, 2],
                    inertial.inertia[0, 1],
                    inertial.inertia[0, 2],
                    inertial.inertia[1, 2],
                )
            ),
        },
    )
    return node


def _shape_to_geom(
    shape: geometry.Geometry, name: str, visual: bool
) -> ElementTree.Element | None:
    if not isinstance(shape, geometry.GeometryMesh):
        return None
    attributes = {
        "name": name,
        "type": "mesh",
        "mesh": _mesh_name(shape.filename),
        "pos": _to_vec3(shape.origin.translation),
        "quat": _to_quat(shape.origin.rotation),
    }
    if visual:
        attributes.update({"contype": "0", "conaffinity": "0", "group": "1"})
    return ElementTree.Element("geom", attributes)


def _body_from_element(node: ElementTree.Element) -> MujocoBody:
    joints = [_joint_from_element(joint) for joint in node.findall("joint")]
    return MujocoBody(
        name=node.get("name", ""),
        transform=_transform_from_element(node),
        joint=joints[0] if joints else None,
        children=[_body_from_element(body) for body in node.findall("body")],
    )


def _joint_from_element(node: ElementTree.Element) -> MujocoJoint:
    limits = geometry.MotionLimits()
    if node.get("range"):
        limits.position = _from_vec(node.get("range"), 2)
    return MujocoJoint(
        name=node.get("name", ""),
        type=node.get("type", "hinge"),
        axis=_from_vec(node.get("axis", "0 0 1"), 3),
        limits=limits,
    )


def _transform_from_element(node: ElementTree.Element) -> geometry.Transform:
    return geometry.Transform(
        translation=_from_vec(node.get("pos", "0 0 0"), 3),
        rotation=_rotation_from_element(node),
    )


def _rotation_from_element(node: ElementTree.Element) -> transform.Rotation:
    if node.get("quat"):
        w, x, y, z = _from_vec(node.get("quat"), 4)
        return transform.Rotation.from_quat([x, y, z, w])
    if node.get("axisangle"):
        axis_angle = _from_vec(node.get("axisangle"), 4)
        return transform.Rotation.from_rotvec(axis_angle[:3] * axis_angle[3])
    if node.get("euler"):
        return transform.Rotation.from_euler("xyz", _from_vec(node.get("euler"), 3))
    return transform.Rotation.identity()


def _set_transform_attributes(node: ElementTree.Element, origin: geometry.Transform):
    if not np.allclose(origin.translation, 0.0):
        node.set("pos", _to_vec3(origin.translation))
    if not np.allclose(origin.rotation.as_rotvec(), 0.0):
        node.set("quat", _to_quat(origin.rotation))


def _axis_rotation(axis: np.ndarray, angle: float) -> geometry.Transform:
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm == 0.0:
        return geometry.Transform()
    return geometry.Transform(
        rotation=transform.Rotation.from_rotvec(axis / norm * angle)
    )


def _mesh_name(filename: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", filename).strip("_")


def _maybe_append(parent: ElementTree.Element, child: ElementTree.Element | None):
    if child is not None:
        parent.append(child)


def _from_vec(vec_string: str, length: int) -> np.ndarray:
    values = np.array([float(x) for x in vec_string.split()[:length]])
    if values.shape != (length,):
        raise ValueError(f"Expected {length} values, got {vec_string!r}.")
    return values


def _to_vec2(vec: np.ndarray) -> str:
    return f"{_format_float(vec[0])} {_format_float(vec[1])}"


def _to_vec3(vec: np.ndarray) -> str:
    return (
        f"{_format_float(vec[0])} "
        f"{_format_float(vec[1])} "
        f"{_format_float(vec[2])}"
    )


def _to_quat(rotation: transform.Rotation) -> str:
    x, y, z, w = rotation.as_quat()
    return " ".join(_format_float(v) for v in (w, x, y, z))


def _format_float(value: float) -> str:
    return f"{value:.17g}"
