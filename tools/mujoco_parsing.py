# Provides basic parsing methods for reading a Mujoco file.

import dataclasses
import numpy as np

from . import geometry
from . import urdf_parsing
from xml.etree import ElementTree


def mujoco_preamble(name: str) -> ElementTree.Element:
    mujoco = ElementTree.Element(name, {"model": name})
    option = ElementTree.Element("option", {"timestep": "0.001"})
    compiler = ElementTree.Element(
        "compiler", {"coordinate": "local", "angle": "radian"}
    )
    mujoco.extend([option, compiler])
    return mujoco


def urdf_to_body(
    link: urdf_parsing.UrdfLink,
    joint: urdf_parsing.UrdfJoint | None,
    world_body: bool = False,
) -> ElementTree.Element:
    body = ElementTree.Element("worldbody" if world_body else "body")
    body["name"] = link.name
    if joint:
        to_pos_axisangle(body, joint.origin)
        body["pos"] = to_vec3(joint.origin.translation)
        body["axisangle"] = joint.origin.rotation.as_rotvec()
        maybe_append(body, urdf_to_joint(joint))
    for s, shape in enumerate(link.collision_shapes):
        maybe_append(body, urdf_to_geom(shape, f"{link.name}_{s:02d}"))
    return body


def urdf_to_joint(joint: urdf_parsing.UrdfJoint) -> ElementTree.Element | None:
    if joint.type.lower() in ["revolute", "continuous"]:
        node = ElementTree.Element("joint", {"type": "hinge"})
        node["name"] = joint.name
        node["axis"] = to_vec3(joint.axis)
        node["range"] = to_vec2(joint.limits.position)
        return node
    return None


def urdf_to_geom(
    shape: geometry.Geometry, name: str
) -> ElementTree.Element | None:
    if isinstance(shape, geometry.GeometryCapsule):
        pass


def maybe_append(
    parent: ElementTree.Element, child: ElementTree.Element | None
):
    if child:
        parent.append(child)


def to_pos_axisangle(node: ElementTree.Element, origin: geometry.Transform):
    node["pos"] = to_vec3(origin.translation)
    rot_vector = origin.rotation.as_rotvec()
    angle = np.linalg.norm(rot_vector)
    axis = rot_vector / angle if angle > 0 else np.array([0, 0, 1])
    node["axisangle"] = f"{to_vec3(axis)} {angle}"


def to_vec2(vec: np.ndarray) -> str:
    return f"{vec[0]} {vec[1]}"


def to_vec3(vec: np.ndarray) -> str:
    return f"{vec[0]} {vec[1]} {vec[2]}"
