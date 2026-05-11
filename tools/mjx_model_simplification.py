"""Helpers for generating a simplified MJX training model."""

from __future__ import annotations

import copy
import dataclasses
from collections import defaultdict
from xml.etree import ElementTree

import mujoco
import numpy as np

from . import geometry
from . import mujoco_parsing
from . import urdf_parsing

DEFAULT_POSITION_KP = 35.0


@dataclasses.dataclass
class ModelSummary:
    bodies: int
    joints: int
    qpos: int
    qvel: int
    actuators: int
    tendons: int
    geoms: int
    visual_geoms: int
    contact_geoms: int
    mesh_assets: int
    sensors: int
    total_mass: float
    center_of_mass: np.ndarray


@dataclasses.dataclass
class _InertialComponent:
    mass: float
    com: np.ndarray
    inertia_at_com: np.ndarray


def urdf_to_mjx_mujoco(
    urdf: urdf_parsing.Urdf,
    position_kp: float = DEFAULT_POSITION_KP,
) -> ElementTree.Element:
    """Converts a URDF into the simplified MJCF used for MJX training."""
    simplified = simplify_urdf_for_mjx(urdf)
    mujoco = mujoco_parsing.urdf_to_mujoco(simplified)
    convert_motors_to_position_actuators(mujoco, position_kp)
    return mujoco


def simplify_urdf_for_mjx(urdf: urdf_parsing.Urdf) -> urdf_parsing.Urdf:
    """Removes visuals and fuses links connected by fixed joints.

    The resulting URDF keeps the original non-fixed joints, collision shapes,
    inertial properties, MuJoCo control metadata, and root floating joint.
    Collision shapes and inertials from fixed child links are moved into the
    nearest non-fixed ancestor link.
    """
    parent_to_joints = urdf.parent_link_name_to_joint
    links: dict[str, urdf_parsing.UrdfLink] = {}
    joints: dict[str, urdf_parsing.UrdfJoint] = {}
    inertials: dict[str, list[_InertialComponent]] = defaultdict(list)

    def ensure_link(name: str) -> urdf_parsing.UrdfLink:
        if name not in links:
            links[name] = urdf_parsing.UrdfLink(name=name)
        return links[name]

    def add_link_contents(
        source_link_name: str,
        target_link_name: str,
        source_to_target: geometry.Transform,
    ) -> None:
        source = urdf.links[source_link_name]
        target = ensure_link(target_link_name)
        component = _inertial_component(source.inertia, source_to_target)
        if component is not None:
            inertials[target_link_name].append(component)
        for shape in source.collision_shapes:
            target.collision_shapes.append(_transform_shape(shape, source_to_target))

    def visit(
        link_name: str,
        target_link_name: str,
        link_to_target: geometry.Transform,
    ) -> None:
        add_link_contents(link_name, target_link_name, link_to_target)
        for joint in parent_to_joints.get(link_name, []):
            if joint.type == "fixed":
                visit(
                    joint.child_name,
                    target_link_name,
                    link_to_target * joint.origin,
                )
                continue
            new_joint = copy.deepcopy(joint)
            new_joint.parent_name = target_link_name
            new_joint.origin = link_to_target * joint.origin
            joints[new_joint.name] = new_joint
            visit(joint.child_name, joint.child_name, geometry.Transform())

    for root_name in sorted(urdf.root_link_names):
        ensure_link(root_name)
        visit(root_name, root_name, geometry.Transform())

    for link_name, components in inertials.items():
        links[link_name].inertia = _merge_inertials(components)

    return urdf_parsing.Urdf(
        name=urdf.name,
        joints=joints,
        links=links,
        mujoco=copy.deepcopy(urdf.mujoco),
    )


def convert_motors_to_position_actuators(
    mujoco: ElementTree.Element, position_kp: float = DEFAULT_POSITION_KP
) -> None:
    """Converts emitted motor shortcuts into position actuator shortcuts."""
    actuator = mujoco.find("actuator")
    if actuator is None:
        return
    for motor in actuator.findall("motor"):
        motor.tag = "position"
        motor.attrib.pop("gear", None)
        motor.set("kp", _format_float(position_kp))


def summarize_mujoco_model(model) -> ModelSummary:
    """Returns basic complexity metrics for a loaded MuJoCo model."""
    body_masses = np.asarray(model.body_mass)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    body_ipos = np.asarray(data.xipos)
    total_mass = float(np.sum(body_masses))
    if total_mass > 0.0:
        center_of_mass = np.sum(body_ipos * body_masses[:, None], axis=0) / total_mass
    else:
        center_of_mass = np.zeros(3)
    geom_groups = np.asarray(model.geom_group)
    return ModelSummary(
        bodies=int(model.nbody),
        joints=int(model.njnt),
        qpos=int(model.nq),
        qvel=int(model.nv),
        actuators=int(model.nu),
        tendons=int(model.ntendon),
        geoms=int(model.ngeom),
        visual_geoms=int(np.sum(geom_groups == 1)),
        contact_geoms=int(np.sum(geom_groups == 2)),
        mesh_assets=int(model.nmesh),
        sensors=int(model.nsensor),
        total_mass=total_mass,
        center_of_mass=center_of_mass,
    )


def _transform_shape(
    shape: geometry.Geometry, source_to_target: geometry.Transform
) -> geometry.Geometry:
    origin = source_to_target * shape.origin
    if isinstance(shape, geometry.GeometryCapsule):
        return geometry.GeometryCapsule(
            radius=shape.radius, length=shape.length, origin=origin
        )
    if isinstance(shape, geometry.GeometrySphere):
        return geometry.GeometrySphere(radius=shape.radius, origin=origin)
    if isinstance(shape, geometry.GeometryBox):
        return geometry.GeometryBox(size_xyz=shape.size_xyz.copy(), origin=origin)
    if isinstance(shape, geometry.GeometryMesh):
        return geometry.GeometryMesh(filename=shape.filename, origin=origin)
    raise ValueError(f"Unknown geometry shape type: {type(shape)}")


def _inertial_component(
    inertial: urdf_parsing.UrdfInertial,
    source_to_target: geometry.Transform,
) -> _InertialComponent | None:
    if inertial.mass <= 0.0:
        return None
    origin = source_to_target * inertial.origin
    rotation = origin.rotation.as_matrix()
    return _InertialComponent(
        mass=inertial.mass,
        com=origin.translation,
        inertia_at_com=rotation @ inertial.inertia @ rotation.T,
    )


def _merge_inertials(
    components: list[_InertialComponent],
) -> urdf_parsing.UrdfInertial:
    if not components:
        return urdf_parsing.UrdfInertial()
    total_mass = sum(component.mass for component in components)
    if total_mass <= 0.0:
        return urdf_parsing.UrdfInertial()
    center_of_mass = (
        sum(component.mass * component.com for component in components) / total_mass
    )
    inertia = np.zeros((3, 3))
    for component in components:
        offset = component.com - center_of_mass
        inertia += component.inertia_at_com + component.mass * (
            np.dot(offset, offset) * np.eye(3) - np.outer(offset, offset)
        )
    return urdf_parsing.UrdfInertial(
        mass=total_mass,
        origin=geometry.Transform(translation=center_of_mass),
        inertia=inertia,
    )


def _format_float(value: float) -> str:
    return f"{value:.17g}"
