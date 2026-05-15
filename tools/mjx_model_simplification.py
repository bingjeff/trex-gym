"""Helpers for generating a simplified MJX training model."""

from __future__ import annotations

import copy
import dataclasses
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree

import mujoco
import numpy as np

from . import geometry
from . import mujoco_parsing
from . import urdf_parsing

DEFAULT_POSITION_KP = 35.0
DEFAULT_PASSIVE_STIFFNESS_PER_ROW_SUM = 1000.0
DEFAULT_PASSIVE_DAMPING_PER_ROW_SUM = 80.0
DEFAULT_ARMATURE_PER_ROW_SUM = 0.2


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
class ModelComparison:
    full: ModelSummary
    simplified: ModelSummary

    def ratio(self, field: str) -> float:
        full_value = getattr(self.full, field)
        simplified_value = getattr(self.simplified, field)
        if full_value == 0:
            return np.inf
        return simplified_value / full_value


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


def configure_ground_only_contacts(mujoco_node: ElementTree.Element) -> None:
    """Configures the training MJCF so only floor-vs-body contacts are valid."""
    contact_default = mujoco_node.find("./default/default[@class='contact']/geom")
    if contact_default is not None:
        contact_default.set("contype", "0")
        contact_default.set("conaffinity", "1")
    for geom in mujoco_node.findall(".//geom"):
        if geom.get("name") == "floor":
            geom.set("contype", "1")
            geom.set("conaffinity", "0")
        elif geom.get("class") == "contact" or geom.get("group") == "2":
            geom.set("contype", "0")
            geom.set("conaffinity", "1")


def apply_mass_scaled_joint_tuning(
    mujoco_node: ElementTree.Element,
    stiffness_per_row_sum: float = DEFAULT_PASSIVE_STIFFNESS_PER_ROW_SUM,
    damping_per_row_sum: float = DEFAULT_PASSIVE_DAMPING_PER_ROW_SUM,
    armature_per_row_sum: float = DEFAULT_ARMATURE_PER_ROW_SUM,
    actuator_kp_per_row_sum: float = DEFAULT_PASSIVE_STIFFNESS_PER_ROW_SUM,
    actuated_joint_stiffness_scale: float = 1.0,
    joint_stiffness_scale_overrides: dict[str, float] | None = None,
    actuator_kp_scale_overrides: dict[str, float] | None = None,
) -> dict[str, float]:
    """Applies mass-matrix-row-sum-scaled passive and actuator gains.

    The scale for each hinge DOF is `sum(abs(M[dof, :]))` at the zero/root
    identity configuration. Tendon actuator scales are the coefficient-weighted
    sum of the tendon joints' DOF scales.
    """
    model = load_mujoco_from_xml_element(mujoco_node)
    scales = mass_matrix_row_sum_scales(model)
    joint_scales = _hinge_joint_scales(model, scales)
    tendon_scales = _fixed_tendon_scales(mujoco_node, joint_scales)
    actuated_joints = _actuated_joint_names(mujoco_node)
    joint_stiffness_scale_overrides = joint_stiffness_scale_overrides or {}
    actuator_kp_scale_overrides = actuator_kp_scale_overrides or {}

    for joint in mujoco_node.findall(".//joint"):
        name = joint.get("name")
        if name not in joint_scales:
            continue
        scale = joint_scales[name]
        if name in joint_stiffness_scale_overrides:
            stiffness_scale = joint_stiffness_scale_overrides[name]
        elif name in actuated_joints:
            stiffness_scale = actuated_joint_stiffness_scale
        else:
            stiffness_scale = 1.0
        joint.set(
            "stiffness", _format_float(stiffness_per_row_sum * scale * stiffness_scale)
        )
        joint.set("damping", _format_float(damping_per_row_sum * scale))
        joint.set("armature", _format_float(armature_per_row_sum * scale))

    actuator = mujoco_node.find("actuator")
    if actuator is not None:
        for position in actuator.findall("position"):
            joint_name = position.get("joint")
            tendon_name = position.get("tendon")
            if joint_name in joint_scales:
                scale = joint_scales[joint_name]
                kp_scale = actuator_kp_scale_overrides.get(joint_name, 1.0)
            elif tendon_name in tendon_scales:
                scale = tendon_scales[tendon_name]
                kp_scale = actuator_kp_scale_overrides.get(tendon_name, 1.0)
            else:
                continue
            position.set(
                "kp", _format_float(actuator_kp_per_row_sum * scale * kp_scale)
            )
    return joint_scales


def _actuated_joint_names(mujoco_node: ElementTree.Element) -> set[str]:
    """Returns joints directly or tendon-actuated by generated actuators."""
    names = set()
    actuator = mujoco_node.find("actuator")
    if actuator is None:
        return names
    tendon_joints: dict[str, set[str]] = defaultdict(set)
    for tendon in mujoco_node.findall("./tendon/fixed"):
        tendon_name = tendon.get("name")
        if tendon_name is None:
            continue
        for joint in tendon.findall("joint"):
            joint_name = joint.get("joint")
            if joint_name is not None:
                tendon_joints[tendon_name].add(joint_name)
    for position in actuator.findall("position"):
        joint_name = position.get("joint")
        if joint_name is not None:
            names.add(joint_name)
        tendon_name = position.get("tendon")
        if tendon_name is not None:
            names.update(tendon_joints.get(tendon_name, set()))
    return names


def mass_matrix_row_sum_scales(model: mujoco.MjModel) -> np.ndarray:
    data = mujoco.MjData(model)
    data.qpos[:] = _zero_qpos(model)
    mujoco.mj_forward(model, data)
    mass_matrix = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, mass_matrix, data.qM)
    if not np.allclose(mass_matrix, mass_matrix.T):
        raise ValueError("Mass matrix is not symmetric.")
    return np.sum(np.abs(mass_matrix), axis=1)


def compare_full_and_simplified(
    urdf: urdf_parsing.Urdf,
    asset_dir: Path,
    position_kp: float = DEFAULT_POSITION_KP,
) -> ModelComparison:
    """Loads full and simplified MuJoCo models and returns their summaries."""
    full_mujoco = mujoco_parsing.urdf_to_mujoco(urdf)
    simplified_mujoco = urdf_to_mjx_mujoco(urdf, position_kp=position_kp)
    return ModelComparison(
        full=summarize_mujoco_model(
            load_mujoco_from_xml_element(full_mujoco, asset_dir)
        ),
        simplified=summarize_mujoco_model(
            load_mujoco_from_xml_element(simplified_mujoco, asset_dir)
        ),
    )


def load_mujoco_from_xml_element(
    node: ElementTree.Element, asset_dir: Path | None = None
) -> mujoco.MjModel:
    """Loads an MJCF element with assets resolved relative to `asset_dir`."""
    if asset_dir is None:
        return mujoco.MjModel.from_xml_string(mujoco_parsing.to_string(node))
    return mujoco.MjModel.from_xml_string(
        mujoco_parsing.to_string(node),
        assets=_asset_bytes(asset_dir),
    )


def comparison_markdown(comparison: ModelComparison) -> str:
    """Formats a model comparison as a Markdown table."""
    rows = [
        ("Bodies", "bodies"),
        ("Joints", "joints"),
        ("qpos", "qpos"),
        ("qvel", "qvel"),
        ("Actuators", "actuators"),
        ("Tendons", "tendons"),
        ("Geoms", "geoms"),
        ("Visual geoms", "visual_geoms"),
        ("Contact geoms", "contact_geoms"),
        ("Mesh assets", "mesh_assets"),
        ("Sensors", "sensors"),
        ("Total mass", "total_mass"),
    ]
    output = [
        "| Metric | Full | Simplified | Simplified / Full |",
        "|---|---:|---:|---:|",
    ]
    for label, field in rows:
        full_value = getattr(comparison.full, field)
        simplified_value = getattr(comparison.simplified, field)
        output.append(
            "| "
            f"{label} | {_format_metric(full_value)} | "
            f"{_format_metric(simplified_value)} | "
            f"{comparison.ratio(field):.3g} |"
        )
    output.append(
        "| Center of mass | "
        f"{_format_vec3(comparison.full.center_of_mass)} | "
        f"{_format_vec3(comparison.simplified.center_of_mass)} | |"
    )
    return "\n".join(output)


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
    joint_ranges = _joint_control_ranges(mujoco)
    tendon_ranges = _tendon_control_ranges(mujoco, joint_ranges)
    for motor in actuator.findall("motor"):
        motor.tag = "position"
        motor.attrib.pop("gear", None)
        joint_name = motor.get("joint")
        tendon_name = motor.get("tendon")
        if joint_name in joint_ranges:
            motor.set("ctrlrange", _format_vec(joint_ranges[joint_name]))
        elif tendon_name in tendon_ranges:
            motor.set("ctrlrange", _format_vec(tendon_ranges[tendon_name]))
        motor.set("kp", _format_float(position_kp))


def _joint_control_ranges(mujoco: ElementTree.Element) -> dict[str, np.ndarray]:
    ranges = {}
    for joint in mujoco.findall(".//joint"):
        name = joint.get("name")
        range_text = joint.get("range")
        if name is None or range_text is None:
            continue
        values = np.fromstring(range_text, sep=" ")
        if values.shape == (2,) and np.all(np.isfinite(values)):
            ranges[name] = values
    return ranges


def _tendon_control_ranges(
    mujoco: ElementTree.Element, joint_ranges: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    ranges = {}
    for tendon in mujoco.findall("./tendon/fixed"):
        tendon_name = tendon.get("name")
        if tendon_name is None:
            continue
        lower = 0.0
        upper = 0.0
        complete = False
        for joint in tendon.findall("joint"):
            joint_name = joint.get("joint")
            if joint_name not in joint_ranges:
                complete = False
                break
            complete = True
            coef = float(joint.get("coef", "1.0"))
            joint_range = joint_ranges[joint_name]
            values = coef * joint_range
            lower += float(np.min(values))
            upper += float(np.max(values))
        if complete:
            ranges[tendon_name] = np.array([lower, upper])
    return ranges


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


def _zero_qpos(model: mujoco.MjModel) -> np.ndarray:
    qpos = np.zeros(model.nq)
    if model.nq >= 7:
        qpos[3] = 1.0
    return qpos


def _hinge_joint_scales(model: mujoco.MjModel, scales: np.ndarray) -> dict[str, float]:
    output = {}
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        name = model.joint(joint_id).name
        output[name] = float(scales[model.jnt_dofadr[joint_id]])
    return output


def _fixed_tendon_scales(
    mujoco_node: ElementTree.Element,
    joint_scales: dict[str, float],
) -> dict[str, float]:
    output = {}
    for tendon in mujoco_node.findall("./tendon/fixed"):
        scale = 0.0
        for joint in tendon.findall("joint"):
            joint_name = joint.get("joint")
            coef = float(joint.get("coef", "1.0"))
            scale += abs(coef) * joint_scales[joint_name]
        output[tendon.get("name")] = scale
    return output


def _asset_bytes(asset_dir: Path) -> dict[str, bytes]:
    assets = {}
    for path in asset_dir.rglob("*"):
        if path.is_file():
            assets[path.relative_to(asset_dir).as_posix()] = path.read_bytes()
    return assets


def _format_metric(value: float | int) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _format_vec3(vec: np.ndarray) -> str:
    return " ".join(f"{value:.6g}" for value in vec)


def _format_vec(vec: np.ndarray) -> str:
    return " ".join(_format_float(float(value)) for value in vec)


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
