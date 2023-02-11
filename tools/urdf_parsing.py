# Provides basic parsing methods for loading a URDF.
# Warning! The parsing methods are incomplete and are only really expected to
# work with the URDFs found in the assets.
# TODO:
#  * Error handling when parsing the URDF, current behavior is "silent on
#    missing fields" and makes some dubious assumptions.
#  * Parsing of geometry fields only looks for `mesh` and assumes no
#    composition.
#  * Parsing of joints really only covers revolute joints.
#  * Parsing of limits and special joint fields are largely ignored.

import dataclasses
import numpy as np

from collections import defaultdict
from scipy.spatial import transform
from xml.etree import ElementTree


@dataclasses.dataclass
class Frame:
    translation: np.ndarray = dataclasses.field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0])
    )
    rotation: transform.Rotation = dataclasses.field(
        default_factory=transform.Rotation.identity
    )


def _inf_bounds():
    return np.array([-np.inf, np.inf])


@dataclasses.dataclass
class MotionLimits:
    position: np.ndarray = dataclasses.field(default_factory=_inf_bounds)
    velocity: np.ndarray = dataclasses.field(default_factory=_inf_bounds)
    acceleration: np.ndarray = dataclasses.field(default_factory=_inf_bounds)


@dataclasses.dataclass
class Geometry:
    _: dataclasses.KW_ONLY
    origin: Frame = Frame()


@dataclasses.dataclass
class GeometryMesh(Geometry):
    filename: str


@dataclasses.dataclass
class GeometrySphere(Geometry):
    radius: float


@dataclasses.dataclass
class GeometryCapsule(Geometry):
    radius: float
    length: float


@dataclasses.dataclass
class UrdfJoint:
    name: str
    parent_name: str
    child_name: str
    axis: np.ndarray = dataclasses.field(
        default_factory=lambda: np.array([0.0, 0.0, 1.0])
    )
    origin: Frame = dataclasses.field(default_factory=Frame)
    type: str = "fixed"
    limits: MotionLimits = dataclasses.field(default_factory=MotionLimits)


@dataclasses.dataclass
class UrdfInertial:
    mass: float = 0.0
    origin: Frame = dataclasses.field(default_factory=Frame)
    inertia: np.ndarray = dataclasses.field(default_factory=lambda:np.eye(3))


@dataclasses.dataclass
class UrdfLink:
    name: str
    inertia: UrdfInertial = dataclasses.field(default_factory=UrdfInertial)
    visual_shapes: list[Geometry] = dataclasses.field(default_factory=list)
    collision_shapes: list[Geometry] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Urdf:
    name: str
    joints: dict[str, UrdfJoint] = dataclasses.field(default_factory=dict)
    links: dict[str, UrdfLink] = dataclasses.field(default_factory=dict)


def read_root_node_from_urdf(urdf_path: str) -> ElementTree.Element:
    with open(urdf_path, "r") as f:
        urdf_string = f.read()
    return ElementTree.fromstring(urdf_string)


def get_link_map(
    joint_nodes: list[ElementTree.Element], link_reference: str
) -> dict[str, list[ElementTree.Element]]:
    """Creates a look-up table for link elements.

    Used to find referenced links from a list of `joint` elements. Assumes
    that the joint nodes given have a named link. Requires separate runs for
    parent or child relationships.

    Args:
        joint_nodes (list[ElementTree.Element]): the set of `joint` elements to
        find corresponding links.
        link_reference (str): either 'parent' or 'child'.

    Returns:
        dict[str, list[ElementTree.Element]]: maps a link name to the
        associated `joint` elements.
    """
    link_map = defaultdict(list)
    for joint_node in joint_nodes:
        link_name = joint_node.find(link_reference).get("link")
        link_map[link_name].append(joint_node)
    return link_map


def get_joint_chains(
    parent_link_map: dict[str, ElementTree.Element],
    link_name: str,
    list_of_chains: list[list[ElementTree.Element]] | None = None,
) -> list[list[ElementTree.Element]]:
    """Finds all joint-chains that connect to the specified link.

    This is primarily for finding all descendent chains from a specific root
    link. The joint chains will be ordered [parent, child] and should find all
    branches that connect to the initially seeded link_name.

    Args:
        parent_link_map (dict[str, ElementTree.Element]): a map whose key is
        the parent_link name and value is the set of joints that share this
        link.
        link_name (str): a link name to find connected joints.
        list_of_chains (list[list[ElementTree.Element]], optional):
        a list of joint-chains where each chain is ordered and ends with a
        joint whose parent is link_name. Defaults to None.

    Returns:
        list[list[ElementTree.Element]]: a list of joint-chains that extend the
        provided list_of_chains and all available paths that are descendent from
        link_name.
    """
    old_chains = list_of_chains if list_of_chains else [[]]
    new_chains = []
    if link_name in parent_link_map:
        for joint_node in parent_link_map[link_name]:
            new_chains.extend(
                get_joint_chains(
                    parent_link_map,
                    joint_node.find("child").get("link"),
                    [oc.append(joint_node) for oc in old_chains],
                )
            )
        return new_chains
    else:
        return old_chains


def get_all_chains(
    urdf_root: ElementTree.Element,
) -> list[list[ElementTree.Element]]:
    joint_nodes = urdf_root.findall("joint")

    child_link_map = get_link_map(joint_nodes, "child")
    parent_link_map = get_link_map(joint_nodes, "parent")

    parent_link_set = set([v for v in parent_link_map])
    child_link_set = set([v for v in child_link_map])

    root_links = list(parent_link_set.difference(child_link_set))
    joint_chains = []
    for root_link in root_links:
        for joint_chain in get_joint_chains(parent_link_map, root_link):
            joint_chains.append(joint_chain)

    return joint_chains


def get_chain(urdf_root, root_link_name, tip_link_name):
    joint_nodes = urdf_root.findall("joint")
    parent_link_map = get_link_map(joint_nodes, "parent")
    joint_chains = get_joint_chains(parent_link_map, root_link_name)
    for chain in joint_chains:
        maybe_chain = []
        for joint in chain:
            link_name = joint.find("child").get("link")
            maybe_chain.append(joint)
            if link_name == tip_link_name:
                return maybe_chain
    return None


def lookup_float(
    node: ElementTree.Element | None, key: str, default=0.0
) -> float:
    if node:
        value = node.get("key")
        return default if value is None else float(value)
    return default


def lookup_str(node: ElementTree.Element | None, key: str, default="") -> str:
    if node:
        return node.get("key") or ""
    return default


def convert_vec3(vec_string: str) -> np.ndarray:
    return np.array([float(x) for x in vec_string.split(" ") if x])[:3]


def convert_rpy(vec_string: str) -> transform.Rotation:
    rpy = convert_vec3(vec_string)
    return transform.Rotation.from_euler("xyz", rpy)


def convert_origin(node: ElementTree.Element) -> Frame:
    origin = node.find("origin")
    if origin is not None:
        xyz = convert_vec3(origin.get("xyz"))
        rpy = convert_rpy(origin.get("rpy"))
        return Frame(translation=xyz, rotation=rpy)
    else:
        return Frame()


def convert_inertia(node: ElementTree.Element) -> np.ndarray:
    output = np.eye(3)
    inertia = node.find("inertia")
    if inertia is not None:
        output[0, 0] = lookup_float(inertia, "ixx")
        output[0, 1] = lookup_float(inertia, "ixy")
        output[0, 2] = lookup_float(inertia, "ixz")
        output[1, 1] = lookup_float(inertia, "iyy")
        output[1, 2] = lookup_float(inertia, "iyz")
        output[2, 2] = lookup_float(inertia, "izz")
        for r in range(3 - 1):
            for c in range(r + 1, 3):
                output[c, r] = output[r, c]

    return output


def convert_inertial(node: ElementTree.Element) -> UrdfInertial:
    inertial = node.find("inertial")
    if inertial is not None:
        origin = convert_origin(inertial)
        mass = lookup_float(inertial, "mass")
        inertia = convert_inertia(inertial)
        return UrdfInertial(mass=mass, origin=origin, inertia=inertia)
    else:
        return UrdfInertial(0.0)


def convert_shape(node: ElementTree.Element) -> Geometry:
    origin = convert_origin(node)
    geometry = node.find("geometry")
    if geometry is not None:
        mesh = geometry.find("mesh")
        if mesh is not None:
            return GeometryMesh(filename=mesh.get("filename"), origin=origin)
    else:
        return Geometry()


def convert_link(node: ElementTree.Element) -> UrdfLink:
    return UrdfLink(
        name=node.find("name"),
        inertia=convert_inertial(node),
        visual_shapes=[convert_shape(e) for e in node.findall("visual")],
        collision_shapes=[convert_shape(e) for e in node.findall("collision")],
    )


def convert_axis(node: ElementTree.Element) -> np.ndarray:
    axis = node.find("axis")
    if axis is not None:
        return convert_vec3(axis.get("xyz"))
    else:
        return np.array([0.0, 0.0, 1.0])


def convert_limit(node: ElementTree.Element) -> MotionLimits:
    limit = node.find("limit")
    motion_limits = MotionLimits()
    if limit is not None:
        motion_limits.position[0] = lookup_float(limit, "lower", -np.inf)
        motion_limits.position[1] = lookup_float(limit, "upper", np.inf)
    return motion_limits


def convert_joint(node: ElementTree.Element) -> UrdfJoint:
    return UrdfJoint(
        name=node.get("name"),
        parent_name=lookup_str(node.find("parent"), "link"),
        child_name=lookup_str(node.find("child"), "link"),
        axis=convert_axis(node),
        origin=convert_origin(node),
        type=node.get("type"),
        limits=convert_limit(node),
    )


def convert_urdf(node: ElementTree.Element) -> Urdf:
    return Urdf(
        name=node.get("name"),
        joints={
            j.get("name"): convert_joint(j) for j in node.findall("joint")
        },
        links={k.get("name"): convert_link(k) for k in node.findall("link")},
    )
