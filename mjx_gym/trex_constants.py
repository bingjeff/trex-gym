"""Constants and MJCF generation helpers for T-Rex MJX environments."""

from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import mujoco
from scipy.spatial import transform

from tools import mjx_model_simplification
from tools import mujoco_parsing
from tools import urdf_parsing

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = PROJECT_ROOT / "assets"
URDF_PATH = ASSET_DIR / "trex.urdf"

ROOT_BODY = "link_vertebrae_sacral"
IMU_SITE = "imu"
GYRO_SENSOR = "gyro"
ACCELEROMETER_SENSOR = "accelerometer"
UPVECTOR_SENSOR = "upvector"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
UPRIGHT_ROOT_ROLL = np.pi / 2.0
UPRIGHT_GRAVITY = np.array([0.0, -1.0, 0.0])
STANDING_HIP_FLEXION = np.deg2rad(-30.0)
STANDING_KNEE_FLEXION = np.deg2rad(30.0)
STANDING_ANKLE_FLEXION = np.deg2rad(-75.0)
STANDING_FOOT_PENETRATION = 0.01

TAIL_ACTUATORS = (
    "actuator_tail_sagittal",
    "actuator_tail_mediolateral",
)

TAIL_TENDON_JOINTS = (
    "joint_vertebra_caudal_02",
    "joint_vertebra_caudal_03",
    "joint_vertebra_caudal_10",
    "joint_vertebra_caudal_11",
    "joint_vertebra_caudal_24",
    "joint_vertebra_caudal_25",
    "joint_vertebra_caudal_34",
)

JOINT_QPOS_DOF = 31
CTRL_DOF = 10

LEG_ACTUATORS = (
    "actuator_hip_adduction_right",
    "actuator_hip_adduction_left",
    "actuator_hip_flexion_right",
    "actuator_hip_flexion_left",
    "actuator_knee_right",
    "actuator_knee_left",
    "actuator_ankle_right",
    "actuator_ankle_left",
)

ACTION_ACTUATORS = LEG_ACTUATORS + TAIL_ACTUATORS


def load_urdf() -> urdf_parsing.Urdf:
    return urdf_parsing.Urdf.from_element(
        urdf_parsing.read_root_node_from_urdf(str(URDF_PATH))
    )


def trex_getup_mjcf(
    position_kp_per_row_sum: float = (
        mjx_model_simplification.DEFAULT_PASSIVE_STIFFNESS_PER_ROW_SUM
    ),
    passive_stiffness_per_row_sum: float = (
        mjx_model_simplification.DEFAULT_PASSIVE_STIFFNESS_PER_ROW_SUM
    ),
    passive_damping_per_row_sum: float = (
        mjx_model_simplification.DEFAULT_PASSIVE_DAMPING_PER_ROW_SUM
    ),
    armature_per_row_sum: float = mjx_model_simplification.DEFAULT_ARMATURE_PER_ROW_SUM,
    actuated_joint_stiffness_scale: float = 1.0,
    tail_joint_stiffness_scale: float = 1.0,
) -> ElementTree.Element:
    """Builds the simplified T-Rex getup scene MJCF."""
    node = mjx_model_simplification.urdf_to_mjx_mujoco(load_urdf())
    _add_scene(node)
    _add_torso_sites(node)
    _add_sensors(node)
    _add_keyframes(node)
    mjx_model_simplification.configure_ground_only_contacts(node)
    mjx_model_simplification.apply_mass_scaled_joint_tuning(
        node,
        stiffness_per_row_sum=passive_stiffness_per_row_sum,
        damping_per_row_sum=passive_damping_per_row_sum,
        armature_per_row_sum=armature_per_row_sum,
        actuator_kp_per_row_sum=position_kp_per_row_sum,
        actuated_joint_stiffness_scale=actuated_joint_stiffness_scale,
        joint_stiffness_scale_overrides={
            joint_name: tail_joint_stiffness_scale
            for joint_name in TAIL_TENDON_JOINTS
        },
    )
    return node


def trex_getup_xml(
    position_kp_per_row_sum: float = (
        mjx_model_simplification.DEFAULT_PASSIVE_STIFFNESS_PER_ROW_SUM
    ),
    passive_stiffness_per_row_sum: float = (
        mjx_model_simplification.DEFAULT_PASSIVE_STIFFNESS_PER_ROW_SUM
    ),
    passive_damping_per_row_sum: float = (
        mjx_model_simplification.DEFAULT_PASSIVE_DAMPING_PER_ROW_SUM
    ),
    armature_per_row_sum: float = mjx_model_simplification.DEFAULT_ARMATURE_PER_ROW_SUM,
    actuated_joint_stiffness_scale: float = 1.0,
    tail_joint_stiffness_scale: float = 1.0,
) -> str:
    return mujoco_parsing.to_string(
        trex_getup_mjcf(
            position_kp_per_row_sum=position_kp_per_row_sum,
            passive_stiffness_per_row_sum=passive_stiffness_per_row_sum,
            passive_damping_per_row_sum=passive_damping_per_row_sum,
            armature_per_row_sum=armature_per_row_sum,
            actuated_joint_stiffness_scale=actuated_joint_stiffness_scale,
            tail_joint_stiffness_scale=tail_joint_stiffness_scale,
        )
    )


def side_lying_qpos(model) -> np.ndarray:
    """Returns a side-lying free-root pose with all joints at zero."""
    qpos = np.zeros(model.nq)
    qpos[0:3] = np.array([0.0, 0.0, 0.0])
    qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    qpos[2] = _ground_clearance_root_height(model, qpos)
    return qpos


def zero_upright_qpos(model) -> np.ndarray:
    qpos = np.zeros(model.nq)
    qpos[0:3] = np.array([0.0, 0.0, 0.0])
    quat_xyzw = transform.Rotation.from_euler("x", UPRIGHT_ROOT_ROLL).as_quat()
    qpos[3:7] = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    qpos[2] = _ground_clearance_root_height(
        model, qpos, margin=-STANDING_FOOT_PENETRATION
    )
    return qpos


def standing_qpos(model) -> np.ndarray:
    """Returns the current getup target stance with feet just into the ground."""
    qpos = zero_upright_qpos(model)
    qpos[2] = 0.0
    _set_joint_qpos(model, qpos, "joint_femur_right", STANDING_HIP_FLEXION)
    _set_joint_qpos(model, qpos, "joint_femur_left", STANDING_HIP_FLEXION)
    _set_joint_qpos(model, qpos, "joint_tibia_right", STANDING_KNEE_FLEXION)
    _set_joint_qpos(model, qpos, "joint_tibia_left", STANDING_KNEE_FLEXION)
    _set_joint_qpos(
        model, qpos, "joint_tarsometatarsus_right", STANDING_ANKLE_FLEXION
    )
    _set_joint_qpos(model, qpos, "joint_tarsometatarsus_left", STANDING_ANKLE_FLEXION)
    qpos[2] = _ground_clearance_root_height(
        model, qpos, margin=-STANDING_FOOT_PENETRATION
    )
    return qpos


def standing_torso_height(model) -> float:
    data = mujoco.MjData(model)
    data.qpos[:] = standing_qpos(model)
    mujoco.mj_forward(model, data)
    return float(data.site_xpos[model.site(IMU_SITE).id, 2])


def _add_scene(node: ElementTree.Element) -> None:
    visual = ElementTree.SubElement(node, "visual")
    ElementTree.SubElement(
        visual,
        "headlight",
        {"diffuse": "0.8 0.8 0.8", "ambient": "0.2 0.2 0.2"},
    )
    ElementTree.SubElement(
        visual,
        "global",
        {"azimuth": "120", "elevation": "-20"},
    )

    asset = node.find("asset")
    if asset is None:
        asset = ElementTree.SubElement(node, "asset")
    ElementTree.SubElement(
        asset,
        "texture",
        {
            "type": "2d",
            "name": "groundplane",
            "builtin": "checker",
            "rgb1": "1 1 1",
            "rgb2": "0.85 0.85 0.85",
            "width": "300",
            "height": "300",
        },
    )
    ElementTree.SubElement(
        asset,
        "material",
        {
            "name": "groundplane",
            "texture": "groundplane",
            "texuniform": "true",
            "texrepeat": "5 5",
        },
    )

    worldbody = node.find("worldbody")
    ElementTree.SubElement(
        worldbody,
        "geom",
        {
            "name": "floor",
            "type": "plane",
            "size": "0 0 0.01",
            "material": "groundplane",
            "priority": "1",
            "friction": "0.8",
            "condim": "3",
            "contype": "1",
            "conaffinity": "1",
        },
    )


def _add_torso_sites(node: ElementTree.Element) -> None:
    root = node.find(f".//body[@name='{ROOT_BODY}']")
    if root is None:
        raise ValueError(f"Root body {ROOT_BODY!r} not found.")
    ElementTree.SubElement(root, "site", {"name": IMU_SITE, "pos": "0 0 0"})
    ElementTree.SubElement(
        root,
        "camera",
        {
            "name": "track",
            "pos": "-4 -8 3",
            "xyaxes": "1 0 0 0 0.35 0.94",
            "mode": "trackcom",
        },
    )


def _add_sensors(node: ElementTree.Element) -> None:
    sensor = node.find("sensor")
    if sensor is None:
        sensor = ElementTree.SubElement(node, "sensor")
    ElementTree.SubElement(sensor, "gyro", {"site": IMU_SITE, "name": GYRO_SENSOR})
    ElementTree.SubElement(
        sensor,
        "accelerometer",
        {"site": IMU_SITE, "name": ACCELEROMETER_SENSOR},
    )
    ElementTree.SubElement(
        sensor,
        "framezaxis",
        {"objtype": "site", "objname": IMU_SITE, "name": UPVECTOR_SENSOR},
    )
    ElementTree.SubElement(
        sensor,
        "framelinvel",
        {"objtype": "site", "objname": IMU_SITE, "name": GLOBAL_LINVEL_SENSOR},
    )
    ElementTree.SubElement(
        sensor,
        "frameangvel",
        {"objtype": "site", "objname": IMU_SITE, "name": GLOBAL_ANGVEL_SENSOR},
    )


def _add_keyframes(node: ElementTree.Element) -> None:
    keyframe = node.find("keyframe")
    if keyframe is None:
        keyframe = ElementTree.SubElement(node, "keyframe")
    ElementTree.SubElement(
        keyframe,
        "key",
        {
            "name": "zero_upright",
            "qpos": _format_values(
                [0.0, 0.0, 1.5, 1.0, 0.0, 0.0, 0.0] + [0.0] * JOINT_QPOS_DOF
            ),
            "ctrl": _format_values([0.0] * CTRL_DOF),
        },
    )
    ElementTree.SubElement(
        keyframe,
        "key",
        {
            "name": "side_lying_zero",
            "qpos": _format_values(
                [
                    0.0,
                    0.0,
                    3.0,
                    0.7071067811865476,
                    0.7071067811865475,
                    0.0,
                    0.0,
                ]
                + [0.0] * JOINT_QPOS_DOF
            ),
            "ctrl": _format_values([0.0] * CTRL_DOF),
        },
    )


def _format_values(values: list[float]) -> str:
    return " ".join(f"{value:.17g}" for value in values)


def _set_joint_qpos(model, qpos: np.ndarray, joint_name: str, value: float) -> None:
    qpos[model.jnt_qposadr[model.joint(joint_name).id]] = value


def _ground_clearance_root_height(
    model, qpos: np.ndarray, margin: float = 0.05
) -> float:
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    floor_id = model.geom("floor").id if model.ngeom else -1
    min_z = np.inf
    for geom_id in range(model.ngeom):
        if geom_id == floor_id:
            continue
        geom_bottom = data.geom_xpos[geom_id, 2] - _geom_vertical_extent(
            model, data, geom_id
        )
        min_z = min(min_z, geom_bottom)
    return float(qpos[2] - min_z + margin)


def _geom_vertical_extent(model, data, geom_id: int) -> float:
    geom_type = model.geom_type[geom_id]
    size = model.geom_size[geom_id]
    if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        return float(size[0])
    if geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
        xmat = data.geom_xmat[geom_id].reshape(3, 3)
        local_z_vertical = abs(float(xmat[2, 2]))
        return float(size[0] + size[1] * local_z_vertical)
    if geom_type in (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        xmat = data.geom_xmat[geom_id].reshape(3, 3)
        return float(np.sum(np.abs(xmat[2, :]) * size))
    return float(np.max(size))
