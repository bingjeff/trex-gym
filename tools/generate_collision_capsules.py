#!/usr/bin/env python3
"""Fits collision capsules from visual meshes and writes them into a URDF."""

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import geometry
from tools import mesh_primitives
from tools import mujoco_parsing
from tools import urdf_parsing

_GENERATED_COLLISION_PREFIX = "collision_capsule_"
_FIT_SAMPLES = 128
_RADIUS_PADDING = 1.01


@dataclasses.dataclass(frozen=True)
class CapsuleFitSpec:
    name: str
    owner_link: str
    source_links: tuple[str, ...]


@dataclasses.dataclass
class CapsuleFit:
    spec: CapsuleFitSpec
    capsule: geometry.GeometryCapsule
    max_outside_distance: float
    mean_abs_surface_error: float
    p95_abs_surface_error: float


def _links(prefix: str, first: int, last: int) -> tuple[str, ...]:
    return tuple(f"{prefix}_{index:02d}" for index in range(first, last + 1))


def _foot_specs(side: str) -> list[CapsuleFitSpec]:
    names = [
        f"link_tarsometatarsus_{side}",
        f"link_toe_01_a_{side}",
        f"link_toe_01_b_{side}",
        f"link_toe_01_c_{side}",
        f"link_toe_02_a_{side}",
        f"link_toe_02_b_{side}",
        f"link_toe_02_c_{side}",
        f"link_toe_03_a_{side}",
        f"link_toe_03_b_{side}",
        f"link_toe_03_c_{side}",
        f"link_toe_03_d_{side}",
        f"link_toe_04_a_{side}",
        f"link_toe_04_b_{side}",
        f"link_toe_04_c_{side}",
        f"link_toe_04_d_{side}",
        f"link_toe_04_e_{side}",
        f"link_toe_05_a_{side}",
    ]
    return [
        CapsuleFitSpec(
            name=link.removeprefix("link_"),
            owner_link=link,
            source_links=(link,),
        )
        for link in names
    ]


CAPSULE_FIT_SPECS = (
    tuple(_foot_specs("right"))
    + tuple(_foot_specs("left"))
    + (
        CapsuleFitSpec(
            name="pelvis",
            owner_link="link_vertebrae_sacral",
            source_links=("link_vertebrae_sacral",),
        ),
        CapsuleFitSpec(
            name="torso_rear",
            owner_link="link_vertebra_dorsal_12",
            source_links=_links("link_vertebra_dorsal", 9, 12),
        ),
        CapsuleFitSpec(
            name="torso_mid",
            owner_link="link_vertebra_dorsal_08",
            source_links=_links("link_vertebra_dorsal", 5, 8),
        ),
        CapsuleFitSpec(
            name="torso_front",
            owner_link="link_vertebra_dorsal_04",
            source_links=_links("link_vertebra_dorsal", 0, 4),
        ),
        CapsuleFitSpec(
            name="neck",
            owner_link="link_vertebra_cervical_09",
            source_links=_links("link_vertebra_cervical", 1, 9) + ("link_atlas_axis",),
        ),
        CapsuleFitSpec(
            name="head",
            owner_link="link_cranium",
            source_links=("link_cranium", "link_mandible"),
        ),
        CapsuleFitSpec(
            name="tail_base",
            owner_link="link_vertebra_caudal_00",
            source_links=_links("link_vertebra_caudal", 0, 9),
        ),
        CapsuleFitSpec(
            name="tail_mid",
            owner_link="link_vertebra_caudal_10",
            source_links=_links("link_vertebra_caudal", 10, 23),
        ),
        CapsuleFitSpec(
            name="tail_distal",
            owner_link="link_vertebra_caudal_24",
            source_links=_links("link_vertebra_caudal", 24, 33),
        ),
        CapsuleFitSpec(
            name="tail_tip",
            owner_link="link_vertebra_caudal_34",
            source_links=_links("link_vertebra_caudal", 34, 44),
        ),
    )
)


def generate_capsule_fits(urdf_path: Path) -> list[CapsuleFit]:
    urdf = urdf_parsing.Urdf.from_element(
        urdf_parsing.read_root_node_from_urdf(str(urdf_path))
    )
    link_poses = mujoco_parsing.urdf_forward_kinematics(urdf)
    mesh_base_path = urdf_path.parent
    return [
        _fit_spec(spec, urdf, link_poses, mesh_base_path) for spec in CAPSULE_FIT_SPECS
    ]


def write_capsules_to_urdf(input_path: Path, output_path: Path, fits: list[CapsuleFit]):
    text = _remove_generated_collision_blocks(input_path.read_text())
    collision_blocks = {}
    for fit in fits:
        collision_blocks.setdefault(fit.spec.owner_link, []).extend(
            _collision_block(fit)
        )
    output_path.write_text(_insert_collision_blocks(text, collision_blocks))


def _fit_spec(
    spec: CapsuleFitSpec,
    urdf: urdf_parsing.Urdf,
    link_poses: dict[str, geometry.Transform],
    mesh_base_path: Path,
) -> CapsuleFit:
    owner_pose_inv = link_poses[spec.owner_link].inverse()
    points = []
    for link_name in spec.source_links:
        link = urdf.links[link_name]
        owner_t_link = owner_pose_inv * link_poses[link_name]
        for visual in link.visual_shapes:
            if not isinstance(visual, geometry.GeometryMesh):
                continue
            mesh_path = mesh_base_path / visual.filename
            link_t_mesh = visual.origin
            points.append((owner_t_link * link_t_mesh).apply(_obj_vertices(mesh_path)))
    if not points:
        raise ValueError(f"No mesh vertices found for capsule spec: {spec.name}")
    return _fit_capsule(spec, np.vstack(points))


def _fit_capsule(spec: CapsuleFitSpec, points: np.ndarray) -> CapsuleFit:
    box = mesh_primitives.get_axis_aligned_bounding_box(points)
    local_points = box.origin.inverse().apply(points)
    z_abs = np.abs(local_points[:, 2])
    radial = np.linalg.norm(local_points[:, :2], axis=1)
    max_half_extent = float(np.max(z_abs))
    best_half_length = 0.0
    best_radius = float(np.max(np.linalg.norm(local_points, axis=1)))
    best_volume = np.inf
    for half_length in np.linspace(0.0, max_half_extent, _FIT_SAMPLES):
        cap_overrun = np.maximum(z_abs - half_length, 0.0)
        radius = float(np.max(np.hypot(radial, cap_overrun)))
        volume = np.pi * radius**2 * (2.0 * half_length)
        volume += (4.0 / 3.0) * np.pi * radius**3
        if volume < best_volume:
            best_volume = volume
            best_half_length = float(half_length)
            best_radius = radius
    capsule = geometry.GeometryCapsule(
        radius=best_radius * _RADIUS_PADDING,
        length=max(2.0 * best_half_length, 1.0e-9),
        origin=box.origin,
    )
    distances = _capsule_signed_distances(points, capsule)
    outside = np.maximum(distances, 0.0)
    return CapsuleFit(
        spec=spec,
        capsule=capsule,
        max_outside_distance=float(np.max(outside)),
        mean_abs_surface_error=float(np.mean(np.abs(distances))),
        p95_abs_surface_error=float(np.percentile(np.abs(distances), 95)),
    )


def _capsule_signed_distances(
    points: np.ndarray, capsule: geometry.GeometryCapsule
) -> np.ndarray:
    local = capsule.origin.inverse().apply(points)
    half_length = 0.5 * capsule.length
    closest_z = np.clip(local[:, 2], -half_length, half_length)
    closest = np.column_stack([np.zeros(len(local)), np.zeros(len(local)), closest_z])
    return np.linalg.norm(local - closest, axis=1) - capsule.radius


def _obj_vertices(path: Path) -> np.ndarray:
    vertices = []
    for line in path.read_text().splitlines():
        if line.startswith("v "):
            vertices.append([float(value) for value in line.split()[1:4]])
    if not vertices:
        raise ValueError(f"No vertices found in OBJ mesh: {path}")
    return np.array(vertices)


def _remove_generated_collision_blocks(text: str) -> str:
    lines = []
    skipping = False
    for line in text.splitlines():
        if f'<collision name="{_GENERATED_COLLISION_PREFIX}' in line:
            skipping = True
        if not skipping:
            lines.append(line)
        if skipping and "</collision>" in line:
            skipping = False
    return "\n".join(lines) + "\n"


def _insert_collision_blocks(text: str, collision_blocks: dict[str, list[str]]) -> str:
    output = []
    current_link = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith('<link name="'):
            current_link = stripped.split('"')[1]
        if stripped == "</link>" and current_link in collision_blocks:
            output.extend(collision_blocks[current_link])
        output.append(line)
        if stripped == "</link>":
            current_link = None
    return "\n".join(output) + "\n"


def _collision_block(fit: CapsuleFit) -> list[str]:
    capsule = fit.capsule
    return [
        f'\t\t<collision name="{_GENERATED_COLLISION_PREFIX}{fit.spec.name}">',
        f'\t\t\t<origin rpy="{urdf_parsing.to_rpy(capsule.origin.rotation)}" '
        f'xyz="{urdf_parsing.to_vec3(capsule.origin.translation)}" />',
        "\t\t\t<geometry>",
        f'\t\t\t\t<capsule radius="{capsule.radius}" length="{capsule.length}" />',
        "\t\t\t</geometry>",
        "\t\t</collision>",
    ]


def _print_report(fits: list[CapsuleFit]):
    print(
        "name,owner_link,radius,length,max_outside_distance,"
        "mean_abs_surface_error,p95_abs_surface_error"
    )
    for fit in fits:
        print(
            f"{fit.spec.name},{fit.spec.owner_link},"
            f"{fit.capsule.radius:.8g},{fit.capsule.length:.8g},"
            f"{fit.max_outside_distance:.8g},"
            f"{fit.mean_abs_surface_error:.8g},"
            f"{fit.p95_abs_surface_error:.8g}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit generated capsule collision geometry into a URDF."
    )
    parser.add_argument("urdf_path", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="Output URDF path. Defaults to updating the input file.",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print capsule fit quality metrics as CSV.",
    )
    args = parser.parse_args()

    fits = generate_capsule_fits(args.urdf_path)
    write_capsules_to_urdf(args.urdf_path, args.output or args.urdf_path, fits)
    if args.report:
        _print_report(fits)


if __name__ == "__main__":
    main()
