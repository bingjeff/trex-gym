# Provides tools to convert meshes into basic geometric primitives.

import numpy as np

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class GeometrySummary:
    axes: np.ndarray
    origin: np.ndarray
    size_xyz: np.ndarray

    @property
    def radius(self) -> float:
        return np.linalg.norm(0.5 * self.size_xyz[:2])

    @property
    def length(self) -> float:
        waist = self.size_xyz[2] - 2.0 * self.radius
        return waist if waist > 0.0 else 0.0

    @property
    def matrix(self) -> np.ndarray:
        parent_t_child = np.eye(4)
        parent_t_child[:3, :3] = self.axes
        parent_t_child[:3, 3] = self.origin
        return parent_t_child

    def transform(self, world_t_root: np.ndarray) -> "GeometrySummary":
        world_t_this = world_t_root @ self.matrix
        return GeometrySummary(
            world_t_this[:3, :3], world_t_this[:3, 3], np.array(self.size_xyz)
        )


def copy_points(points: np.ndarray) -> np.ndarray:
    _, cols = np.shape(points)
    return np.array(points) if cols == 3 else np.transpose(points)


def get_aligned_xyz(points: np.ndarray, rotation: np.ndarray, translation:np.ndarray) -> np.ndarray:
    xyz = copy_points(points)
    return (rotation @ (xyz + translation).T).T


def get_octant_xyz(
    points: np.ndarray, min_points: np.ndarray = 100
) -> list[np.ndarray]:
    xyz = copy_points(points)
    x = xyz[:, 0] >= 0
    y = xyz[:, 1] >= 0
    z = xyz[:, 2] >= 0
    octants = []

    def check(logical_indices):
        if logical_indices.sum() > min_points:
            octants.append(xyz[logical_indices])

    check(x & y & z)
    check(x & ~y & z)
    check(x & ~y & ~z)
    check(x & y & ~z)
    check(~x & y & z)
    check(~x & ~y & z)
    check(~x & ~y & ~z)
    check(~x & y & ~z)
    return octants


def get_principle_axes(points: np.ndarray) -> GeometrySummary:
    xyz = copy_points(points)
    centroid = np.mean(xyz, axis=0)
    xyz -= centroid
    moment = xyz.T @ xyz
    # Axes are sorted x, y, z; make z the dominate axis.
    u, _, _ = np.linalg.svd(moment)
    z = u[:, 0]
    y = u[:, 1]
    x = np.cross(y, z)
    axes = np.vstack([x, y, z]).T
    aligned_xyz = (axes.T @ xyz.T).T
    aligned_center = 0.5 * (
        np.max(aligned_xyz, axis=0) + np.min(aligned_xyz, axis=0)
    )
    aligned_hwl = np.max(aligned_xyz, axis=0) - np.min(aligned_xyz, axis=0)
    center = axes @ aligned_center + centroid
    return GeometrySummary(axes, center, aligned_hwl)


def get_fitting_geometry(
    points: np.ndarray,
    max_radius: float,
    max_divisions: int = 4,
    world_t_parent: Optional[np.ndarray] = None,
    division_count: int = 0,
) -> list[GeometrySummary]:
    geometries = []
    # Calculate the initial bounding box and orientation.
    child = get_principle_axes(points)
    # Rotate the points into an aligned frame to begin subdivision.
    xyz_aligned = get_aligned_xyz(points, child.axes.T, -child.origin)
    world_t_child = (
        np.array(world_t_parent) @ child.matrix
        if world_t_parent is not None
        else child.matrix
    )
    # Subdivide until bounding geometry is smaller than desired radius.
    for xyz_partition in get_octant_xyz(xyz_aligned):
        geometry = get_principle_axes(xyz_partition)
        if geometry.radius > max_radius and division_count < max_divisions:
            # Need to keep subdividing.
            geometries.extend(
                get_fitting_geometry(
                    xyz_partition,
                    max_radius,
                    max_divisions,
                    world_t_child,
                    division_count + 1,
                )
            )
        else:
            geometries.append(geometry.transform(world_t_child))
    return geometries
