# Provides tools to convert meshes into basic geometric primitives.

import numpy as np

from . import geometry
from scipy.spatial import transform


class UvMesh:
    def __init__(self):
        self._vertices = []
        self._normals = []
        self._triangles = []
        self._num_u = 0
        self._num_v = 0

    @property
    def vertices(self) -> np.ndarray:
        return np.array(self._vertices)

    @property
    def normals(self) -> np.ndarray:
        return np.array(self._normals)

    @property
    def triangles(self) -> np.ndarray:
        return np.array(self._triangles)

    def get_vertex_normal(
        self, u: int, v: int
    ) -> tuple[list[float], list[float]]:
        del u, v
        return [0, 0, 0], [0, 0, 1]

    def get_index(self, u: int, v: int) -> int:
        return self._num_u * u + v

    def get_br_triangle(self, u: int, v: int) -> list[int]:
        return [
            self.get_index(u - 1, v - 1),
            self.get_index(u, v - 1),
            self.get_index(u, v),
        ]

    def get_tl_triangle(self, u: int, v: int) -> list[int]:
        return [
            self.get_index(u - 1, v - 1),
            self.get_index(u, v),
            self.get_index(u - 1, v),
        ]

    def _draw(self):
        # Create a grid of points and insert them u-major.
        for u in range(self._num_u):
            for v in range(self._num_v):
                vertex, normal = self.get_vertex_normal(u, v)
                self._vertices.append(vertex)
                self._normals.append(normal)
        # Create the strip of triangles for each quadrangle.
        for u in range(1, self._num_u):
            for v in range(1, self._num_v):
                self._triangles.append(self.get_br_triangle(u, v))
                self._triangles.append(self.get_tl_triangle(u, v))


class SphereMesh(UvMesh):
    def __init__(
        self,
        radius: float,
        num_sectors: int = 10,
        num_stacks: int = 10,
        lo_phi: float = -0.5 * np.pi,
        hi_phi: float = 0.5 * np.pi,
        z_offset: float = 0.0,
    ):
        super().__init__()
        self.radius = radius
        self.z_offset = z_offset
        self._theta = np.linspace(0.0, 2 * np.pi, num_sectors)
        self._phi = np.linspace(lo_phi, hi_phi, num_stacks)
        self._num_u = num_sectors
        self._num_v = num_stacks
        self._draw()

    def get_vertex_normal(
        self, u: int, v: int
    ) -> tuple[list[float], list[float]]:
        phi = self._phi[v]
        theta = self._theta[u]
        normal = [
            np.cos(phi) * np.cos(theta),
            np.cos(phi) * np.sin(theta),
            np.sin(phi),
        ]
        vertex = [self.radius * n for n in normal]
        vertex[2] += self.z_offset
        return vertex, normal


class TubeMesh(UvMesh):
    def __init__(
        self,
        radius: float,
        length: float,
        num_sectors: int = 10,
        num_stacks: int = 10,
        z_offset: float = 0.0,
    ):
        super().__init__()
        self.radius = radius
        self.length = length
        self.z_offset = z_offset
        self._theta = np.linspace(0.0, 2 * np.pi, num_sectors)
        self._z = np.linspace(-0.5 * length, 0.5 * length, num_stacks)
        self._num_u = num_sectors
        self._num_v = num_stacks
        self._draw()

    def get_vertex_normal(
        self, u: int, v: int
    ) -> tuple[list[float], list[float]]:
        z = self._z[v]
        theta = self._theta[u]
        normal = [np.cos(theta), np.sin(theta), 0.0]
        vertex = [
            self.radius * normal[0],
            self.radius * normal[1],
            z + self.z_offset,
        ]
        return vertex, normal


class CapsuleMesh(UvMesh):
    def __init__(
        self,
        radius: float,
        length: float,
        num_sectors: int = 10,
        num_stacks: int = 10,
    ):
        self.radius = radius
        self.length = length
        self.top_cap = SphereMesh(
            radius,
            num_sectors,
            num_stacks,
            lo_phi=-0.5 * np.pi,
            hi_phi=0.0,
            z_offset=-0.5 * length,
        )
        self.tube = TubeMesh(radius, length, num_sectors, num_stacks=2)
        self.bot_cap = SphereMesh(
            radius,
            num_sectors,
            num_stacks,
            lo_phi=0.0,
            hi_phi=0.5 * np.pi,
            z_offset=0.5 * length,
        )
        self._num_u = num_sectors
        self._num_v = (
            self.bot_cap._num_v + self.tube._num_v + self.top_cap._num_v
        )

    @property
    def vertices(self) -> np.ndarray:
        return np.vstack(
            (self.bot_cap.vertices, self.tube.vertices, self.top_cap.vertices)
        )

    @property
    def normals(self) -> np.ndarray:
        return np.vstack(
            (self.bot_cap.normals, self.tube.normals, self.top_cap.normals)
        )

    @property
    def triangles(self) -> np.ndarray:
        return np.vstack(
            (
                self.bot_cap.triangles,
                self.tube.triangles,
                self.top_cap.triangles,
            )
        )

    def get_vertex_normal(
        self, u: int, v: int
    ) -> tuple[list[float], list[float]]:
        index = self.get_index(u, v)
        return self.vertices[index, :], self.normals[index, :]

    def get_index(self, u: int, v: int) -> int:
        v_bottom = self.top_cap._num_v
        v_middle = v_bottom + self.tube._num_v
        if v < v_bottom:
            return self.bot_cap.get_index(u, v)
        elif v < v_middle:
            return self.tube.get_index(u, v - v_bottom)
        else:
            return self.top_cap.get_index(u, v - v_middle)

    def _draw(self):
        pass


def get_octant_xyz(
    points: np.ndarray, min_points: np.ndarray = 100
) -> list[np.ndarray]:
    _, cols = np.shape(points)
    xyz = np.array(points) if cols == 3 else np.transpose(points)
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


def get_axis_aligned_bounding_box(points: np.ndarray) -> geometry.GeometryBox:
    _, cols = np.shape(points)
    xyz = np.array(points) if cols == 3 else np.transpose(points)
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
    world_t_box = geometry.Transform(
        translation=center, rotation=transform.Rotation.from_matrix(axes)
    )
    return geometry.GeometryBox(size_xyz=aligned_hwl, origin=world_t_box)


def get_sphere_or_capsule(
    box: geometry.GeometryBox, world_t_parent: geometry.Transform | None = None
) -> geometry.GeometryCapsule | geometry.GeometrySphere:
    capsule_radius = 0.5 * np.max(box.size_xyz[:2])
    capsule_length = box.length_z - 2.0 * capsule_radius
    world_t_child = (
        world_t_parent * box.origin if world_t_parent else world_t_parent
    )
    if capsule_length > 0.0:
        return geometry.GeometryCapsule(
            radius=capsule_radius, length=capsule_radius, origin=world_t_child
        )
    else:
        return geometry.GeometrySphere(
            radius=capsule_radius, origin=world_t_child
        )


def subdivide_points_to_geometry(
    parent_xyz: np.ndarray,
    max_radius: float,
    max_divisions: int = 4,
    world_t_parent: geometry.Transform | None = None,
    division_count: int = 0,
) -> list[geometry.GeometryBox]:
    geometries = []
    # Calculate the initial bounding box and orientation.
    parent_aabb = get_axis_aligned_bounding_box(parent_xyz)
    # Check if subdivision is necessary.
    sphere_or_capsule = get_sphere_or_capsule(parent_aabb, world_t_parent)
    if (
        sphere_or_capsule.radius > max_radius
        and division_count < max_divisions
    ):
        # Rotate the points into an aligned frame to begin subdivision.
        child_xyz = parent_aabb.origin.inverse().apply(parent_xyz)
        # Keep track of the accumulated transform into the world frame.
        world_t_child = (
            world_t_parent * parent_aabb.origin
            if world_t_parent
            else parent_aabb.origin
        )
        # Subdivide and find fitting geometries.
        for xyz_partition in get_octant_xyz(child_xyz):
            geometries.extend(
                subdivide_points_to_geometry(
                    xyz_partition,
                    max_radius,
                    max_divisions,
                    world_t_child,
                    division_count + 1,
                )
            )
    else:
        geometries.append(sphere_or_capsule)
    return geometries
