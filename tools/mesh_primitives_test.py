import unittest

import numpy as np

from tools import mesh_primitives


class TestMeshPrimitives(unittest.TestCase):
    def test_get_octant_xyz(self):
        spread = np.linspace(-1.0, 1.0, 10)
        x, y, z = np.meshgrid(spread, spread, spread)
        points = [x.flatten(), y.flatten(), z.flatten()]
        octants = mesh_primitives.get_octant_xyz(points, min_points=4)
        self.assertEqual(len(octants), 8)

    def test_subdivide_points_to_geometry(self):
        spread = np.linspace(-1.0, 1.0, 10)
        x, y, z = np.meshgrid(spread, spread, spread)
        points = [x.flatten(), y.flatten(), z.flatten()]
        geometries = mesh_primitives.subdivide_points_to_geometry(
            points, max_radius=1.0
        )
        self.assertEqual(len(geometries), 8)

    def test_obj_mesh_round_trip(self):
        vertices = np.array(
            [
                [0.0, 0.0, 0.5],
                [0.0, 0.3, 0.0],
                [0.1, 0.3, 0.0],
                [0.1, 0.0, 0.0],
            ]
        )
        normals = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
            ]
        )
        texcoords = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        faces = [
            ([0, 1, 2], [0, 1, 3], [1, 0, 0]),
            ([1, 2, 3], [1, 3, 2], [0, 0, 2]),
        ]
        obj_original = mesh_primitives.ObjMesh(
            vertices=vertices,
            texcoords=texcoords,
            normals=normals,
            triangles=faces,
        )
        obj_roundtrip = mesh_primitives.ObjMesh.from_string(
            obj_original.to_string()
        )
        self.assertSequenceEqual(
            obj_roundtrip.triangles, obj_original.triangles
        )
        np.testing.assert_allclose(
            obj_roundtrip.vertices, obj_original.vertices
        )
        np.testing.assert_allclose(
            obj_roundtrip.texcoords, obj_original.texcoords
        )
        np.testing.assert_allclose(obj_roundtrip.normals, obj_original.normals)

        faces = [
            ([0, 1, 2], None, [1, 0, 0]),
            ([1, 2, 3], None, [0, 0, 2]),
        ]
        obj_original = mesh_primitives.ObjMesh(
            vertices=vertices, normals=normals, triangles=faces
        )
        obj_roundtrip = mesh_primitives.ObjMesh.from_string(
            obj_original.to_string()
        )
        self.assertSequenceEqual(
            obj_roundtrip.triangles, obj_original.triangles
        )

        faces = [
            ([0, 1, 2], None, None),
            ([1, 2, 3], None, None),
        ]
        obj_original = mesh_primitives.ObjMesh(
            vertices=vertices, triangles=faces
        )
        obj_roundtrip = mesh_primitives.ObjMesh.from_string(
            obj_original.to_string()
        )
        self.assertSequenceEqual(
            obj_roundtrip.triangles, obj_original.triangles
        )


if __name__ == "__main__":
    unittest.main()
