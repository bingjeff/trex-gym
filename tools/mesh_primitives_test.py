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


if __name__ == "__main__":
    unittest.main()
