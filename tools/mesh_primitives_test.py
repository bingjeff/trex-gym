import unittest

import numpy as np

import mesh_primitives

class TestMeshPrimitives(unittest.TestCase):

    def test_geometry_summary(self):
        axes = np.eye(3)
        origin = np.array([0.0, 0.0, 1.0])
        edge = 1.0
        size_xyz = np.array([edge] * 3)
        geometry = mesh_primitives.GeometrySummary(axes, origin, size_xyz)
        self.assertAlmostEqual(geometry.radius, np.sqrt(0.5 * edge**2))
        self.assertEqual(geometry.length, 0.0)
        self.assertEqual(geometry.matrix[3, 3], 1.0)

    def test_get_octant_xyz(self):
        spread = np.linspace(-1.0, 1.0, 10)
        x, y, z = np.meshgrid(spread, spread, spread)
        points = [x.flatten(), y.flatten(), z.flatten()]
        octants = mesh_primitives.get_octant_xyz(points, min_points=4)
        self.assertEqual(len(octants), 8)
    
    def test_get_fitting_geometry(self):
        spread = np.linspace(-1.0, 1.0, 10)
        x, y, z = np.meshgrid(spread, spread, spread)
        points = [x.flatten(), y.flatten(), z.flatten()]
        geometries = mesh_primitives.get_fitting_geometry(points, max_radius=1.0)
        self.assertEqual(len(geometries), 8)

if __name__ == '__main__':
    unittest.main()
        