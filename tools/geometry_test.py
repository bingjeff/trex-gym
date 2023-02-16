import unittest

import numpy as np

from tools import geometry

from scipy.spatial import transform

_NP_TOLERANCE = 1.0e-12


class TestGeometry(unittest.TestCase):
    def test_transform_identity(self):
        identity = geometry.Transform()
        self.assertAlmostEqual(identity.translation.sum(), 0.0)
        np.testing.assert_allclose(
            np.eye(3), identity.rotation.as_matrix(), atol=_NP_TOLERANCE
        )

    def test_transform_rotation(self):
        translation_x = np.array([1.0, 0.0, 0.0])
        rotation_x = transform.Rotation.from_euler(
            "xyz", (0.5 * np.pi, 0.0, 0.0)
        )
        translation_y = np.array([0.0, 1.0, 0.0])
        rotation_y = transform.Rotation.from_euler("xyz", (0.0, 0.7, 0.0))
        rotation_xy = rotation_x * rotation_y
        root_t_x = geometry.Transform(
            translation=translation_x, rotation=rotation_x
        )
        x_t_y = geometry.Transform(
            translation=translation_y, rotation=rotation_y
        )

        root_t_y = root_t_x * x_t_y
        np.testing.assert_allclose(
            rotation_xy.as_matrix(),
            root_t_y.rotation.as_matrix(),
            atol=_NP_TOLERANCE,
        )
        np.testing.assert_allclose(
            np.array([1.0, 0.0, 1.0]), root_t_y.translation, atol=_NP_TOLERANCE
        )

        vector = np.array([0.0, 0.0, 1.0])
        rotated_vector = root_t_x.apply(vector)
        np.testing.assert_allclose(
            np.array([1.0, -1.0, 0.0]), rotated_vector, atol=_NP_TOLERANCE
        )

    def test_transform_inverse(self):
        rotation = transform.Rotation.from_euler("xyz", (2.0, 0.5, 1.5))
        translation = np.array([0.3, 1.2, -0.7])
        world_t_frame = geometry.Transform(translation, rotation)
        frame_t_world = world_t_frame.inverse()
        identity = world_t_frame * frame_t_world
        self.assertAlmostEqual(identity.translation.sum(), 0.0)
        np.testing.assert_allclose(
            np.eye(3),
            identity.rotation.as_matrix(),
            atol=_NP_TOLERANCE,
        )

    def test_transform_apply(self):
        translation = np.array([1.0, -0.5, 4.0])
        rotation_x = transform.Rotation.from_euler(
            "xyz", (0.5 * np.pi, 0.0, 0.0)
        )
        world_t_frame = geometry.Transform(
            translation=translation, rotation=rotation_x
        )
        vector = np.array([0.0, 0.0, 3.0])

        translated = world_t_frame.apply(vector)
        np.testing.assert_allclose(
            translation + np.array([0.0, -3.0, 0.0]),
            translated,
            atol=_NP_TOLERANCE,
        )

        vectors_wide = np.array([vector, vector + 1.0])
        translated_wide = world_t_frame.apply(vectors_wide)
        self.assertEqual(vectors_wide.shape, translated_wide.shape)

        vectors_tall = vectors_wide.T
        translated_tall = world_t_frame.apply(vectors_tall)
        self.assertEqual(vectors_tall.shape, translated_tall.shape)

        untranslated_tall = world_t_frame.inverse().apply(translated_tall)
        np.testing.assert_allclose(
            vectors_tall, untranslated_tall, atol=_NP_TOLERANCE
        )


if __name__ == "__main__":
    unittest.main()
