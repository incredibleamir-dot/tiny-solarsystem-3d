"""Tests for scene3d geometry helpers (arrays, matrices). No GL context needed."""

import math
import unittest

import numpy as np

from scene3d import (circle_line_arrays, flat_ring_arrays, look_at,
                     model_matrix, perspective, quad_arrays,
                     rot_translate_matrix, sphere_arrays)


class TestMeshArrays(unittest.TestCase):
    def test_sphere_arrays(self):
        stacks, sectors = 8, 12
        interleaved, idx = sphere_arrays(stacks=stacks, sectors=sectors)
        vc = (stacks + 1) * (sectors + 1)
        self.assertEqual(len(interleaved), vc * 8)  # pos + nrm + uv
        self.assertEqual(len(idx), stacks * sectors * 6)
        self.assertLess(int(idx.max()), vc)
        pos = interleaved.reshape(-1, 8)[:, :3]
        norms = np.linalg.norm(pos, axis=1)
        self.assertTrue(np.allclose(norms, 1.0))

    def test_flat_ring_arrays(self):
        rim, rom = 1.0, 2.0
        segs = 16
        interleaved, idx = flat_ring_arrays(rim, rom, segs=segs)
        vc = (segs + 1) * 2
        self.assertEqual(len(interleaved), vc * 5)  # pos + uv
        self.assertEqual(len(idx), segs * 6)
        arr = interleaved.reshape(-1, 5)
        pos, uv = arr[:, :3], arr[:, 3:]
        # all vertices lie in the XZ plane
        self.assertTrue(np.allclose(pos[:, 1], 0.0))
        # outer vertices at rom, inner at rim
        radii = np.hypot(pos[:, 0], pos[:, 2])
        out = radii[::2]
        inn = radii[1::2]
        self.assertTrue(np.allclose(out, rom))
        self.assertTrue(np.allclose(inn, rim))
        # uv: outer uy 0, inner uy 1; azimuth ux within [0, 1]
        self.assertTrue(np.allclose(uv[::2, 1], 0.0))
        self.assertTrue(np.allclose(uv[1::2, 1], 1.0))
        self.assertTrue(np.all((uv[::2, 0] >= 0.0) & (uv[::2, 0] <= 1.0)))

    def test_ring_radii_scale(self):
        # regression guard: real mesh sized by (rim, rom), not a unit quad
        arr, idx = flat_ring_arrays(3.0, 7.0, segs=8)
        pos = arr.reshape(-1, 5)[:, :3]
        radii = np.hypot(pos[:, 0], pos[:, 2])
        self.assertGreater(float(radii.max()), 6.0)

    def test_quad_arrays(self):
        interleaved, idx = quad_arrays()
        self.assertEqual(len(interleaved), 4 * 5)
        self.assertEqual(len(idx), 6)

    def test_circle_line_arrays(self):
        points = circle_line_arrays([(5.0, 1, 2, 3), (10.0, 4, 5, 6)])
        self.assertEqual(points.size, 2 * 128 * 2 * 6)


class TestMatrices(unittest.TestCase):
    def test_look_at_orthonormal(self):
        view = look_at((10, 0, 0), (0, 0, 0), (0, 1, 0))
        col = view[:3, :3]
        gram = col.T @ col
        self.assertTrue(np.allclose(gram, np.eye(3)))
        self.assertAlmostEqual(float(np.linalg.det(col)), 1.0, places=5)

    def test_look_at_target_in_front(self):
        view = look_at((10, 0, 0), (0, 0, 0), (0, 1, 0))
        # camera at +x looking at origin -> target sits at -10 in view space
        p = view @ np.array([0, 0, 0, 1], dtype=np.float32)
        self.assertAlmostEqual(float(p[2]), -10.0, places=4)

    def test_perspective_layout(self):
        m = perspective(45.0, 1600.0 / 1000.0, 0.1, 1200.0)
        self.assertEqual(m.shape, (4, 4))
        self.assertEqual(m[3, 2], -1.0)
        f = 1.0 / math.tan(math.radians(45.0) / 2.0)
        self.assertAlmostEqual(float(m[1, 1]), f, places=5)

    def test_model_matrix_translate_scale(self):
        m = model_matrix((1, 2, 3), 2.0)
        self.assertTrue(np.allclose(m[:3, 3], (1, 2, 3)))
        self.assertTrue(np.allclose(m[:3, :3], 2.0 * np.eye(3)))
        self.assertEqual(m[3, 3], 1.0)
        p = m @ np.array([0, 0, 0, 1], dtype=np.float32)
        self.assertTrue(np.allclose(p[:3], (1, 2, 3)))

    def test_rot_translate_no_scale(self):
        m = rot_translate_matrix((5, 0, 7))
        self.assertTrue(np.allclose(m[:3, :3], np.eye(3)))
        self.assertTrue(np.allclose(m[:3, 3], (5, 0, 7)))

    def test_model_rotation_preserves_norm(self):
        m = model_matrix((0, 0, 0), 1.0, rot_y=0.9, tilt=0.3)
        p = m @ np.array([1, 2, 3, 1], dtype=np.float32)
        self.assertAlmostEqual(float(np.linalg.norm(p[:3])),
                               float(np.linalg.norm((1, 2, 3))), places=5)


if __name__ == "__main__":
    unittest.main()