"""Headless engine tests (SolarSystemApp + render), using the SDL dummy driver.

These need an OpenGL 3.3-capable context; on machines without one they are
skipped automatically.  Run from the project root:

    python -m unittest discover -s tests -v
"""

import datetime
import math
import os
import tempfile
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pygame
import moderngl

import solar_system as ss
from scene3d import render, to_surface


def _make_app():
    pygame.init()
    app = ss.SolarSystemApp()
    app.W, app.H = ss.W0, ss.H0
    app._rebuild_for_size(center=True)
    app.update_world(0.0)
    return app


class TestApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.app = _make_app()
        except Exception as exc:  # pragma: no cover - depends on the machine
            raise unittest.SkipTest("no OpenGL 3.3 context available: %r" % exc)

    def test_window_and_scene_size(self):
        app = self.app
        self.assertTrue(app.W > 0 and app.H > 0)
        self.assertEqual(app.scene.vw, app.W)
        self.assertEqual(app.scene.vh, app.CANVAS_H)
        self.assertEqual(app.scene.fbo.size, (app.W, app.CANVAS_H))

    def test_textures_loaded(self):
        textures = self.app.scene.textures
        for name in ss.ALL_BODIES:
            self.assertIsNotNone(textures.get(name), name)
        for name in ("SaturnRing", "EarthNight", "Moon", "Sun"):
            self.assertIsNotNone(textures.get(name), name)
        for name, tex in textures.items():
            self.assertGreater(tex.size[0], 0, name)
            self.assertGreater(tex.size[1], 0, name)

    def test_visible_names(self):
        self.assertEqual(set(self.app.visible_names()), set(ss.ALWAYS))
        self.app.show_extras = True
        self.assertEqual(set(self.app.visible_names()), set(ss.ALL_BODIES))
        self.app.show_extras = False

    def test_world_positions(self):
        world = self.app.world
        self.assertTrue(np.allclose(world["Sun"], 0.0))
        for name in ss.ALWAYS:
            self.assertIn(name, world)
            v = world[name]
            self.assertTrue(np.all(np.isfinite(v)))
            dist = float(np.linalg.norm(v))
            self.assertGreater(dist, 0.0)
            self.assertLess(dist, ss.DIST_MAX)

    def test_planet_on_its_orbit_ring(self):
        for name in ss.ALWAYS:
            ring_r = ss.orbit_radius(ss.SEMI_MAJOR[name])
            dist = float(np.linalg.norm(self.app.world[name]))
            self.assertAlmostEqual(dist, ring_r, delta=ring_r * 0.20, msg=name)

    def test_sun_at_origin(self):
        self.assertTrue(np.allclose(self.app.world["Sun"], 0.0))

    def test_build_bodies(self):
        bodies = self.app.build_bodies()
        names = [b["name"] for b in bodies]
        self.assertIn("Sun", names)
        self.assertIn("Moon", names)
        sat = next(b for b in bodies if b["name"] == "Saturn")
        self.assertEqual(tuple(sat["ring"]), ss.SATURN_RING)
        sun = next(b for b in bodies if b["name"] == "Sun")
        self.assertTrue(sun["emissive"])

    def test_build_bodies_night_follows_show_daynight(self):
        self.app.show_daynight = True
        earth = next(b for b in self.app.build_bodies()
                     if b["name"] == "Earth")
        self.assertEqual(earth["night"], 1.0)
        self.app.show_daynight = False
        earth = next(b for b in self.app.build_bodies()
                     if b["name"] == "Earth")
        self.assertEqual(earth["night"], 0.0)
        self.app.show_daynight = True

    def test_render_output_size(self):
        data = render(self.app.ctx, self.app.scene, self.app.cam,
                      self.app.build_bodies(), sun_glow=ss.SUN_GLOW)
        self.assertEqual(len(data), self.app.scene.vw * self.app.scene.vh * 3)

    def test_saturn_ring_is_visible(self):
        """Regression: the ring must produce pixels (fixed the unit-quad bug)."""
        app = self.app
        pos = app.world["Saturn"]
        e = np.array(pos, dtype=np.float32)
        n = e / float(np.linalg.norm(e))
        app.cam.target = np.array(pos, dtype=np.float32)
        app.cam.dist = 13.0
        app.cam.yaw = 0.6
        app.cam.pitch = 0.5
        app.cam.target = np.array(pos, dtype=np.float32)
        bodies = app.build_bodies()
        with_img = render(app.ctx, app.scene, app.cam, bodies,
                          sun_glow=ss.SUN_GLOW)
        saved = app.scene.textures.get("SaturnRing")
        app.scene.textures.pop("SaturnRing", None)
        try:
            without = render(app.ctx, app.scene, app.cam, bodies,
                             sun_glow=ss.SUN_GLOW)
        finally:
            if saved is not None:
                app.scene.textures["SaturnRing"] = saved
        diff = sum(1 for a, b in zip(with_img, without) if a != b)
        self.assertGreater(diff, 1000,
                           "Saturn's ring rendered no pixels in the frame")

    def test_camera_matrices(self):
        cam = self.app.cam
        self.assertGreaterEqual(cam.dist, ss.DIST_MIN)
        self.assertLessEqual(cam.dist, ss.DIST_MAX)
        self.assertEqual(cam.view_matrix().shape, (4, 4))
        self.assertEqual(cam.proj_matrix().shape, (4, 4))
        d0 = cam.dist
        cam.zoom_to(cam.dist, cam.dist, 1.5, self.app.W, self.app.CANVAS_H)
        self.assertNotEqual(cam.dist, d0)

    def test_orbit_drag_direction(self):
        # dragging the left mouse button right must swing the view so the
        # solar system appears to move LEFT (i.e. yaw increases)
        cam = self.app.cam
        yaw0 = cam.yaw
        cam.orbit(200, 0)
        self.assertGreater(cam.yaw, yaw0)

    def test_zoom_wheel_direction(self):
        # wheel up (event.y = +1) must zoom IN: dist * 1.15**-1 < dist
        cam = self.app.cam
        d0 = cam.dist
        cam.zoom_to(cam.dist, cam.dist, 1.15 ** -1.0,
                    self.app.W, self.app.CANVAS_H)
        self.assertLess(cam.dist, d0)

    def test_sub_solar_point(self):
        lon, decl = self.app._dl_subsolar()
        self.assertGreaterEqual(lon, -math.pi)
        self.assertLessEqual(lon, math.pi)
        self.assertLessEqual(abs(math.degrees(decl)), 23.44 + 1e-6)
        # 6 h later, longitude moves ~90 deg (decl stays similar)
        t0 = self.app.sim_time
        self.app.sim_time = t0 + datetime.timedelta(hours=6)
        lon2, _ = self.app._dl_subsolar()
        self.app.sim_time = t0
        delta = (math.degrees(lon2) - math.degrees(lon)) % 360.0
        delta = min(delta, 360.0 - delta)
        self.assertAlmostEqual(delta, 90.0, places=0)

    def test_daylight_map_builds(self):
        app = self.app
        app.view_mode = "daylight"
        surf = app.daylight_surface()
        self.assertEqual(surf.get_size(), (ss.DL_MAP_W, ss.DL_MAP_H))
        app._dl_surf = None
        t0 = app.sim_time
        app.sim_time = t0 + datetime.timedelta(hours=6)
        try:
            surf2 = app.daylight_surface()
            b0 = pygame.image.tostring(surf, "RGB", True)
            b1 = pygame.image.tostring(surf2, "RGB", True)
            self.assertNotEqual(b0, b1, "daylight map did not change over 6h")
        finally:
            app.sim_time = t0
            app._dl_surf = None
            app.view_mode = "3d"

    def test_to_screen_sun_centered(self):
        app = self.app
        app.cam.target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        app._view = app.cam.view_matrix()
        app._proj = app.cam.proj_matrix()
        px, py = app.to_screen((0, 0, 0))
        self.assertGreater(px, 0)
        self.assertLess(px, self.app.W)
        self.assertGreater(py, 0)
        self.assertLess(py, self.app.CANVAS_H)

    def test_snap_display_plain_surface(self):
        app = self.app
        buf = pygame.display.get_surface() or app.screen
        out = ss.snap_display(buf)
        self.assertEqual(out.get_size(), buf.get_size())
        self.assertNotEqual(id(out), id(buf))

    def test_snap_display_rotates_180(self):
        # shots come back rotated 180 (sideways + upside) from the display,
        # so snap_display must re-rotate them upright before saving
        surf = pygame.Surface((100, 60))
        surf.fill((0, 0, 0))
        surf.fill((255, 0, 0), rect=(0, 0, 30, 30))     # top-left marker
        out = ss.snap_display(surf)
        self.assertEqual(out.get_at((15, 15))[:3], (0, 0, 0),
                         "marker must leave top-left")
        self.assertEqual(out.get_at((85, 45))[:3], (255, 0, 0),
                         "top-left content must move to bottom-right (180 flip)")

    def test_shot_composite_orientation(self):
        """End-to-end: a +Y world marker must land near the TOP and a +Z
        marker near the RIGHT of the PNG a composite shot produces."""
        app = self.app
        app.cam.target = np.zeros(3, dtype=np.float32)
        app.cam.yaw = 0.0
        app.cam.pitch = 0.0
        app.cam.dist = 40.0
        for label, pos in (("up", [0.0, 12.0, 0.0]), ("right", [0.0, 0.0, 12.0])):
            bodies = [{"name": "m", "pos": np.array(pos, dtype=np.float32),
                       "radius": 1.2, "rot": 0.0, "tilt": 0.0, "tex": "Sun",
                       "emissive": True, "night": False}]
            data = render(app.ctx, app.scene, app.cam, bodies,
                          sun_glow=ss.SUN_GLOW)
            surf3d = to_surface(data, app.scene.vw, app.scene.vh)
            app.screen.blit(surf3d, (0, 0))
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "orient.png")
                ss.save_png(ss.snap_display(app.screen), path)
                img = pygame.image.load(path)
                w, h = img.get_size()
                xs, ys = [], []
                for y in range(h):
                    for x in range(w):
                        if sum(img.get_at((x, y))[:3]) > 500:
                            xs.append(x); ys.append(y)
                self.assertTrue(xs, "marker not found in %s composite" % label)
                cx = sum(xs) / len(xs)
                cy = sum(ys) / len(ys)
                if label == "up":
                    self.assertLess(cy, h / 2.0,
                                    "+Y marker must appear in the TOP half")
                else:
                    self.assertGreater(cx, w / 2.0,
                                       "+Z marker must appear in the RIGHT half")

    def test_save_png_roundtrip(self):
        surf = pygame.Surface((64, 48))
        surf.fill((30, 60, 90))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "t.png")
            ss.save_png(surf, path)
            self.assertTrue(os.path.exists(path))
            with open(path, "rb") as fh:
                self.assertEqual(fh.read(8), b"\x89PNG\r\n\x1a\n")
            loaded = pygame.image.load(path)
            self.assertEqual(loaded.get_size(), (64, 48))


class TestMathAgain(unittest.TestCase):
    def test_module_importable(self):
        self.assertTrue(hasattr(ss, "SolarSystemApp"))


if __name__ == "__main__":
    unittest.main()