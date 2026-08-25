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
from scene3d import render, to_surface, ORBIT_SEGS, model_matrix


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

    # ------------------------------------------------- regression: ring buffer
    def test_no_ghost_rings_after_hiding_extras(self):
        """Regression: shrinking the orbit-ring set must not leave the old
        rings' geometry in the pre-allocated VBO (they used to keep
        rendering because the VAO drew a fixed vertex count)."""
        app = self.app
        app.show_extras = False
        app.update_world(0.0)
        base = render(app.ctx, app.scene, app.cam, app.build_bodies(),
                      sun_glow=ss.SUN_GLOW)
        app.show_extras = True
        app.update_world(0.0)
        on = render(app.ctx, app.scene, app.cam, app.build_bodies(),
                    sun_glow=ss.SUN_GLOW)
        app.show_extras = False
        app.update_world(0.0)
        off = render(app.ctx, app.scene, app.cam, app.build_bodies(),
                     sun_glow=ss.SUN_GLOW)
        self.assertNotEqual(on, off, "extras toggle changed nothing")
        self.assertEqual(off, base,
                         "stale extra rings leaked into the extras-off frame")
        self.assertEqual(app.scene.orbit_vert_count,
                         len(ss.ALWAYS) * ORBIT_SEGS * 2)

    # ------------------------------------------------------- perf: ephemeris
    def test_update_world_caches_ephemeris(self):
        app = self.app
        calls = []
        orig = ss.solarsystem.Heliocentric

        def spy(*a, **k):
            calls.append(1)
            return orig(*a, **k)

        ss.solarsystem.Heliocentric = spy
        try:
            app.sim_time += datetime.timedelta(hours=1)   # force a recompute
            app.update_world(0.0)
            n = len(calls)
            app.update_world(0.0)                          # cached
            self.assertEqual(len(calls), n)
            app.sim_time += datetime.timedelta(minutes=1)  # clock moved
            app.update_world(0.0)
            self.assertEqual(len(calls), n + 1)
        finally:
            ss.solarsystem.Heliocentric = orig

    def test_spins_freeze_while_paused(self):
        app = self.app
        app.paused = True
        app.spin_angle["Earth"] = 10.0
        try:
            app.update_world(1.0 / 60.0)
            self.assertEqual(app.spin_angle["Earth"], 10.0)
            app.paused = False
            app.update_world(1.0 / 60.0)
            self.assertGreater(app.spin_angle["Earth"], 10.0)
        finally:
            app.paused = True

    # ------------------------------------------------------------ scrub units
    def test_scrub_realtime_is_one_hour_per_second(self):
        app = self.app
        t0 = app.sim_time
        old_rt, old_idx = app.realtime, app.speed_index
        try:
            app.realtime = True
            app.scrub(1, 1.0)
            self.assertEqual(app.sim_time - t0, datetime.timedelta(hours=1))
            app.scrub(-1, 0.5)
            self.assertEqual(app.sim_time - t0, datetime.timedelta(minutes=30))
        finally:
            app.sim_time = t0
            app.realtime = old_rt
            app.speed_index = old_idx

    def test_scrub_preset_runs_double_speed(self):
        app = self.app
        t0 = app.sim_time
        old_rt, old_idx = app.realtime, app.speed_index
        try:
            app.realtime = False
            app.speed_index = 2   # "1 day per second"
            app.scrub(1, 1.0)
            self.assertEqual(app.sim_time - t0, datetime.timedelta(days=2))
        finally:
            app.sim_time = t0
            app.realtime = old_rt
            app.speed_index = old_idx

    # ------------------------------------------------------- input hit region
    def test_in_scene_view_excludes_panel_and_taskbar(self):
        app = self.app
        app.view_mode = "3d"
        try:
            self.assertTrue(app.in_scene_view((10, 10)))
            self.assertFalse(app.in_scene_view((app.W - 10, 10)),
                             "panel area must not pick worlds")
            self.assertFalse(app.in_scene_view((app.VIEW_W + 5, 10)))
            self.assertFalse(app.in_scene_view((10, app.CANVAS_H + 2)),
                             "taskbar must not pick worlds")
            self.assertFalse(app.in_scene_view((-1, 10)))
            app.view_mode = "daylight"
            self.assertFalse(app.in_scene_view((10, 10)))
        finally:
            app.view_mode = "3d"

    # ---------------------------------------------------------------- today
    def test_today_button_uses_wall_clock(self):
        app = self.app
        t0 = app.sim_time
        try:
            app.sim_time = datetime.datetime(2000, 6, 1)
            app.activate("today")
            delta = abs((app.sim_time - ss.utc_now()).total_seconds())
            self.assertLess(delta, 5.0)
        finally:
            app.sim_time = t0

    def test_minute_bucket_is_timezone_independent(self):
        app = self.app
        t = datetime.datetime(2024, 5, 5, 12, 0, 30)
        expected = int((t - ss._UNIX_EPOCH).total_seconds()) // 60
        self.assertEqual(app._minute_bucket(t), expected)

    # ----------------------------------------------- science: sub-solar point
    def test_subsolar_includes_equation_of_time(self):
        # In early November the equation of time peaks near +16.4 min, so at
        # 12:00 UTC the sub-solar point sits ~4 degrees WEST of Greenwich
        # instead of on the meridian.
        app = self.app
        t0 = app.sim_time
        try:
            app.sim_time = datetime.datetime(2026, 11, 3, 12, 0)
            lon, _ = app._dl_subsolar()
            self.assertGreater(math.degrees(lon), -5.5)
            self.assertLess(math.degrees(lon), -2.5)
        finally:
            app.sim_time = t0

    def test_subsolar_declination_tracks_seasons(self):
        app = self.app
        t0 = app.sim_time
        try:
            cases = [
                (datetime.datetime(2026, 6, 21, 12, 0), +23.44),
                (datetime.datetime(2026, 12, 21, 12, 0), -23.44),
                (datetime.datetime(2026, 3, 20, 15, 0), 0.0),
            ]
            for when, want in cases:
                app.sim_time = when
                _, decl = app._dl_subsolar()
                self.assertAlmostEqual(math.degrees(decl), want, delta=0.35,
                                       msg=str(when))
        finally:
            app.sim_time = t0

    # ------------------------------------------------ science: Earth spin phase
    def test_earth_spin_sweeps_daily(self):
        # Advancing 6 hours must turn the mesh ~90 degrees relative to the
        # Sun; a tidal-locked phase would leave it fixed.
        app = self.app
        t0 = app.sim_time
        try:
            app.sim_time = datetime.datetime(2026, 8, 1, 0, 0)
            app.update_world(0.0)
            r0 = app.earth_rot
            app.sim_time += datetime.timedelta(hours=6)
            app.update_world(0.0)
            dr = math.degrees((app.earth_rot - r0) % math.tau)
            dr = min(dr, 360.0 - dr)
            self.assertAlmostEqual(dr, 90.0, delta=2.0)
        finally:
            app.sim_time = t0
            app.update_world(0.0)

    def test_earth_lighting_matches_daynight_map(self):
        # Spherical-trig cross-check: the shader lights a surface point by
        # cos(solar zenith angle).  For the texture's Greenwich equator point
        # that value follows from the sub-solar latitude/longitude of the 2D
        # map; the rendered mesh orientation must reproduce it.
        app = self.app
        app.update_world(0.0)
        earth = next(b for b in app.build_bodies() if b["name"] == "Earth")
        g = np.array([-1.0, 0.0, 0.0], dtype=np.float32)   # u=.5, equator
        m = model_matrix(earth["pos"], 1.0, rot_y=earth["rot"],
                         tilt=earth["tilt"])
        w = m[:3, :3] @ g
        w = w / np.linalg.norm(w)
        sh = -np.asarray(earth["pos"], dtype=np.float64)
        sh /= np.linalg.norm(sh)
        got = float(np.dot(w, sh))
        lon_s, decl = app._dl_subsolar()
        want = math.cos(decl) * math.cos(lon_s)            # lat 0, lon 0
        self.assertAlmostEqual(got, want, delta=1e-3)

    def test_earth_axis_gives_real_seasons(self):
        # The tilt axis is fixed in inertial space, so the pole-to-Sun dot
        # product must peak near the June solstice and bottom out near the
        # December one.
        app = self.app
        t0 = app.sim_time
        vals = {}
        try:
            for label, when in (
                    ("jun", datetime.datetime(2026, 6, 21, 12, 0)),
                    ("dec", datetime.datetime(2026, 12, 21, 12, 0))):
                app.sim_time = when
                app.update_world(0.0)
                ew = app.world["Earth"].astype(np.float64)
                sh = -ew / np.linalg.norm(ew)
                eps = ss.TILTS["Earth"]
                pole = np.array([0.0, math.cos(eps), math.sin(eps)])
                vals[label] = float(np.dot(pole, sh))
            self.assertGreater(vals["jun"], 0.30)
            self.assertLess(vals["dec"], -0.30)
        finally:
            app.sim_time = t0
            app.update_world(0.0)

    # --------------------------------------------------- science: Moon incline
    def test_moon_orbit_ring_is_inclined(self):
        app = self.app
        app.sim_time += datetime.timedelta(days=3)
        app.update_world(0.0)
        raw = app.scene.moon_ring_buf.read()
        pts = np.frombuffer(raw, dtype=np.float32).reshape(-1, 6)
        ys = pts[:, 1].astype(np.float64)
        amp = (ys.max() - ys.min()) / 2.0
        # 5.14-deg inclination at the stylised ORB_MOON radius -> ~0.23.
        self.assertGreater(amp, 0.15, "Moon ring looks flat")
        self.assertLess(amp, 0.31, "Moon ring inclination out of range")

    def test_moon_position_uses_ecliptic_latitude(self):
        app = self.app
        t0 = app.sim_time
        try:
            app.sim_time += datetime.timedelta(days=7)
            app.update_world(0.0)
            t = app.sim_time
            m = ss.solarsystem.Moon(
                year=t.year, month=t.month, day=t.day,
                hour=t.hour, minute=t.minute, UT=0, dst=0,
                longtitude=0.0, latitude=0.0, topographic=False)
            lo, la, _ = m.position()
            lo, la = math.radians(lo), math.radians(la)
            rel = app.world["Moon"] - app.world["Earth"]
            expect = ss.ORB_MOON * np.array(
                [math.cos(lo) * math.cos(la), math.sin(la),
                 math.sin(lo) * math.cos(la)], dtype=np.float32)
            self.assertTrue(np.allclose(rel, expect, atol=1e-3))
        finally:
            app.sim_time = t0
            app.update_world(0.0)


class TestMathAgain(unittest.TestCase):
    def test_module_importable(self):
        self.assertTrue(hasattr(ss, "SolarSystemApp"))


if __name__ == "__main__":
    unittest.main()