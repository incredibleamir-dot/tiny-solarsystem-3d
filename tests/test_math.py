"""Pure-math tests for solar_system.py helpers (no display or GPU needed)."""

import datetime
import math
import unittest

import numpy as np

import solar_system as ss


class TestHelpers(unittest.TestCase):
    def test_clamp(self):
        self.assertEqual(ss.clamp(5, 0, 10), 5)
        self.assertEqual(ss.clamp(-3, 0, 10), 0)
        self.assertEqual(ss.clamp(99, 0, 10), 10)

    def test_lerp_color(self):
        self.assertEqual(ss.lerp_color((0, 0, 0), (200, 100, 50), 0.0), (0, 0, 0))
        self.assertEqual(
            ss.lerp_color((0, 0, 0), (200, 100, 50), 1.0), (200, 100, 50))
        self.assertEqual(ss.lerp_color((0, 0, 0), (200, 100, 50), 0.5), (100, 50, 25))

    def test_signpower(self):
        self.assertEqual(ss.signpower(0.0, 0.65), 0.0)
        self.assertAlmostEqual(ss.signpower(8.0, 3.0), 512.0)
        self.assertAlmostEqual(ss.signpower(-8.0, 3.0), -512.0)
        self.assertAlmostEqual(ss.signpower(-4.0, 0.5), -2.0)


class TestTime(unittest.TestCase):
    def test_julian_day_known_epoch(self):
        # J2000.0 = 2000-01-01 12:00 TT -> JD 2451545.0
        jd = ss.julian_day(datetime.datetime(2000, 1, 1, 12, 0, 0))
        self.assertAlmostEqual(jd, 2451545.0, places=5)

    def test_gmst_known_epoch(self):
        # GMST at J2000.0 is the constant used by the formula, ~280.46 deg.
        g = ss.gmst_deg(datetime.datetime(2000, 1, 1, 12, 0, 0))
        self.assertAlmostEqual(g, 280.46061837, places=4)

    def test_gmst_range(self):
        t = datetime.datetime(2024, 6, 15, 3, 42, 0)
        g = ss.gmst_deg(t)
        self.assertGreaterEqual(g, 0.0)
        self.assertLess(g, 360.0)

    def test_gmst_advances(self):
        t0 = datetime.datetime(2024, 1, 1, 0, 0, 0)
        t1 = t0 + datetime.timedelta(hours=6)
        # 6 sidereal-ish hours ~ (6h * 15.041 deg/h) ~ 90 deg
        diff = (ss.gmst_deg(t1) - ss.gmst_deg(t0)) % 360.0
        self.assertAlmostEqual(diff, 90.25, places=1)


class TestScaling(unittest.TestCase):
    def test_orbit_radius_identity_au(self):
        self.assertAlmostEqual(ss.orbit_radius(1.0), ss.AU_UNIT)

    def test_orbit_radius_monotonic(self):
        radii = [ss.orbit_radius(sma) for sma in (0.387, 1.0, 5.203, 39.48)]
        self.assertEqual(radii, sorted(radii))
        self.assertGreater(min(radii), 0.0)

    def test_scene_radius_earth(self):
        self.assertAlmostEqual(ss.scene_radius(6371.0), 0.85)

    def test_scene_radius_clamped(self):
        self.assertEqual(ss.scene_radius(1.0), 0.24)
        self.assertEqual(ss.scene_radius(1e9), 3.0)


class TestConstants(unittest.TestCase):
    def test_saturn_ring_radii(self):
        rim, rom = ss.SATURN_RING
        self.assertGreater(rim, ss.R_3D["Saturn"])
        self.assertGreater(rom, rim)
        self.assertAlmostEqual(rim, ss.R_3D["Saturn"] * 1.45)
        self.assertAlmostEqual(rom, ss.R_3D["Saturn"] * 2.6)

    def test_planets_in_always(self):
        for name in ("Mercury", "Venus", "Earth", "Mars", "Jupiter",
                     "Saturn", "Uranus", "Neptune"):
            self.assertIn(name, ss.ALWAYS)

    def test_period_text(self):
        self.assertIn("28 days", ss.period_text("Moon"))
        self.assertIn("year", ss.period_text("Pluto"))