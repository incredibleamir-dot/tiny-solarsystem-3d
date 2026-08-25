"""solar_system.py - tiny-solarsystem-3d, a 3D Solar System for kids.

A pygame + ModernGL explorer that renders the whole Solar System as real
textured spheres: the Sun with an additive corona, every planet on its orbit
ring, Saturn's ring, and a day/night Earth with real city lights on the dark
side.  Positions come from the vendored "solarsystem" library - the orbits
are stylised (not to scale) so everything stays visible.

  * Real NASA textures (solarsystemscope.com, CC-BY) with procedural
    fallbacks.
  * Orbit camera: drag to spin, right/middle-drag to slide, wheel to zoom
    (towards whatever your pointer is over, or the world you pinned).
  * Fully responsive: resizable window, fullscreen, adaptive panel & taskbar.
"""

import math
import sys
import os
import random
import struct
import zlib
import datetime

import pygame

import numpy as np
import moderngl

LIB_DIR = os.path.dirname(os.path.abspath(__file__))
if LIB_DIR not in sys.path:
    sys.path.insert(0, LIB_DIR)
sys.path.insert(0, os.path.join(LIB_DIR, "vendor"))

import solarsystem
from solarsystem.functions import precession_longitude_correction
from scene3d import (Camera, Scene, render, to_surface, FOVY,
                     ORBIT_SEGS, MOON_SEGS)

# ----------------------------------------------------------------------------
# Defaults (windowed; everything adapts to the current window size)
W0, H0 = 1280, 800
BAR_H0 = 68
PANEL_W0 = 310

FPS = 60
MARGIN = 70
AU_KM = 149597870.7
EARTH_RADIUS_PER_AU = 23455.0
_UNIX_EPOCH = datetime.datetime(1970, 1, 1)

ORB_MOON = 2.6
SUN_R = 3.4
R_MOON = 0.5
DIST_HOME = 62.0
DIST_MIN, DIST_MAX = 4.0, 520.0
SUN_GLOW = 7.5

AU_UNIT = 24.0
POWER = 0.65

MOON_ORBIT_COLOR = (140, 180, 255)
ORBIT_COLOR = (80, 130, 230)

DL_MAP_W = 1280
DL_MAP_H = 640

# ----------------------------------------------------------------------------
# Palette
C_BG = (4, 7, 18)
C_STAR_MIN, C_STAR_MAX = 80, 230
C_TASKBAR_EDGE = (64, 96, 180)

C_CYAN = (0, 232, 255)
C_CYAN_DIM = (38, 140, 180)
C_MAGENTA = (255, 70, 170)
C_AMBER = (255, 208, 96)
C_GREEN = (110, 255, 150)
C_RED = (255, 96, 96)
C_TEXT = (226, 236, 252)
C_DIM = (138, 152, 188)
C_GLASS = (14, 20, 48)

TASKBAR_BG = (9, 13, 32)
PANEL_FILL = (12, 18, 44, 178)
PANEL_LINE = (0, 232, 255, 130)

BTN_W = 46
BTN_GAP = 8
BTN_RADIUS = 12
ICON_COLOR = (205, 244, 255)

# ----------------------------------------------------------------------------
# Time
SPEED_DAYS = [0.1, 0.5, 1, 3, 7, 15, 30, 90, 365]
SPEED_LABEL = {
    0.1: "6 hours per second",
    0.5: "12 hours per second",
    1: "1 day per second",
    3: "3 days per second",
    7: "1 week per second",
    15: "2 weeks per second",
    30: "1 month per second",
    90: "3 months per second",
    365: "1 year per second",
}

# ----------------------------------------------------------------------------
# Worlds
ALWAYS = ["Mercury", "Venus", "Earth", "Mars", "Jupiter",
          "Saturn", "Uranus", "Neptune", "Pluto"]
EXTRAS = ["Ceres", "Chiron", "Eris"]
ALL_BODIES = ALWAYS + EXTRAS

SEMI_MAJOR = {
    "Mercury": 0.387, "Venus": 0.723, "Earth": 1.000, "Mars": 1.524,
    "Jupiter": 5.203, "Saturn": 9.537, "Uranus": 19.191, "Neptune": 30.069,
    "Pluto": 39.480, "Ceres": 2.770, "Chiron": 13.600, "Eris": 67.700,
}

PERIOD_DAYS = {
    "Mercury": 88, "Venus": 225, "Earth": 365, "Mars": 687,
    "Jupiter": 4333, "Saturn": 10759, "Uranus": 30687, "Neptune": 60190,
    "Pluto": 90560, "Ceres": 1682, "Chiron": 18440, "Eris": 203600,
    "Moon": 27.3,
}

COLORS = {
    "Sun":     (255, 214,  92),
    "Mercury": (178, 180, 190),
    "Venus":   (238, 190, 108),
    "Earth":   (72, 150, 250),
    "Mars":    (232,  96,  70),
    "Jupiter": (236, 174, 122),
    "Saturn":  (244, 214, 154),
    "Uranus":  (140, 226, 238),
    "Neptune": (82, 100, 240),
    "Pluto":   (210, 172, 148),
    "Ceres":   (160, 160, 145),
    "Chiron":  (196, 226, 124),
    "Eris":    (196, 190, 248),
    "Moon":    (218, 218, 224),
}

R_KM = {
    "Sun": 695700, "Mercury": 2440, "Venus": 6052, "Earth": 6371,
    "Mars": 3390, "Jupiter": 69911, "Saturn": 58232, "Uranus": 25362,
    "Neptune": 24622, "Pluto": 1188, "Ceres": 473, "Chiron": 120,
    "Eris": 1163, "Moon": 1737,
}

TILTS = {
    "Mercury": 0.034, "Venus": 2.64, "Earth": 0.409, "Mars": 0.439,
    "Jupiter": 0.055, "Saturn": 0.467, "Uranus": 1.706, "Neptune": 0.494,
    "Pluto": 2.11, "Ceres": 0.09, "Chiron": 0.52, "Eris": 0.78,
}

CATEGORY = {
    "Sun": "OUR STAR",
    "Mercury": "PLANET", "Venus": "PLANET", "Earth": "PLANET",
    "Mars": "PLANET", "Jupiter": "PLANET", "Saturn": "PLANET",
    "Uranus": "PLANET", "Neptune": "PLANET",
    "Pluto": "DWARF PLANET", "Ceres": "DWARF PLANET", "Eris": "DWARF PLANET",
    "Chiron": "CENTAUR", "Moon": "EARTH'S MOON",
}

FACTS = {
    "Sun": "The Sun is a giant star made of hot glowing gas! You could fit about one million Earths inside it.",
    "Mercury": "Mercury is the smallest planet and the closest to the Sun. One year on Mercury is only 88 days!",
    "Venus": "Venus is the hottest planet of all! It spins the opposite way to most other planets.",
    "Earth": "Earth is our home. It is the only planet we know of that has oceans and life!",
    "Mars": "Mars is the Red Planet. It has the tallest volcano in the whole Solar System!",
    "Jupiter": "Jupiter is the biggest planet. More than 1,300 Earths could fit inside it!",
    "Saturn": "Saturn is famous for its amazing rings, which are made of ice and rock.",
    "Uranus": "Uranus rolls around the Sun on its side, like a giant ball.",
    "Neptune": "Neptune is the windiest planet. Its winds blow even faster than a jet airplane!",
    "Pluto": "Pluto is a dwarf planet, even smaller than Earth's Moon!",
    "Ceres": "Ceres is a dwarf planet that lives between Mars and Jupiter in the asteroid belt.",
    "Chiron": "Chiron is a strange icy world that travels between Saturn and Uranus.",
    "Eris": "Eris is a dwarf planet that lives far, far away - even farther than Pluto!",
    "Moon": "The Moon goes around Earth. It takes about 28 days to travel all the way around!",
}

NAME_TITLES = {
    "Mercury": "Mercury", "Venus": "Venus", "Earth": "Earth", "Mars": "Mars",
    "Jupiter": "Jupiter", "Saturn": "Saturn", "Uranus": "Uranus",
    "Neptune": "Neptune", "Pluto": "Pluto", "Ceres": "Ceres",
    "Chiron": "Chiron", "Eris": "Eris", "Moon": "The Moon", "Sun": "The Sun",
}

REPEAT_BTNS = {"slower", "faster", "rewind", "forward"}

# ----------------------------------------------------------------------------
# Textures (solarsystemscope.com, CC-BY 4.0)
TEX_DIR = os.path.join(LIB_DIR, "textures")
TEX_FILES = {
    "Sun": "2k_sun.jpg",
    "Mercury": "2k_mercury.jpg",
    "Venus": "2k_venus_surface.jpg",
    "Earth": "2k_earth_daymap.jpg",
    "Mars": "2k_mars.jpg",
    "Jupiter": "2k_jupiter.jpg",
    "Saturn": "2k_saturn.jpg",
    "Uranus": "2k_uranus.jpg",
    "Neptune": "2k_neptune.jpg",
    "Pluto": "2k_pluto.png",
    "Ceres": "2k_ceres_fictional.jpg",
    "Eris": "2k_eris_fictional.jpg",
    "Moon": "2k_moon.jpg",
}
NIGHT_FILE = "2k_earth_nightmap.jpg"
RING_FILE = "2k_saturn_ring_alpha.png"

# decorative spin speed (deg/second) - purely for looks
SPIN = {
    "Mercury": 1.4, "Venus": 0.9, "Earth": 2.6, "Mars": 2.2,
    "Jupiter": 6.0, "Saturn": 5.4, "Uranus": 2.4, "Neptune": 3.6,
    "Pluto": 1.0, "Ceres": 1.5, "Chiron": 1.2, "Eris": 1.0,
    "Moon": 1.6,
}


# ----------------------------------------------------------------------------
# small helpers
def clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def lerp_color(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def utc_now():
    """Naive datetime holding the current UTC wall-clock time."""
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)


def julian_day(t):
    y, m, d = t.year, t.month, t.day
    a = (14 - m) // 12
    yy = y + 4800 - a
    mm = m + 12 * a - 3
    jdn = d + (153 * mm + 2) // 5 + 365 * yy + yy // 4 \
        - yy // 100 + yy // 400 - 32045
    frac = (t.hour + t.minute / 60.0 + t.second / 3600.0) / 24.0
    return jdn + frac - 0.5


def gmst_deg(t):
    jd = julian_day(t)
    return (280.46061837 + 360.98564736629 * (jd - 2451545.0)) % 360.0


def sun_ra_decl(t):
    """Apparent geocentric RA / declination of the Sun, in degrees.

    Low-precision solar theory (Paul Schlyter's formulas, the same model the
    vendored library uses), accurate to ~0.01 deg - far better than needed
    for terminator placement.  The precession correction matches the one the
    vendored library applies to its own longitudes, keeping every part of
    the app in the same equinox-of-date frame.  Because the RA is the
    *apparent* sun, the equation of time is included automatically."""
    d = julian_day(t) - 2451543.5
    w = 282.9404 + 4.70935e-5 * d
    e = 0.016709 - 1.151e-9 * d
    M = math.radians((356.047 + 0.9856002585 * d) % 360.0)
    E = M + e * math.sin(M) * (1.0 + e * math.cos(M))
    x = math.cos(E) - e
    y = math.sin(E) * math.sqrt(1.0 - e * e)
    v = math.atan2(y, x)
    lam = v + math.radians(w)
    lam -= math.radians(precession_longitude_correction(julian_day(t)))
    oblecl = math.radians(23.4393 - 3.563e-7 * d)
    xe = math.cos(lam)
    ye = math.cos(oblecl) * math.sin(lam)
    ze = math.sin(oblecl) * math.sin(lam)
    ra = math.degrees(math.atan2(ye, xe)) % 360.0
    decl = math.degrees(math.asin(max(-1.0, min(1.0, ze))))
    return ra, decl


def orbit_radius(sma):
    return AU_UNIT * (sma ** POWER)


def scene_radius(km):
    return max(0.24, min(3.0, 0.85 * math.sqrt(km / 6371.0)))


R_3D = {
    "Sun": SUN_R, "Moon": R_MOON,
    "Earth": 1.0,
    **{n: scene_radius(R_KM[n]) for n in ALL_BODIES if n != "Earth"},
}

SATURN_RING = (R_3D["Saturn"] * 1.45, R_3D["Saturn"] * 2.6)

def period_text(name):
    days = PERIOD_DAYS.get(name)
    if days is None:
        return ""
    if name == "Moon":
        return "about 28 days"
    if days < 1000:
        return "about %d days" % round(days)
    return "about %d years" % round(days / 365)


def _poly(surf, points, color):
    pygame.draw.polygon(surf, color, [(int(x), int(y)) for x, y in points])


# ----------------------------------------------------------------------------
# procedural fallback texture (a 512x256 equirectangular-ish surface)
def procedural_texture(name):
    rng = random.Random("tex-" + name)
    w, h = 512, 256
    base = COLORS.get(name, (150, 150, 160))
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    surf.fill((*base, 255))

    if name in ("Jupiter", "Saturn", "Uranus", "Neptune"):
        for y in range(h):
            t = y / h
            wave = math.sin(t * math.tau * (2.5 + rng.random() * 2.5)
                            + rng.random() * math.tau)
            wave2 = math.sin(t * math.tau * (9 + rng.random() * 4)) * 0.4
            k = 1.0 + 0.22 * wave + 0.10 * wave2
            r = int(clamp(base[0] * k, 0, 255))
            g = int(clamp(base[1] * k, 0, 255))
            b = int(clamp(base[2] * k, 0, 255))
            pygame.draw.line(surf, (r, g, b, 255), (0, y), (w, y))
        for _ in range(50):
            cx = rng.randrange(0, w)
            cy = rng.randrange(0, h)
            cr = rng.randrange(4, 22)
            k = 0.80 + rng.random() * 0.35
            col = (int(clamp(base[0] * k, 0, 255)),
                   int(clamp(base[1] * k, 0, 255)),
                   int(clamp(base[2] * k, 0, 255)), 150)
            pygame.draw.ellipse(surf, col, (cx - cr, cy - cr, cr * 2, cr))
        if name == "Jupiter":
            cx, cy, cr = int(w * 0.72), int(h * 0.56), int(h * 0.09)
            pygame.draw.ellipse(surf, (216, 120, 74, 255),
                                (cx - cr, cy - cr, cr * 2, cr * 2))
            pygame.draw.ellipse(surf, (240, 158, 110, 200),
                                (cx - cr, cy - cr, cr * 2, cr * 2), 3)
    elif name in ("Mercury", "Moon", "Mars", "Pluto", "Ceres",
                  "Chiron", "Eris"):
        for _ in range(1600):
            x = rng.randrange(0, w)
            y = rng.randrange(0, h)
            k = 0.82 + rng.random() * 0.36
            r = int(clamp(base[0] * k, 0, 255))
            g = int(clamp(base[1] * k, 0, 255))
            b = int(clamp(base[2] * k, 0, 255))
            sz = 1 if rng.random() < 0.8 else 2
            pygame.draw.rect(surf, (r, g, b, 255), (x, y, sz, sz))
        for _ in range(40):
            cx = rng.randrange(0, w)
            cy = rng.randrange(0, h)
            cr = rng.randrange(3, 16)
            col = (int(clamp(base[0] * 0.7, 0, 255)),
                   int(clamp(base[1] * 0.7, 0, 255)),
                   int(clamp(base[2] * 0.7, 0, 255)), 120)
            pygame.draw.circle(surf, col, (cx, cy), cr, 2)
            pygame.draw.circle(surf, (255, 255, 255, 40),
                               (cx - cr // 3, cy - cr // 3), max(1, cr // 3))
    elif name == "Venus":
        for y in range(h):
            t = y / h
            wave = math.sin(t * math.tau * 3 + 0.7) * 0.18 \
                + math.sin(t * math.tau * 7 + 1.4) * 0.10
            k = 1.0 + wave
            pygame.draw.line(surf,
                             (int(clamp(base[0] * k, 0, 255)),
                              int(clamp(base[1] * k, 0, 255)),
                              int(clamp(base[2] * k, 0, 255)), 255),
                             (0, y), (w, y))
        for _ in range(26):
            cx = rng.randrange(0, w)
            cy = rng.randrange(0, h)
            cr = rng.randrange(10, 40)
            col = (240, 226, 196, 70)
            pygame.draw.ellipse(surf, col, (cx - cr, cy - cr, cr * 2, cr))
    elif name == "Sun":
        for _ in range(4000):
            x = rng.randrange(0, w)
            y = rng.randrange(0, h)
            k = 0.55 + rng.random() * 0.9
            r = int(clamp(255 * k, 0, 255))
            g = int(clamp(200 * k, 0, 255))
            b = int(clamp(70 * k, 0, 255))
            pygame.draw.rect(surf, (r, g, b, 255), (x, y, 2, 2))
    else:
        for _ in range(1200):
            x = rng.randrange(0, w)
            y = rng.randrange(0, h)
            k = 0.8 + rng.random() * 0.4
            r = int(clamp(base[0] * k, 0, 255))
            g = int(clamp(base[1] * k, 0, 255))
            b = int(clamp(base[2] * k, 0, 255))
            pygame.draw.rect(surf, (r, g, b, 255), (x, y, 1, 1))
    return surf


def procedural_night():
    rng = random.Random("tex-night")
    w, h = 512, 256
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    surf.fill((6, 10, 26, 255))
    for _ in range(1200):
        x = rng.randrange(0, w)
        y = rng.randrange(0, h)
        k = 0.5 + rng.random() * 0.6
        pygame.draw.rect(surf, (int(30 * k), int(44 * k), int(80 * k), 255),
                         (x, y, 1, 1))
    for _ in range(160):
        x = rng.randrange(0, w)
        y = rng.randrange(0, h)
        br = rng.random()
        col = (int(255 * br), int(210 * br), int(140 * br), 255)
        pygame.draw.rect(surf, col, (x, y, 1, 1))
    return surf


# ----------------------------------------------------------------------------
# icon painters
def _icon_play(surf, s):
    _poly(surf, [(0.30 * s, 0.18 * s), (0.30 * s, 0.82 * s),
                 (0.84 * s, 0.50 * s)], ICON_COLOR)


def _icon_pause(surf, s):
    pygame.draw.rect(surf, ICON_COLOR,
                     (int(0.26 * s), int(0.20 * s), int(0.17 * s), int(0.60 * s)))
    pygame.draw.rect(surf, ICON_COLOR,
                     (int(0.57 * s), int(0.20 * s), int(0.17 * s), int(0.60 * s)))


def _icon_rewind(surf, s):
    _poly(surf, [(0.78 * s, 0.22 * s), (0.78 * s, 0.78 * s),
                 (0.42 * s, 0.50 * s)], ICON_COLOR)
    _poly(surf, [(0.50 * s, 0.22 * s), (0.50 * s, 0.78 * s),
                 (0.14 * s, 0.50 * s)], ICON_COLOR)


def _icon_forward(surf, s):
    _poly(surf, [(0.22 * s, 0.22 * s), (0.22 * s, 0.78 * s),
                 (0.58 * s, 0.50 * s)], ICON_COLOR)
    _poly(surf, [(0.50 * s, 0.22 * s), (0.50 * s, 0.78 * s),
                 (0.86 * s, 0.50 * s)], ICON_COLOR)


def _icon_minus(surf, s):
    pygame.draw.rect(surf, ICON_COLOR,
                     (int(0.16 * s), int(0.44 * s), int(0.68 * s), int(0.12 * s)))


def _icon_plus(surf, s):
    pygame.draw.rect(surf, ICON_COLOR,
                     (int(0.16 * s), int(0.44 * s), int(0.68 * s), int(0.12 * s)))
    pygame.draw.rect(surf, ICON_COLOR,
                     (int(0.44 * s), int(0.16 * s), int(0.12 * s), int(0.68 * s)))


def _icon_clock(surf, s):
    pygame.draw.circle(surf, ICON_COLOR,
                       (int(0.5 * s), int(0.5 * s)), int(0.33 * s), int(0.09 * s))
    pygame.draw.line(surf, ICON_COLOR,
                     (int(0.5 * s), int(0.5 * s)), (int(0.5 * s), int(0.26 * s)),
                     int(0.08 * s))
    pygame.draw.line(surf, ICON_COLOR,
                     (int(0.5 * s), int(0.5 * s)), (int(0.66 * s), int(0.58 * s)),
                     int(0.08 * s))


def _icon_world(surf, s):
    pygame.draw.circle(surf, ICON_COLOR,
                       (int(0.45 * s), int(0.52 * s)), int(0.26 * s), int(0.08 * s))
    pygame.draw.ellipse(surf, ICON_COLOR,
                        (int(0.30 * s), int(0.32 * s), int(0.48 * s), int(0.34 * s)),
                        int(0.08 * s))


def _icon_fit(surf, s):
    pygame.draw.line(surf, ICON_COLOR,
                     (int(0.24 * s), int(0.22 * s)), (int(0.40 * s), int(0.22 * s)),
                     int(0.08 * s))
    pygame.draw.line(surf, ICON_COLOR,
                     (int(0.60 * s), int(0.22 * s)), (int(0.76 * s), int(0.22 * s)),
                     int(0.08 * s))
    pygame.draw.line(surf, ICON_COLOR,
                     (int(0.24 * s), int(0.78 * s)), (int(0.40 * s), int(0.78 * s)),
                     int(0.08 * s))
    pygame.draw.line(surf, ICON_COLOR,
                     (int(0.60 * s), int(0.78 * s)), (int(0.76 * s), int(0.78 * s)),
                     int(0.08 * s))
    pygame.draw.circle(surf, ICON_COLOR,
                       (int(0.5 * s), int(0.5 * s)), int(0.16 * s), int(0.08 * s))
    pygame.draw.circle(surf, ICON_COLOR, (int(0.5 * s), int(0.5 * s)), 2)


def _icon_label(surf, s):
    pygame.draw.rect(surf, ICON_COLOR,
                     (int(0.16 * s), int(0.30 * s), int(0.68 * s), int(0.38 * s)),
                     int(0.07 * s))
    pygame.draw.line(surf, ICON_COLOR,
                     (int(0.30 * s), int(0.49 * s)), (int(0.70 * s), int(0.49 * s)),
                     int(0.06 * s))


def _icon_about(surf, s):
    pygame.draw.circle(surf, ICON_COLOR,
                       (int(0.5 * s), int(0.5 * s)), int(0.34 * s), int(0.08 * s))
    pygame.draw.circle(surf, ICON_COLOR,
                       (int(0.5 * s), int(0.34 * s)), int(0.06 * s))
    pygame.draw.line(surf, ICON_COLOR,
                     (int(0.5 * s), int(0.44 * s)), (int(0.5 * s), int(0.68 * s)),
                     int(0.09 * s))


def _icon_daynight(surf, s):
    pygame.draw.circle(surf, ICON_COLOR,
                       (int(0.5 * s), int(0.5 * s)), int(0.36 * s), int(0.08 * s))
    _poly(surf, [(int(0.5 * s), int(0.14 * s)), (int(0.5 * s), int(0.86 * s)),
                 (int(0.82 * s), int(0.5 * s))], ICON_COLOR)


def _icon_daylight(surf, s):
    pygame.draw.circle(surf, ICON_COLOR,
                       (int(0.58 * s), int(0.5 * s)), int(0.30 * s), int(0.08 * s))
    pygame.draw.polygon(surf, ICON_COLOR, [
        (int(0.58 * s), int(0.20 * s)), (int(0.58 * s), int(0.80 * s)),
        (int(0.86 * s), int(0.5 * s))])
    c = int(0.16 * s)
    pygame.draw.circle(surf, ICON_COLOR, (c, int(0.24 * s)), int(0.10 * s),
                       int(0.05 * s))
    for a in range(8):
        th = a * math.pi / 4
        x0, y0 = c + math.cos(th) * 0.18 * s, int(0.24 * s) + math.sin(th) * 0.18 * s
        x1, y1 = c + math.cos(th) * 0.27 * s, int(0.24 * s) + math.sin(th) * 0.27 * s
        pygame.draw.line(surf, ICON_COLOR, (int(x0), int(y0)),
                         (int(x1), int(y1)), int(0.05 * s))


def _icon_distance(surf, s):
    y = int(0.5 * s)
    x0, x1 = int(0.14 * s), int(0.86 * s)
    pygame.draw.line(surf, ICON_COLOR, (x0, y), (x1, y), int(0.07 * s))
    pygame.draw.line(surf, ICON_COLOR, (x0, y), (x0, int(0.32 * s)), int(0.07 * s))
    pygame.draw.line(surf, ICON_COLOR, (x0, y), (x0, int(0.68 * s)), int(0.07 * s))
    pygame.draw.line(surf, ICON_COLOR, (x1, y), (x1, int(0.32 * s)), int(0.07 * s))
    pygame.draw.line(surf, ICON_COLOR, (x1, y), (x1, int(0.68 * s)), int(0.07 * s))
    pygame.draw.circle(surf, ICON_COLOR, (int(0.5 * s), y), int(0.08 * s))


def _icon_realtime(surf, s):
    c = int(0.5 * s)
    pygame.draw.circle(surf, ICON_COLOR, (c, c), int(0.33 * s), int(0.07 * s))
    pygame.draw.line(surf, ICON_COLOR,
                     (c, c), (c, int(0.5 * s) - int(0.12 * s)), int(0.05 * s))
    pygame.draw.line(surf, ICON_COLOR,
                     (c, c), (int(0.5 * s) + int(0.16 * s), c),
                     int(0.05 * s))
    pygame.draw.circle(surf, ICON_COLOR, (c, c), int(0.05 * s))


def _icon_fullscreen(surf, s):
    m = int(0.12 * s)
    r = int(0.08 * s)
    pygame.draw.line(surf, ICON_COLOR, (m, m), (int(0.38 * s), m), r)
    pygame.draw.line(surf, ICON_COLOR, (m, m), (m, int(0.38 * s)), r)
    pygame.draw.line(surf, ICON_COLOR, (int(0.62 * s), m), (int(0.88 * s), m), r)
    pygame.draw.line(surf, ICON_COLOR, (int(0.88 * s), m),
                     (int(0.88 * s), int(0.38 * s)), r)
    pygame.draw.line(surf, ICON_COLOR, (m, int(0.62 * s)), (m, int(0.88 * s)), r)
    pygame.draw.line(surf, ICON_COLOR, (m, int(0.88 * s)),
                     (int(0.38 * s), int(0.88 * s)), r)
    pygame.draw.line(surf, ICON_COLOR, (int(0.62 * s), int(0.88 * s)),
                     (int(0.88 * s), int(0.88 * s)), r)
    pygame.draw.line(surf, ICON_COLOR, (int(0.88 * s), int(0.62 * s)),
                     (int(0.88 * s), int(0.88 * s)), r)


# ----------------------------------------------------------------------------
# app
class SolarSystemApp:
    def __init__(self):
        pygame.init()
        self.W, self.H = W0, H0
        self.screen = pygame.display.set_mode((self.W, self.H),
                                              pygame.RESIZABLE)
        pygame.display.set_caption("Tiny Solar System - 3D")

        self.font_title = self.get_font(30)
        self.font_section = self.get_font(21)
        self.font_body = self.get_font(18)
        self.font_small = self.get_font(15)
        self.font_tiny = self.get_font(13)
        self.font_label = self.get_font(16)
        self.font_big = self.get_font(28)
        self.font_tooltip = self.get_font(14)

        self.sim_time = utc_now()
        self.paused = True
        self.speed_index = 2
        self.realtime = False
        self.show_extras = False
        self.show_labels = True
        self.show_about = False
        self.show_distance = False
        self.show_daynight = True
        self.fullscreen = False
        self.view_mode = "3d"

        # Ephemeris cache: recomputed only when the clock or the body set
        # changes (previously a full heliocentric solve ran every frame,
        # even while paused).
        self._astro_key = None
        self._bodyset_key = None
        self.planets_au = {}
        self.positions = {}
        self.body_r = {}
        self.moon_r_au = 0.0
        self.world = {}
        self.earth_rot = 0.0

        # Frame cache: the GL scene is re-rendered only when something can
        # change it (camera moved, clock running, toggle flipped); while
        # paused and untouched the last rendered surface is re-blitted.
        self._scene_dirty = True
        self._frame_surf = None
        self._pick_bodies = None

        self._dl_base = None
        self._dl_key = None
        self._dl_surf = None
        self._dl_t0 = -1000

        self.pinned = None
        self.hovered = None
        self.dragging = False
        self.drag_mode = None
        self.drag_start = (0, 0)
        self.drag_moved = False
        self.held_btn = None
        self.held_start = 0
        self.held_last = 0
        self.pressed_btn = None

        self.spin_angle = {}

        self._geometry()

        try:
            self.ctx = moderngl.create_context(standalone=True, require=330)
        except Exception as exc:
            pygame.quit()
            raise SystemExit(
                "Could not create an OpenGL 3.3 context: %s\n"
                "tiny-solarsystem-3d needs a GPU that supports OpenGL 3.3."
                % exc)

        self.cam = Camera(self.W / max(1, self.CANVAS_H),
                          dist_min=DIST_MIN, dist_max=DIST_MAX,
                          start_dist=DIST_HOME)
        self._homog = np.zeros(4, dtype=np.float32)
        self.scene = Scene(self.ctx, self.W, self.CANVAS_H,
                           ring_radii=SATURN_RING)
        self.scene.upload_textures(self._load_texture_surfaces())

        self.label_cache = {}
        self.name_cache = {}
        self.cat_cache = {}
        self.fact_cache = {}
        self.tip_cache = {}
        self.title_surf = self.neon("SOLAR SYSTEM", self.font_title, C_CYAN)

        self.clock = pygame.time.Clock()

        self.cursor = pygame.SYSTEM_CURSOR_ARROW
        self.date_key = None
        self.date_surf = None
        self.status_static_key = None
        self.status_static = None
        self.zoom_slot = 80
        self._moon_phase_key = -1
        self._moon_phase_val = None

        self.btn_size = BTN_W
        self.btn_icon = BTN_W - 14

        self._circle_masks = {}
        self._sun_glows = {}

        self._rebuild_for_size(center=True)

    # ------------------------------------------------------------- texture loading
    def _load_texture_surfaces(self):
        surfs = {}
        def load(fname):
            if not fname:
                return None
            path = os.path.join(TEX_DIR, fname)
            if not os.path.exists(path):
                return None
            return pygame.image.load(path)
        for name, fname in TEX_FILES.items():
            surfs[name] = load(fname) or procedural_texture(name)
        for name in ALL_BODIES:
            if name not in surfs:
                surfs[name] = procedural_texture(name)
        surfs["EarthNight"] = load(NIGHT_FILE) or procedural_night()
        surfs["SaturnRing"] = load(RING_FILE)
        return surfs

    def get_font(self, size):
        for name in ("bahnschrift", "segoeui", "segoe ui", "consolas"):
            try:
                return pygame.font.SysFont(name, size)
            except Exception:
                continue
        return pygame.font.Font(None, size)

    def neon(self, text, font, color):
        base = font.render(text, True, color)
        w, h = base.get_size()
        out = pygame.Surface((w + 12, h + 12), pygame.SRCALPHA)
        small = pygame.transform.smoothscale(base,
                                             (max(1, w // 2), max(1, h // 2)))
        blur = pygame.transform.smoothscale(small, (w, h))
        blur.set_alpha(110)
        out.blit(blur, (6, 6))
        out.blit(base, (6, 6))
        return out

    def cached(self, cache, key, font, text, color):
        item = cache.get(key)
        if item is None:
            item = font.render(text, True, color)
            cache[key] = item
        return item

    def make_icon(self, painter, size=26):
        big = pygame.Surface((size * 4, size * 4), pygame.SRCALPHA)
        painter(big, size * 4)
        return pygame.transform.smoothscale(big, (size, size))

    def make_scanlines(self, w, h):
        s = pygame.Surface((w, h), pygame.SRCALPHA)
        for y in range(0, h, 4):
            pygame.draw.line(s, (0, 0, 0, 24), (0, y), (w, y))
        return s

    def make_divider(self, w):
        s = pygame.Surface((w, 2), pygame.SRCALPHA)
        for x in range(w):
            t = x / max(1, w - 1)
            alpha = int(140 * (1 - t))
            pygame.draw.line(s, (0, 232, 255, alpha), (x, 0), (x, 1))
        return s

    def make_taskbar_grad(self):
        s = pygame.Surface((self.W, self.BAR_H))
        top, bottom = (11, 16, 40), (5, 8, 22)
        for y in range(self.BAR_H):
            t = y / max(1, self.BAR_H - 1)
            pygame.draw.line(s, lerp_color(top, bottom, t), (0, y), (self.W, y))
        return s

    def make_button_glow(self):
        s = pygame.Surface((self.btn_size + 14, self.btn_size + 14),
                           pygame.SRCALPHA)
        pygame.draw.rect(s, (0, 232, 255, 36),
                         (3, 3, self.btn_size + 8, self.btn_size + 8),
                         border_radius=BTN_RADIUS + 4)
        pygame.draw.rect(s, (0, 232, 255, 70),
                         (7, 7, self.btn_size, self.btn_size),
                         border_radius=BTN_RADIUS)
        return s

    def wrap_text(self, text, font, max_w):
        words = text.split(" ")
        lines, cur = [], ""
        for w in words:
            trial = (cur + " " + w).strip()
            if font.size(trial)[0] <= max_w or not cur:
                cur = trial
            else:
                lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return lines

    # ------------------------------------------------------------- layout
    def _geometry(self):
        self.BAR_H = max(54, min(78, round(self.H * 0.085)))
        self.PANEL_W = max(220, min(380, round(self.W * 0.24)))
        self.CANVAS_H = self.H - self.BAR_H
        self.VIEW_W = self.W - self.PANEL_W

    def _taskbar_right_width(self):
        date = self.font_section.size(
            self.sim_time.strftime("%A, %d %B %Y"))[0]
        speed = self.font_small.size(
            "SPEED " + SPEED_LABEL[SPEED_DAYS[self.speed_index]])[0]
        rt = self.font_small.size("REALTIME 1x")[0]
        zoom = self.font_small.size("ZOOM 4000%")[0]
        return date + max(speed, rt) + zoom + 70

    def build_buttons(self):
        defs = [
            ("play", "play", "Play / Pause"),
            ("rewind", "rewind", "Rewind 1 day"),
            ("forward", "forward", "Fast forward 1 day"),
            ("slower", "slower", "Slower"),
            ("faster", "faster", "Faster"),
            ("today", "today", "Back to today"),
            ("realtime", "realtime", "Realtime speed (1 s = 1 s) (T)"),
            ("worlds", "worlds", "More worlds"),
            ("labels", "labels", "Names on / off"),
            ("distance", "distance", "Show distances from Sun"),
            ("daynight", "daynight", "Earth day / night (N)"),
            ("lightmap", "lightmap", "Earth daylight map (G)"),
            ("about", "about", "About this app"),
            ("fullscreen", "fullscreen", "Toggle fullscreen (F11)"),
            ("fit", "fit", "Reset view"),
        ]
        n = len(defs)
        right = self._taskbar_right_width()
        avail = max(40, self.W - right - 30)
        btn = (avail - BTN_GAP * (n - 1)) // n
        gap = BTN_GAP
        if btn < 24:
            # Narrow windows: tighten the gaps first, then the buttons.
            gap = max(2, min(BTN_GAP, (avail - n * 24) // max(1, n - 1)))
            btn = max(16, (avail - gap * (n - 1)) // n)
        btn = int(min(BTN_W, btn))
        self.btn_size = btn
        self.btn_icon = max(10, btn - 12)

        icons = {
            "play": self.make_icon(_icon_play, self.btn_icon),
            "pause": self.make_icon(_icon_pause, self.btn_icon),
            "rewind": self.make_icon(_icon_rewind, self.btn_icon),
            "forward": self.make_icon(_icon_forward, self.btn_icon),
            "slower": self.make_icon(_icon_minus, self.btn_icon),
            "faster": self.make_icon(_icon_plus, self.btn_icon),
            "today": self.make_icon(_icon_clock, self.btn_icon),
            "realtime": self.make_icon(_icon_realtime, self.btn_icon),
            "worlds": self.make_icon(_icon_world, self.btn_icon),
            "fit": self.make_icon(_icon_fit, self.btn_icon),
            "labels": self.make_icon(_icon_label, self.btn_icon),
            "distance": self.make_icon(_icon_distance, self.btn_icon),
            "daynight": self.make_icon(_icon_daynight, self.btn_icon),
            "lightmap": self.make_icon(_icon_daylight, self.btn_icon),
            "about": self.make_icon(_icon_about, self.btn_icon),
            "fullscreen": self.make_icon(_icon_fullscreen, self.btn_icon),
        }
        self.icons = icons

        btns = []
        x = 30
        y = self.CANVAS_H + (self.BAR_H - btn) // 2
        for bid, icon, tip in defs:
            btns.append({
                "id": bid, "icon": icon, "tip": tip,
                "rect": pygame.Rect(x, y, btn, btn),
            })
            x += btn + gap
        return btns

    def _rebuild_for_size(self, center):
        self._geometry()
        self.scene.resize(self.W, self.CANVAS_H)
        self.cam.set_aspect(self.W / max(1, self.CANVAS_H))
        self.panel_surface = pygame.Surface(
            (self.PANEL_W, self.CANVAS_H), pygame.SRCALPHA)
        self.scanlines = self.make_scanlines(self.PANEL_W, self.CANVAS_H)
        self.divider = self.make_divider(self.PANEL_W - 48)
        self.taskbar_grad = self.make_taskbar_grad()
        self.button_glow = self.make_button_glow()
        self.buttons = self.build_buttons()
        self.fact_cache.clear()
        if center:
            self.cam.reset(DIST_HOME)
        self._scene_dirty = True

    def _handle_resize(self, w, h):
        w = max(480, int(w))
        h = max(360, int(h))
        if self.screen.get_size() != (w, h):
            self.screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
        self.W, self.H = w, h
        self._rebuild_for_size(center=False)

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        info = pygame.display.Info()
        if self.fullscreen:
            self.W, self.H = info.current_w, info.current_h
        else:
            self.W, self.H = W0, H0
        flags = pygame.FULLSCREEN if self.fullscreen else pygame.RESIZABLE
        self.screen = pygame.display.set_mode((self.W, self.H), flags)
        self.W, self.H = self.screen.get_size()
        self._rebuild_for_size(center=True)

    def reset_view(self):
        self.pinned = None
        self.cam.reset(DIST_HOME)
        self._scene_dirty = True

    # ---------------------------------------------------------- computation
    def visible_names(self):
        names = list(ALWAYS)
        if self.show_extras:
            names += list(EXTRAS)
        return names

    def update_world(self, dt):
        names = self.visible_names()
        bodyset = tuple(names)
        t = self.sim_time
        if t == self._astro_key and bodyset == self._bodyset_key:
            # Clock and body set unchanged - keep the cached ephemeris.
            if not self.paused:
                self._update_spins(dt)
            return

        self._astro_key = t
        self._bodyset_key = bodyset
        h = solarsystem.Heliocentric(
            year=t.year, month=t.month, day=t.day,
            hour=t.hour, minute=t.minute,
            UT=0, dst=0, view="rectangular", precession=True)
        self.planets_au = h.planets()

        self.positions = {}
        self.body_r = {}
        for n in names:
            x, y, z = self.planets_au[n]
            self.positions[n] = (x, y, z)
            self.body_r[n] = math.sqrt(x * x + y * y + z * z)

        self.world = {}
        for n in names:
            x, y, z = self.planets_au[n]
            rr = math.hypot(x, y)
            ring_r = orbit_radius(SEMI_MAJOR[n])
            k = ring_r / rr if rr > 1e-9 else 0.0
            self.world[n] = np.array([x * k, z * k, y * k], dtype=np.float32)

        self.world["Sun"] = np.zeros(3, dtype=np.float32)

        earth_w = self.world["Earth"]

        moon = solarsystem.Moon(
            year=t.year, month=t.month, day=t.day,
            hour=t.hour, minute=t.minute,
            UT=0, dst=0, longtitude=0.0, latitude=0.0, topographic=False)
        mlon, mlat, mr = moon.position()
        self.moon_r_au = mr / EARTH_RADIUS_PER_AU
        mlon_r = math.radians(mlon)
        mlat_r = math.radians(mlat)
        # True geocentric direction in the ecliptic frame, mapped into the
        # scene (ecliptic x,y,z -> scene x,z,y), so the Moon's 5.14-degree
        # orbital inclination is preserved instead of being flattened.
        moon_dir = np.array(
            [math.cos(mlon_r) * math.cos(mlat_r), math.sin(mlat_r),
             math.sin(mlon_r) * math.cos(mlat_r)], dtype=np.float64)
        self.world["Moon"] = earth_w + ORB_MOON * moon_dir.astype(np.float32)

        # The Moon's orbital plane: fitted from three true geocentric
        # directions sampled a quarter-month apart, so the ring carries the
        # real inclination and the slowly regressing node line.
        dirs = [moon_dir]
        for k in (1, 2):
            mt = t + datetime.timedelta(days=(27.321661 / 4.0) * k)
            mm = solarsystem.Moon(
                year=mt.year, month=mt.month, day=mt.day,
                hour=mt.hour, minute=mt.minute,
                UT=0, dst=0, longtitude=0.0, latitude=0.0, topographic=False)
            lo, la, _ = mm.position()
            lo, la = math.radians(lo), math.radians(la)
            dirs.append(np.array(
                [math.cos(lo) * math.cos(la), math.sin(la),
                 math.sin(lo) * math.cos(la)], dtype=np.float64))
        normal = (np.cross(dirs[0], dirs[1]) + np.cross(dirs[1], dirs[2]) +
                  np.cross(dirs[2], dirs[0]))
        norm = np.linalg.norm(normal)
        if norm > 1e-9:
            normal = normal / norm
            u = dirs[0] / np.linalg.norm(dirs[0])
            v = np.cross(normal, u)
            offsets = []
            for i in range(MOON_SEGS):
                a = math.tau * i / MOON_SEGS
                offsets.append(
                    ORB_MOON * (math.cos(a) * u + math.sin(a) * v))
            self.scene.set_moon_ring(earth_w, offsets, MOON_ORBIT_COLOR)

        # Earth's spin phase: rotate the mesh so the texture longitude that
        # faces the Sun is exactly the sub-solar longitude the 2D daylight
        # map computes.  This replaces the old GMST shortcut and keeps the
        # 3D terminator in sync with the map (equation of time included).
        sh = -earth_w.astype(np.float64)
        sh /= max(np.linalg.norm(sh), 1e-9)
        eps = TILTS["Earth"]
        ce, se = math.cos(eps), math.sin(eps)
        ox = sh[0]
        oz = -se * sh[1] + ce * sh[2]
        rho = math.hypot(ox, oz)
        if rho > 1e-9:
            az_body = math.degrees(math.atan2(oz, ox))
            lon_s, _ = self._dl_subsolar()
            self.earth_rot = math.radians(
                math.degrees(lon_s) + 180.0 - az_body)
        else:
            self.earth_rot = 0.0

        if not self.paused:
            self._update_spins(dt)
        self._update_rings(names)

    def _update_spins(self, dt):
        for n in ALL_BODIES:
            self.spin_angle[n] = (self.spin_angle.get(n, 0.0)
                                  + SPIN.get(n, 0) * dt) % 360.0

    def _update_rings(self, names):
        if getattr(self, "_ring_names", None) != tuple(names):
            self._ring_names = tuple(names)
            rings = [(orbit_radius(SEMI_MAJOR[n]), ORBIT_COLOR)
                     for n in names]
            self.scene.set_orbit_rings(rings)

    def build_bodies(self):
        names = self.visible_names()
        bodies = []
        earth_rot = self.earth_rot
        for n in names:
            is_earth = n == "Earth"
            is_sun = n == "Sun"
            rot = earth_rot if is_earth else math.radians(
                self.spin_angle.get(n, 0.0))
            body = {
                "name": n,
                "pos": self.world[n],
                "radius": R_3D[n],
                "rot": rot,
                "tilt": TILTS.get(n, 0.0),
                "tex": n,
                "emissive": is_sun,
                "night": is_earth and self.show_daynight,
                "is_earth": is_earth,
            }
            if n == "Saturn":
                body["ring_tilt"] = TILTS.get(n, 0.0)
                body["ring_rot"] = rot
                body["ring"] = SATURN_RING
            bodies.append(body)
        bodies.append({
            "name": "Moon",
            "pos": self.world["Moon"],
            "radius": R_MOON,
            "rot": math.radians(self.spin_angle.get("Moon", 0.0)),
            "tilt": 0.0,
            "tex": "Moon",
            "emissive": False,
            "night": False,
        })
        bodies.append({
            "name": "Sun",
            "pos": self.world["Sun"],
            "radius": R_3D["Sun"],
            "rot": math.radians(self.spin_angle.get("Sun", 0.0)),
            "tilt": 0.0,
            "tex": "Sun",
            "emissive": True,
            "night": False,
        })
        return bodies

    # ------------------------------------------------------------- projection
    def to_screen(self, p3):
        h = self._homog
        h[0], h[1], h[2], h[3] = p3[0], p3[1], p3[2], 1.0
        clip = self._proj @ self._view @ h
        if clip[3] <= 1e-9:
            return None
        ndc = clip[:3] / clip[3]
        if ndc[2] < -1.0 or ndc[2] > 1.0:
            return None
        x = (ndc[0] * 0.5 + 0.5) * self.W
        y = (0.5 - ndc[1] * 0.5) * self.CANVAS_H
        return x, y

    def pick(self, px, py, bodies=None):
        if bodies is None:
            bodies = self.build_bodies()
        o, d = self.cam.ray(px, py, self.W, self.CANVAS_H)
        best_t, best = None, None
        for b in bodies:
            c = b["pos"]
            oc = o - c
            bq = float(np.dot(oc, d))
            cq = float(np.dot(oc, oc)) - b["radius"] * b["radius"]
            disc = bq * bq - cq
            if disc < 0.0:
                continue
            t = -bq - math.sqrt(disc)
            if t < 0.0:
                t = -bq + math.sqrt(disc)
            if t < 0.0:
                continue
            if best_t is None or t < best_t:
                best_t, best = t, b["name"]
        return best

    def screen_radius(self, pos, radius):
        eye = self.cam.eye_vec()
        camdist = math.dist(eye, pos)
        if camdist < 1e-6:
            return 1.0
        return radius * (self.CANVAS_H * 0.5) / (
            math.tan(math.radians(FOVY) / 2.0) * camdist)

    def activate(self, bid):
        if bid == "play":
            self.paused = not self.paused
        elif bid == "rewind":
            self.sim_time -= datetime.timedelta(days=1)
            self._scene_dirty = True
        elif bid == "forward":
            self.sim_time += datetime.timedelta(days=1)
            self._scene_dirty = True
        elif bid == "slower":
            self.speed_index = max(0, self.speed_index - 1)
            self.realtime = False
        elif bid == "faster":
            self.speed_index = min(len(SPEED_DAYS) - 1,
                                   self.speed_index + 1)
            self.realtime = False
        elif bid == "today":
            self.sim_time = utc_now()
            self._scene_dirty = True
        elif bid == "realtime":
            self.realtime = not self.realtime
        elif bid == "worlds":
            self.show_extras = not self.show_extras
            self._scene_dirty = True
        elif bid == "labels":
            self.show_labels = not self.show_labels
        elif bid == "distance":
            self.show_distance = not self.show_distance
            self.show_about = False
        elif bid == "daynight":
            self.show_daynight = not self.show_daynight
            self._scene_dirty = True
        elif bid == "lightmap":
            self.toggle_view()
        elif bid == "about":
            self.show_about = not self.show_about
            self.show_distance = False
        elif bid == "fullscreen":
            self.toggle_fullscreen()
        elif bid == "fit":
            self.reset_view()

    def scrub(self, direction, dt):
        """Nudge the clock while an arrow key is held (direction is +/-1).

        In realtime mode this scrubs one hour of sim time per real second;
        at preset speeds it runs twice the current days-per-second rate."""
        if self.realtime:
            self.sim_time += datetime.timedelta(hours=direction * dt)
        else:
            self.sim_time += datetime.timedelta(
                days=direction * SPEED_DAYS[self.speed_index] * dt * 2)
        self._scene_dirty = True

    def toggle_view(self):
        """Switch between the 3D scene and the 2D daylight map."""
        if self.view_mode == "3d":
            self.view_mode = "daylight"
        else:
            self.view_mode = "3d"
            self._scene_dirty = True

    def in_scene_view(self, pos):
        """True when a window coordinate sits on the interactive 3D canvas.

        Excludes the taskbar, the info-panel overlay on the right, and the
        daylight-map view - clicks, drags, wheel and hover must never act on
        worlds hidden underneath those."""
        x, y = pos
        return (self.view_mode == "3d" and 0 <= x < self.VIEW_W
                and 0 <= y < self.CANVAS_H)

    def button_at(self, mx, my):
        for b in self.buttons:
            if b["rect"].collidepoint(mx, my):
                return b
        return None

    # ------------------------------------------------------- daylight 2-D map
    def _dl_subsolar(self):
        """Sub-solar point at self.sim_time: (lon, decl) in radians.

        The longitude comes from the Sun's apparent right ascension versus
        Greenwich mean sidereal time, so the equation of time is included
        (the old mean-sun shortcut drifted up to ~4 degrees over the year);
        the declination uses the true solar declination rather than a sine
        of the day-of-year."""
        ra, decl = sun_ra_decl(self.sim_time)
        lon = (ra - gmst_deg(self.sim_time) + 180.0) % 360.0 - 180.0
        return math.radians(lon), math.radians(decl)

    def _dl_base_maps(self):
        if self._dl_base is None:
            w, h = DL_MAP_W, DL_MAP_H

            def load(fname, fallback):
                path = os.path.join(TEX_DIR, fname)
                if os.path.exists(path):
                    try:
                        return pygame.image.load(path).convert()
                    except Exception:
                        pass
                return fallback

            day_fb = procedural_texture("Earth")
            night_fb = procedural_night()
            cloud_fb = pygame.Surface((w, h))
            cloud_fb.fill((150, 150, 150))
            base = {}
            for key, fname, fb in (
                    ("day", "2k_earth_daymap.jpg", day_fb),
                    ("night", "2k_earth_nightmap.jpg", night_fb),
                    ("cloud", "2k_earth_clouds.jpg", cloud_fb)):
                img = pygame.transform.smoothscale(
                    load(fname, fb), (w, h))
                arr = np.frombuffer(pygame.image.tostring(img, "RGB", False),
                                    dtype=np.uint8).reshape(h, w, 3)
                base[key] = arr.astype(np.float32) / 255.0
            lon = np.linspace(-math.pi, math.pi, w, endpoint=False)
            lat = np.linspace(math.pi / 2.0, -math.pi / 2.0, h)
            base["lon"] = lon[None, :]
            base["sinlat"] = np.sin(lat)[:, None]
            base["coslat"] = np.cos(lat)[:, None]
            self._dl_base = base
        return self._dl_base

    def daylight_surface(self):
        """Build (or refresh) the 2-D equirectangular daylight map."""
        now = pygame.time.get_ticks()
        st = self.sim_time
        key = (st.year, st.month, st.day, st.hour * 60 + st.minute)
        if self._dl_surf is None or (key != self._dl_key and
                                     now - self._dl_t0 > 120):
            self._dl_t0 = now
            self._dl_key = key
            base = self._dl_base_maps()
            w, h = DL_MAP_W, DL_MAP_H
            lon_s, decl = self._dl_subsolar()
            sin_a = (base["sinlat"] * math.sin(decl) +
                     base["coslat"] * math.cos(decl) *
                     np.cos(base["lon"] - lon_s))
            t = np.clip((sin_a + 0.05) / 0.16, 0.0, 1.0)
            t = t * t * (3.0 - 2.0 * t)
            day = base["day"] * (0.80 + 0.30 * base["cloud"])
            night = base["night"] * 2.1
            arr = day * t[..., None] + night * (1.0 - t[..., None])
            buf = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
            self._dl_surf = pygame.image.frombuffer(
                buf.tobytes(), (w, h), "RGB").convert()
        return self._dl_surf

    def _dl_fit_rect(self):
        scale = min(self.VIEW_W / float(DL_MAP_W),
                    self.CANVAS_H / float(DL_MAP_H))
        dw = max(2, int(DL_MAP_W * scale))
        dh = max(1, int(DL_MAP_H * scale))
        return pygame.Rect((self.VIEW_W - dw) // 2, (self.CANVAS_H - dh) // 2,
                           dw, dh)

    def _dl_terminator(self, fit, lon_s, decl, sign):
        pts = []
        cur = []
        for j in range(DL_MAP_H):
            lat = math.pi / 2.0 - math.pi * j / DL_MAP_H
            v = math.tan(lat) * math.tan(decl)
            if abs(v) < 0.999:
                dl = math.acos(-v)
                lon = (lon_s + sign * dl + math.pi) % math.tau - math.pi
                cur.append((fit.x + (lon + math.pi) / math.tau * fit.w,
                            fit.y + j / float(DL_MAP_H) * fit.h))
            else:
                if len(cur) >= 2:
                    pts.append(cur)
                cur = []
        if len(cur) >= 2:
            pts.append(cur)
        return pts

    def _dl_lon_label(self, lon):
        d = round(abs(math.degrees(lon)))
        if d == 180:
            return "180°"
        return "%d°%s" % (d, "E" if lon >= 0 else "W")

    def draw_daylight_map(self):
        fit = self._dl_fit_rect()
        surf = self.daylight_surface()
        if fit.size != surf.get_size():
            surf = pygame.transform.smoothscale(surf, fit.size)
        self.screen.fill(C_BG)
        self.screen.blit(surf, fit)
        pygame.draw.rect(self.screen, (56, 84, 150), fit, 2)

        lon_s, decl = self._dl_subsolar()

        gcol = (70, 96, 140)
        for lon in range(-180, 181, 30):
            x = fit.x + (lon + 180.0) / 360.0 * fit.w
            pygame.draw.line(self.screen, gcol,
                             (int(x), fit.y), (int(x), fit.bottom), 1)
        for lat in range(-60, 61, 30):
            y = fit.y + (90.0 - lat) / 180.0 * fit.h
            pygame.draw.line(self.screen, gcol,
                             (fit.x, int(y)), (fit.right, int(y)), 1)

        for seg in self._dl_terminator(fit, lon_s, decl, +1):
            pygame.draw.lines(self.screen, C_CYAN, False, seg, 2)
        for seg in self._dl_terminator(fit, lon_s, decl, -1):
            pygame.draw.lines(self.screen, C_CYAN, False, seg, 2)

        sx = fit.x + (lon_s + math.pi) / math.tau * fit.w
        sy = fit.y + (math.pi / 2.0 - decl) / math.pi * fit.h
        sx, sy = int(sx), int(sy)
        pygame.draw.circle(self.screen, (255, 190, 90), (sx, sy), 14, 3)
        pygame.draw.circle(self.screen, (255, 236, 170), (sx, sy), 6)
        for a in range(8):
            th = a * math.tau / 8
            pygame.draw.line(self.screen, (255, 205, 110),
                             (sx + int(math.cos(th) * 18),
                              sy + int(math.sin(th) * 18)),
                             (sx + int(math.cos(th) * 26),
                              sy + int(math.sin(th) * 26)), 2)

        pill = pygame.Surface((330, 118), pygame.SRCALPHA)
        pygame.draw.rect(pill, (4, 8, 20, 190),
                         (0, 0, 330, 118), border_radius=10)
        pygame.draw.rect(pill, (0, 232, 255, 80),
                         (0, 0, 330, 118), 1, border_radius=10)
        title = self.neon("EARTH DAYLIGHT MAP", self.font_title, C_CYAN)
        pill.blit(title, (12, 4))
        date = self.font_small.render(
            self.sim_time.strftime("%A, %d %B %Y  %H:%M UTC"), True, C_AMBER)
        pill.blit(date, (14, 50))
        sub = self.font_small.render(
            "Sun is directly over " + self._dl_lon_label(lon_s), True, C_TEXT)
        pill.blit(sub, (14, 76))
        sub2 = self.font_tiny.render(
            "cyan lines = sunrise / sunset  |  yellow dot = high noon",
            True, C_DIM)
        pill.blit(sub2, (14, 100))
        self.screen.blit(pill, (12, 10))

        tip = self.cached(self.tip_cache, "dl-back",
                          self.font_tiny,
                          "Press G or the map button to return to the 3D view",
                          C_DIM)
        self.screen.blit(tip, (24, self.CANVAS_H - 26))

    # --------------------------------------------------------------- drawing
    def draw_hud(self):
        col = (0, 190, 255)
        L = 24
        W, CH = self.W, self.CANVAS_H
        pts = [
            [(8, 8), (8 + L, 8), (8, 8)],
            [(8 + L, 8), (8, 8), (8, 8 + L)],
            [(W - 8, 8), (W - 8 - L, 8), (W - 8, 8)],
            [(W - 8 - L, 8), (W - 8, 8), (W - 8, 8 + L)],
            [(8, CH - 8), (8 + L, CH - 8), (8, CH - 8)],
            [(8 + L, CH - 8), (8, CH - 8), (8, CH - 8 - L)],
        ]
        for (ax, ay), (bx, by), (cx, cy) in pts:
            pygame.draw.line(self.screen, col, (ax, ay), (bx, by), 2)
            pygame.draw.line(self.screen, col, (bx, by), (cx, cy), 2)

    def draw_labels(self):
        if not self.show_labels:
            return
        for n, pos in self.world.items():
            sp = self.to_screen(pos)
            if sp is None:
                continue
            px, py = sp
            if px < 4 or px > self.W - 4 or py < 2 or py > self.CANVAS_H - 4:
                continue
            if n == "Sun" and self.cam.dist > 40:
                continue
            color = COLORS.get(n, C_TEXT)
            surf = self.cached(self.label_cache, n, self.font_label,
                               NAME_TITLES.get(n, n), color)
            rect = surf.get_rect(center=(px, py))
            rect.clamp_ip(pygame.Rect(0, 0, self.W, self.CANVAS_H))
            self.screen.blit(surf, rect)

    def draw_selection_rings(self):
        for name, col in ((self.hovered, C_CYAN), (self.pinned, C_MAGENTA)):
            if not name or name not in self.world:
                continue
            sp = self.to_screen(self.world[name])
            if sp is None:
                continue
            px, py = sp
            r = max(4, int(self.screen_radius(
                self.world[name], R_3D.get(name, 1.0))))
            pygame.draw.circle(self.screen, col, (int(px), int(py)),
                               r + 6, 2)

    def draw_distance(self):
        if not self.show_distance:
            return
        sun = self.to_screen(self.world["Sun"])
        if sun is None:
            return
        sx, sy = int(sun[0]), int(sun[1])
        for n in self.visible_names():
            sp = self.to_screen(self.world[n])
            if sp is None:
                continue
            px, py = int(sp[0]), int(sp[1])
            dist_au = self.body_r.get(n, 0.0)
            dist_km = dist_au * AU_KM
            pygame.draw.line(self.screen, (60, 120, 200), (sx, sy), (px, py), 1)
            mx, my = (sx + px) // 2, (sy + py) // 2
            if dist_au >= 1.0:
                txt = "%.2f AU" % dist_au
            else:
                txt = "%.4f AU" % dist_au
            if dist_km >= 1e6:
                txt += "  (%.0fM km)" % (dist_km / 1e6)
            elif dist_km >= 1000:
                txt += "  (%.0fK km)" % (dist_km / 1000)
            else:
                txt += "  (%.0f km)" % dist_km
            lbl = self.font_tiny.render(txt, True, (100, 180, 255))
            self.screen.blit(lbl, (mx - lbl.get_width() // 2,
                                   my - lbl.get_height() // 2))
        # moon line
        est = self.to_screen(self.world["Earth"])
        mst = self.to_screen(self.world["Moon"])
        if est is not None and mst is not None:
            ex, ey = int(est[0]), int(est[1])
            mx, my = int(mst[0]), int(mst[1])
            pygame.draw.line(self.screen, (100, 160, 220), (ex, ey), (mx, my), 1)
            mmx, mmy = (ex + mx) // 2, (ey + my) // 2
            mtxt = "%.4f AU (%.0f km)" % (self.moon_r_au,
                                           self.moon_r_au * AU_KM)
            mlbl = self.font_tiny.render(mtxt, True, (140, 200, 255))
            self.screen.blit(mlbl, (mmx - mlbl.get_width() // 2,
                                    mmy - mlbl.get_height() // 2))

    def draw_about(self):
        if not self.show_about:
            return
        w = min(self.W - 20, 560)
        h = min(self.CANVAS_H - 20, 420)
        w = max(320, w)
        h = max(260, h)
        x = (self.W - w) // 2
        y = (self.CANVAS_H - h) // 2
        box = pygame.Surface((w, h), pygame.SRCALPHA)
        box.fill((10, 16, 40, 220))
        pygame.draw.rect(box, C_CYAN, (0, 0, w, h), 2, border_radius=12)
        pad = 28
        ty = pad
        title = self.font_title.render("Tiny Solar System 3D", True, C_CYAN)
        box.blit(title, (pad, ty))
        ty += 42
        sub = self.font_small.render("A Solar System Explorer for Kids",
                                     True, C_AMBER)
        box.blit(sub, (pad, ty))
        ty += 30
        pygame.draw.line(box, C_CYAN_DIM, (pad, ty), (w - pad, ty), 1)
        ty += 14
        story = (
            "Real NASA maps from Solar System Scope (CC-BY) on every world, "
            "rendered as 3D spheres with ModernGL. Drag to spin the camera, "
            "use the wheel to zoom towards whatever your pointer is over "
            "(or the world you pinned), and press N to turn Earth's city "
            "lights on and off. Real positions every moment from the "
            "solarsystem library - orbits are stylised so everyone stays "
            "visible. Resize the window or press F11 for fullscreen."
        )
        for line in self.wrap_text(story, self.font_body, w - pad * 2):
            box.blit(self.font_body.render(line, True, C_TEXT), (pad, ty))
            ty += 24
        ty += 10
        pygame.draw.line(box, C_CYAN_DIM, (pad, ty), (w - pad, ty), 1)
        ty += 12
        credit = self.font_body.render("Coded with love by Amir for Aiman",
                                       True, C_GREEN)
        box.blit(credit, (pad, ty))
        ty += 30
        close = self.font_small.render(
            "Press I or click About again to close", True, C_DIM)
        box.blit(close, (pad, ty))
        self.screen.blit(box, (x, y))

    # ------------------------------------------------------------- taskbar
    def draw_taskbar(self):
        self.screen.blit(self.taskbar_grad, (0, self.CANVAS_H))
        pygame.draw.line(self.screen, C_TASKBAR_EDGE, (0, self.CANVAS_H),
                         (self.W, self.CANVAS_H))

        led_color = C_GREEN if not self.paused else C_AMBER
        blink = (0.5 + 0.5 * math.sin(pygame.time.get_ticks() * 0.006)) \
            if self.paused else 1.0
        lx, ly = 16, self.CANVAS_H + self.BAR_H // 2
        pygame.draw.circle(self.screen, lerp_color((10, 14, 30), led_color, 0.35),
                           (lx, ly), 7)
        pygame.draw.circle(self.screen, lerp_color((10, 14, 30), led_color, blink),
                           (lx, ly), 5)
        led = self.font_tiny.render("LIVE" if not self.paused else "HOLD",
                                    True, led_color)
        self.screen.blit(led, (lx + 14, ly - led.get_height() // 2))

        mouse = pygame.mouse.get_pos()
        for b in self.buttons:
            self.draw_button(b, mouse)

        tx = self.W - 14
        date = self.sim_time.strftime("%A, %d %B %Y")
        if date != self.date_key:
            self.date_key = date
            self.date_surf = self.font_section.render(date, True, C_TEXT)
        self.screen.blit(self.date_surf,
                         (tx - self.date_surf.get_width(), self.CANVAS_H + 8))

        static_key = (self.speed_index, self.show_extras, self.realtime)
        if static_key != self.status_static_key:
            self.status_static_key = static_key
            label = ("REALTIME 1x" if self.realtime else
                     "SPEED " + SPEED_LABEL[SPEED_DAYS[self.speed_index]])
            parts = [label]
            if self.show_extras:
                parts.append("ALL WORLDS")
            self.status_static = self.font_small.render(
                "   |   ".join(parts) + "   |   ", True, C_CYAN)
            self.zoom_slot = self.font_small.size("ZOOM 4000%")[0]
        y = self.CANVAS_H + 40
        zoom_x = tx - self.zoom_slot
        self.screen.blit(self.status_static,
                         (zoom_x - self.status_static.get_width(), y))

        zoom_pct = round(DIST_HOME / self.cam.dist * 100)
        zoom_img = self.font_small.render(
            "ZOOM %d%%" % zoom_pct, True, C_CYAN)
        self.screen.blit(zoom_img,
                         (zoom_x + self.zoom_slot - zoom_img.get_width(), y))

    def draw_button(self, b, mouse):
        rect = b["rect"]
        over = rect.collidepoint(mouse)
        active = (b["id"] == "play" and self.paused) or \
                 (b["id"] == "worlds" and self.show_extras) or \
                 (b["id"] == "labels" and self.show_labels) or \
                 (b["id"] == "distance" and self.show_distance) or \
                 (b["id"] == "daynight" and self.show_daynight) or \
                 (b["id"] == "lightmap" and self.view_mode == "daylight") or \
                 (b["id"] == "realtime" and self.realtime) or \
                 (b["id"] == "about" and self.show_about) or \
                 (b["id"] == "fit" and self.pinned is not None)
        pressed = (self.pressed_btn == b["id"] and over and
                   pygame.mouse.get_pressed()[0])

        if over or active:
            self.screen.blit(self.button_glow, (rect.x - 7, rect.y - 7))

        if pressed:
            bg, border = (8, 12, 26), C_CYAN
        elif over:
            bg, border = (26, 36, 76), C_CYAN
        elif active:
            bg, border = (22, 30, 68), C_CYAN
        else:
            bg, border = (15, 20, 44), (62, 74, 122)

        offset = 2 if pressed else 0
        r = rect.move(0, offset)
        pygame.draw.rect(self.screen, bg, r, border_radius=BTN_RADIUS)
        pygame.draw.rect(self.screen, border, r, 2, border_radius=BTN_RADIUS)
        pygame.draw.line(self.screen, (120, 160, 220),
                         (r.x + 5, r.y + 3), (r.right - 5, r.y + 3))

        if b["id"] == "play":
            icon = self.icons["pause"] if not self.paused else self.icons["play"]
        else:
            icon = self.icons[b["icon"]]
        icon_rect = icon.get_rect(center=r.center)
        self.screen.blit(icon, icon_rect)

    def draw_tooltip(self):
        b = self.button_at(*pygame.mouse.get_pos())
        if not b:
            return
        tip = self.cached(self.tip_cache, b["tip"], self.font_tooltip,
                          b["tip"], C_TEXT)
        pad = 10
        pill = pygame.Rect(0, 0, tip.get_width() + pad * 2,
                           tip.get_height() + 8)
        pill.centerx = b["rect"].centerx
        pill.bottom = b["rect"].top - 6
        if pill.left < 6:
            pill.left = 6
        if pill.right > self.W - 6:
            pill.right = self.W - 6
        pygame.draw.rect(self.screen, (10, 14, 36), pill, border_radius=8)
        pygame.draw.rect(self.screen, C_CYAN_DIM, pill, 1, border_radius=8)
        self.screen.blit(tip, tip.get_rect(center=pill.center))

    # --------------------------------------------------------------- panel
    def draw_panel(self):
        surf = self.panel_surface
        surf.fill(PANEL_FILL)
        pygame.draw.line(surf, C_CYAN, (1, 0), (1, self.CANVAS_H), 2)
        pygame.draw.line(surf, (0, 232, 255, 60), (0, 0), (0, self.CANVAS_H))

        surf.blit(self.title_surf, (20, 18))
        sub = self.font_small.render("A little map of space", True, C_DIM)
        surf.blit(sub, (24, 54))
        surf.blit(self.divider, (24, 84))

        focus = self.pinned or self.hovered
        if focus:
            self._panel_body(surf, focus)
        else:
            self._panel_idle(surf)

        surf.blit(self.scanlines, (0, 0))
        self.screen.blit(surf, (self.W - self.PANEL_W, 0))

    def _panel_idle(self, surf):
        pad = 24
        y = 108
        tag = self.font_tiny.render("// STATUS", True, C_CYAN_DIM)
        surf.blit(tag, (pad, y))
        y += 28
        status = "SIMULATION PAUSED" if self.paused else "SIMULATION RUNNING"
        s = self.cached(self.tip_cache, "status-" + status, self.font_section,
                        status, C_AMBER if self.paused else C_GREEN)
        surf.blit(s, (pad, y))
        y += 56

        hello = self.font_body.render("Hello, space explorer!", True, C_TEXT)
        surf.blit(hello, (pad, y))
        y += 30
        for line in self.wrap_text(
                "Move time forward and watch the planets and the Moon "
                "travel around the Sun.", self.font_body,
                self.PANEL_W - pad * 2):
            surf.blit(self.font_body.render(line, True, C_DIM), (pad, y))
            y += 27

        y += 16
        weekday = self.sim_time.strftime("%A")
        date_img = self.cached(self.tip_cache, "wk-" + weekday,
                               self.font_big, weekday, C_AMBER)
        surf.blit(date_img, (pad, y))
        y += 38
        day_month = self.sim_time.strftime("%d %B %Y")
        dm_img = self.cached(self.tip_cache, "dm-" + day_month,
                             self.font_body, day_month, C_TEXT)
        time_img = self.font_body.render(
            self.sim_time.strftime("%H:%M:%S UTC"), True, C_TEXT)
        surf.blit(dm_img, (pad, y))
        surf.blit(time_img, (pad, y + 28))
        y += 56

        y += 10
        surf.blit(self.divider, (pad, y))
        y += 18
        tips = [("WHEEL", "zoom in and out"),
                ("DRAG", "spin the camera"),
                ("R-DRAG", "slide the view"),
                ("CLICK", "a world to pin it"),
                ("N", "Earth day / night")]
        for k, v in tips:
            k_img = self.cached(self.tip_cache, "tip-" + k, self.font_small,
                                k, C_CYAN)
            v_img = self.cached(self.tip_cache, "tipv-" + v, self.font_small,
                                v, C_DIM)
            surf.blit(k_img, (pad, y))
            surf.blit(v_img, (pad + 86, y))
            y += 24

    def _panel_body(self, surf, name):
        pad = 24
        y = 108

        name_surf = self.name_cache.get(name)
        if name_surf is None:
            name_surf = self.neon(NAME_TITLES.get(name, name), self.font_big,
                                  COLORS.get(name, C_TEXT))
            self.name_cache[name] = name_surf
        surf.blit(name_surf, (pad - 4, y - 4))
        cat = self.cached(self.cat_cache, name, self.font_small,
                          CATEGORY.get(name, ""), C_CYAN_DIM)
        surf.blit(cat, (pad, y + 36))

        if self.pinned == name:
            pin = self.cached(self.tip_cache, "pinned", self.font_tiny,
                              "PINNED  -  click empty space to close",
                              C_MAGENTA)
            surf.blit(pin, (pad, y + 58))

        y += 98
        fact_surfs = self.fact_cache.get(name)
        if fact_surfs is None:
            fact_surfs = [self.font_body.render(line, True, (236, 240, 255))
                          for line in self.wrap_text(FACTS[name], self.font_body,
                                                     self.PANEL_W - pad * 2)]
            self.fact_cache[name] = fact_surfs
        for fs in fact_surfs:
            surf.blit(fs, (pad, y))
            y += 27

        y += 12
        surf.blit(self.divider, (pad, y))
        y += 18

        if name == "Sun":
            stats = [("DISTANCE FROM SUN", "0.0 AU - it is the center!"),
                     ("ORBIT", "everything goes around it!")]
        elif name == "Moon":
            phase = self._moon_phase()
            ph = ("%.0f%% lit" % (phase * 100)) if phase is not None else "n/a"
            stats = [("DISTANCE FROM EARTH",
                      "%.4f AU   (%d km)" % (self.moon_r_au,
                                             round(self.moon_r_au * AU_KM))),
                     ("ORBIT TIME", period_text(name)),
                     ("ILLUMINATION", ph)]
        else:
            earth = self.positions.get("Earth")
            ex, ey = 0.0, 0.0
            if earth is not None:
                ex, ey = earth[0], earth[1]
            px, py, pz = self.positions.get(name, (0, 0, 0))
            d_earth = math.sqrt((px - ex) ** 2 + (py - ey) ** 2 + pz ** 2)
            stats = [("DISTANCE FROM SUN", "%.2f AU" % self.body_r.get(name, 0)),
                     ("DISTANCE FROM EARTH", "%.2f AU" % d_earth),
                     ("ORBIT TIME", period_text(name))]

        for label, value in stats:
            lbl = self.font_tiny.render(label, True, C_CYAN_DIM)
            surf.blit(lbl, (pad, y))
            y += 24
            val = self.font_body.render(value, True, C_TEXT)
            surf.blit(val, (pad, y))
            y += 34

    def _minute_bucket(self, t):
        """Timezone-independent minute index for a naive-UTC datetime.

        datetime.timestamp() interprets naive datetimes as *local* time, so
        the bucket used to shift by the machine's UTC offset."""
        return int((t - _UNIX_EPOCH).total_seconds()) // 60

    def _moon_phase(self):
        key = self._minute_bucket(self.sim_time)
        if key != self._moon_phase_key:
            self._moon_phase_key = key
            t = self.sim_time
            try:
                m = solarsystem.Moon(
                    year=t.year, month=t.month, day=t.day,
                    hour=t.hour, minute=t.minute,
                    UT=0, dst=0, longtitude=0.0, latitude=51.48,
                    topographic=False)
                self._moon_phase_val = m.phase()
            except Exception:
                self._moon_phase_val = None
        return self._moon_phase_val

    # ----------------------------------------------------------------- loop
    def run(self):
        running = True
        dt = 0.0
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    self._handle_resize(event.w, event.h)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if self.view_mode == "daylight":
                            self.toggle_view()
                        elif self.fullscreen:
                            self.toggle_fullscreen()
                        else:
                            running = False
                    elif event.key == pygame.K_SPACE:
                        self.activate("play")
                    elif event.key in (pygame.K_UP, pygame.K_EQUALS,
                                       pygame.K_PLUS):
                        self.activate("faster")
                    elif event.key in (pygame.K_DOWN, pygame.K_MINUS):
                        self.activate("slower")
                    elif event.key == pygame.K_t:
                        self.activate("realtime")
                    elif event.key == pygame.K_d:
                        self.activate("worlds")
                    elif event.key == pygame.K_l:
                        self.activate("labels")
                    elif event.key == pygame.K_m:
                        self.activate("distance")
                    elif event.key == pygame.K_n:
                        self.activate("daynight")
                    elif event.key == pygame.K_g:
                        self.activate("lightmap")
                    elif event.key == pygame.K_i:
                        self.activate("about")
                    elif event.key == pygame.K_r:
                        self.activate("today")
                    elif event.key == pygame.K_HOME:
                        self.activate("fit")
                    elif event.key == pygame.K_F11:
                        self.toggle_fullscreen()
                elif event.type == pygame.MOUSEWHEEL:
                    mpos = pygame.mouse.get_pos()
                    if self.in_scene_view(mpos):
                        factor = 1.15 ** -event.y
                        self.cam.zoom_to(
                            mpos[0], mpos[1], factor, self.W, self.CANVAS_H,
                            move_target=(self.pinned is None))
                        self._scene_dirty = True
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.drag_moved = False
                    if event.button == 1:
                        b = self.button_at(*event.pos)
                        if b is not None:
                            self.pressed_btn = b["id"]
                            self.held_btn = b["id"]
                            self.held_start = pygame.time.get_ticks()
                            self.held_last = self.held_start
                        elif self.in_scene_view(event.pos):
                            self.dragging = True
                            self.drag_mode = "orbit"
                            self.drag_start = event.pos
                    elif event.button in (2, 3) and \
                            self.in_scene_view(event.pos):
                        self.dragging = True
                        self.drag_mode = "pan"
                        self.drag_start = event.pos
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        if self.pressed_btn is not None:
                            b = self.button_at(*event.pos)
                            if b is not None and b["id"] == self.pressed_btn:
                                self.activate(self.pressed_btn)
                        elif not self.drag_moved and \
                                self.in_scene_view(event.pos):
                            picked = self.pick(*event.pos, self._pick_bodies)
                            new_pin = None if picked == self.pinned else picked
                            if new_pin != self.pinned:
                                self.pinned = new_pin
                                self._scene_dirty = True
                        self.pressed_btn = None
                        self.held_btn = None
                        self.dragging = False
                        self.drag_mode = None
                    elif event.button in (2, 3):
                        self.dragging = False
                        self.drag_mode = None
                    self.drag_moved = False
                elif event.type == pygame.MOUSEMOTION:
                    if self.dragging:
                        dx = event.pos[0] - self.drag_start[0]
                        dy = event.pos[1] - self.drag_start[1]
                        if abs(dx) + abs(dy) > 4:
                            self.drag_moved = True
                        if self.drag_mode == "orbit":
                            self.cam.orbit(dx, dy)
                        elif self.drag_mode == "pan":
                            self.cam.pan(dx, dy, self.CANVAS_H)
                        if self.drag_mode is not None:
                            self._scene_dirty = True
                        self.drag_start = event.pos

            if self.held_btn in REPEAT_BTNS and pygame.mouse.get_pressed()[0]:
                b = self.button_at(*pygame.mouse.get_pos())
                now = pygame.time.get_ticks()
                if b is not None and b["id"] == self.held_btn and \
                        now - self.held_start > 350 and now - self.held_last >= 140:
                    self.activate(self.held_btn)
                    self.held_last = now
            elif self.held_btn and not pygame.mouse.get_pressed()[0]:
                self.held_btn = None

            keys = pygame.key.get_pressed()
            if not self.paused:
                if self.realtime:
                    self.sim_time += datetime.timedelta(seconds=dt)
                else:
                    self.sim_time += datetime.timedelta(
                        days=SPEED_DAYS[self.speed_index] * dt)
            if keys[pygame.K_LEFT]:
                self.scrub(-1, dt)
            if keys[pygame.K_RIGHT]:
                self.scrub(1, dt)

            self.update_world(dt)

            if self.pinned is not None:
                target = self.world.get(self.pinned)
                if target is not None:
                    self.cam.target = np.array(target, dtype=np.float32)

            if self.view_mode == "daylight":
                self.draw_daylight_map()
                self.hovered = None
            else:
                # Re-render the GL scene only when something can have changed;
                # while paused and untouched, re-blit the cached frame.
                if self._scene_dirty or not self.paused or \
                        self._frame_surf is None:
                    bodies = self.build_bodies()
                    data = render(self.ctx, self.scene, self.cam, bodies,
                                  sun_glow=SUN_GLOW)
                    self._frame_surf = to_surface(data, self.scene.vw,
                                                  self.scene.vh)
                    self._pick_bodies = bodies
                    self._view = self.cam.view_matrix()
                    self._proj = self.cam.proj_matrix()
                    self._scene_dirty = False
                self.screen.blit(self._frame_surf, (0, 0))
                mpos = pygame.mouse.get_pos()
                if self.in_scene_view(mpos):
                    self.hovered = self.pick(mpos[0], mpos[1],
                                             self._pick_bodies)
                else:
                    self.hovered = None
                self.draw_hud()
                self.draw_selection_rings()
                self.draw_labels()
                self.draw_distance()

            want = pygame.SYSTEM_CURSOR_ARROW
            if self.dragging:
                want = pygame.SYSTEM_CURSOR_SIZEALL
            elif self.button_at(*pygame.mouse.get_pos()) or self.hovered:
                want = pygame.SYSTEM_CURSOR_HAND
            if want != self.cursor:
                self.cursor = want
                try:
                    pygame.mouse.set_cursor(want)
                except pygame.error:
                    pass

            self.draw_about()
            self.draw_panel()
            self.draw_taskbar()
            self.draw_tooltip()

            pygame.display.flip()
            dt = self.clock.tick(FPS) / 1000.0

        pygame.quit()


def snap_display(buf):
    """Copy the display surface (after a flip) to a plain surface whose raw
    pixels are reliable to export.  The display read-back comes back rotated
    180 degrees, so the copy is flipped both ways (sideways + upside) again
    to land upright in the saved image."""
    pygame.display.flip()
    out = pygame.Surface(buf.get_size())
    out.blit(buf, (0, 0))
    return pygame.transform.flip(out, True, True)


def save_png(surface, path):
    """Write a pygame surface as PNG without relying on pygame.image.save
    (some pygame builds cannot save image files)."""
    w, h = surface.get_size()
    rgba = surface.get_bytesize() == 4
    px = pygame.image.tostring(surface, "RGBA" if rgba else "RGB", True)
    bpp = 4 if rgba else 3
    stride = w * bpp
    parts = []
    for y in range(h):
        parts.append(b"\x00")
        parts.append(px[y * stride:(y + 1) * stride])
    raw = b"".join(parts)
    color_type = 6 if rgba else 2

    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + \
            struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0)
    out = b"\x89PNG\r\n\x1a\n"
    out += chunk(b"IHDR", ihdr)
    out += chunk(b"IDAT", zlib.compress(bytes(raw), 9))
    out += chunk(b"IEND", b"")
    with open(path, "wb") as fh:
        fh.write(out)


def save_shot(path, w=1600, h=1000):
    """Headless single-frame render (for batch shots / CI checks)."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    app = SolarSystemApp()
    app.W, app.H = w, h
    app._rebuild_for_size(center=True)
    app.update_world(0.0)
    bodies = app.build_bodies()
    data = render(app.ctx, app.scene, app.cam, bodies, sun_glow=SUN_GLOW)
    surf = to_surface(data, app.scene.vw, app.scene.vh)
    save_png(surf, path)
    pygame.quit()
    return path


def save_lightmap(path, w=1600, h=1000):
    """Headless daylight-map frame (for batch shots / CI checks)."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    app = SolarSystemApp()
    app.W, app.H = w, h
    app._rebuild_for_size(center=True)
    app.screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
    app.view_mode = "daylight"
    app.update_world(0.0)
    app.draw_daylight_map()
    app.draw_panel()
    app.draw_taskbar()
    save_png(snap_display(app.screen), path)
    pygame.quit()
    return path


USAGE = ("usage: python solar_system.py\n"
         "       python solar_system.py --shot OUTPUT.png\n"
         "       python solar_system.py --lightmap OUTPUT.png")


def main(argv):
    if not argv:
        SolarSystemApp().run()
    elif len(argv) == 2 and argv[0] == "--shot":
        save_shot(argv[1])
    elif len(argv) == 2 and argv[0] == "--lightmap":
        save_lightmap(argv[1])
    else:
        try:
            print(USAGE, file=sys.stderr)
        except Exception:
            pass
        os._exit(2)


if __name__ == "__main__":
    main(sys.argv[1:])