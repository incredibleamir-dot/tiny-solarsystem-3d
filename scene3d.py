"""scene3d.py - the 3D engine for tiny-solarsystem-3d.

Renders the whole Solar System as textured spheres with ModernGL: the Sun as
an emissive sphere with an additive corona, every planet on its orbit ring,
Saturn's ring, the Earth's day/night city lights and a starfield.  The camera
is a free orbit camera that can rotate (drag), pan (right/middle drag) and
zoom (mouse wheel, centred on whatever your pointer is over or the selected
body).  Everything renders into an off-screen FBO so the app can also run
headless for batch shots.

The scene is stylised (not to true scale) so every world stays visible.
"""

import math

import numpy as np
import moderngl

FOVY = 45.0
NEAR, FAR = 0.1, 1200.0

STARFIELD = 900

ORBIT_RINGS_MAX = 12   # ring-buffer capacity (ALWAYS + EXTRAS bodies)
ORBIT_SEGS = 128       # polyline segments per orbit ring
MOON_SEGS = 96         # polyline segments in the Moon's orbit ring

# ---------------------------------------------------------------------------
# Shaders
# ---------------------------------------------------------------------------

SPHERE_VERT = """
#version 330
in vec3 in_position;
in vec3 in_normal;
in vec2 in_uv;
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;
uniform vec3 u_sun_pos;
out vec2 v_uv;
out vec3 v_normal;
out vec3 v_pos;
void main() {
    vec4 world = u_model * vec4(in_position, 1.0);
    gl_Position = u_proj * u_view * world;
    v_uv = in_uv;
    v_normal = mat3(u_model) * in_normal;
    v_pos = world.xyz;
}
"""

SPHERE_FRAG = """
#version 330
in vec2 v_uv;
in vec3 v_normal;
in vec3 v_pos;
uniform sampler2D u_tex;
uniform sampler2D u_tex2;
uniform vec3 u_view_pos;
uniform vec3 u_sun_pos;
uniform float u_emissive;
uniform vec3 u_ambient;
uniform float u_has_night;
out vec4 f_color;
void main() {
    vec3 n = normalize(v_normal);
    vec3 l = normalize(u_sun_pos - v_pos);
    vec3 col = texture(u_tex, v_uv).rgb;
    if (u_emissive > 0.5) {
        f_color = vec4(col * 1.8, 1.0);
        return;
    }
    float diff = max(dot(n, l), 0.0);
    vec3 v = normalize(u_view_pos - v_pos);
    vec3 h = normalize(l + v);
    float spec = pow(max(dot(n, h), 0.0), 22.0) * diff * 0.45;
    vec3 lit = col * (u_ambient + diff) + vec3(spec);
    if (u_has_night > 0.5) {
        vec3 ncol = texture(u_tex2, v_uv).rgb;
        float night = clamp(1.0 - diff * 2.6, 0.0, 1.0);
        vec3 daylit = col * (u_ambient * (1.0 - 0.92 * night) + diff) + vec3(spec);
        vec3 nightlit = ncol * (0.5 + 2.0 * night);
        lit = mix(daylit, nightlit, night);
    }
    f_color = vec4(lit, 1.0);
}
"""

LINE_VERT = """
#version 330
in vec3 in_position;
in vec3 in_color;
uniform mat4 u_view;
uniform mat4 u_proj;
out vec3 v_color;
void main() {
    gl_Position = u_proj * u_view * vec4(in_position, 1.0);
    v_color = in_color;
}
"""

LINE_FRAG = """
#version 330
in vec3 v_color;
out vec4 f_color;
void main() { f_color = vec4(v_color, 1.0); }
"""

QUAD_VERT = """
#version 330
in vec3 in_position;
in vec2 in_uv;
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;
out vec2 v_uv;
void main() {
    gl_Position = u_proj * u_view * u_model * vec4(in_position, 1.0);
    v_uv = in_uv;
}
"""

HALO_FRAG = """
#version 330
in vec2 v_uv;
uniform vec3 u_color;
out vec4 f_color;
void main() {
    vec2 p = v_uv - vec2(0.5);
    float d = length(p);
    float a = smoothstep(0.5, 0.0, d);
    a = pow(a, 3.0);
    f_color = vec4(u_color * a, a);
}
"""

RING_FRAG = """
#version 330
in vec2 v_uv;
uniform sampler2D u_tex;
uniform float u_alpha;
out vec4 f_color;
void main() {
    vec4 t = texture(u_tex, vec2(1.0 - v_uv.y, v_uv.x));
    float a = t.a * u_alpha;
    if (a < 0.02) discard;
    f_color = vec4(t.rgb * a, a);
}
"""

STAR_VERT = """
#version 330
in vec3 in_position;
in vec3 in_color;
in float in_size;
uniform mat4 u_view;
uniform mat4 u_proj;
out vec3 v_color;
void main() {
    gl_Position = u_proj * u_view * vec4(in_position, 1.0);
    gl_PointSize = in_size;
    v_color = in_color;
}
"""

STAR_FRAG = """
#version 330
in vec3 v_color;
out vec4 f_color;
void main() {
    vec2 p = gl_PointCoord - vec2(0.5);
    float d = length(p);
    float a = pow(1.0 - smoothstep(0.0, 0.5, d), 2.5);
    f_color = vec4(v_color * a, a);
}
"""


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def sphere_arrays(stacks=36, sectors=60):
    """Unit-radius sphere; real size is applied by the model scale."""
    positions, normals, uvs, indices = [], [], [], []
    for i in range(stacks + 1):
        phi = math.pi * i / stacks
        for j in range(sectors + 1):
            theta = 2.0 * math.pi * j / sectors
            x = math.sin(phi) * math.cos(theta)
            y = math.cos(phi)
            z = math.sin(phi) * math.sin(theta)
            positions.append((x, y, z))
            normals.append((x, y, z))
            uvs.append((j / sectors, i / stacks))
    for i in range(stacks):
        for j in range(sectors):
            a = i * (sectors + 1) + j
            b = a + sectors + 1
            indices += [a, b, a + 1, a + 1, b, b + 1]
    pos = np.array(positions, dtype=np.float32)
    nrm = np.array(normals, dtype=np.float32)
    uv = np.array(uvs, dtype=np.float32)
    idx = np.array(indices, dtype=np.uint32)
    interleaved = np.hstack([pos, nrm, uv]).ravel()
    return interleaved, idx


def quad_arrays():
    v = np.array([
        (-1, -1, 0, 0, 0), (1, -1, 0, 1, 0),
        (1, 1, 0, 1, 1), (-1, 1, 0, 0, 1),
    ], dtype=np.float32)
    idx = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
    return v.ravel(), idx


def flat_ring_arrays(rim, rom, segs=160):
    """A flat ring in the XZ plane (radius rim -> rom).

    uv: x = azimuth around the ring, y = radius (0 outer, 1 inner); the
    shader samples the strip texture as vec2(1.0 - v_uv.y, v_uv.x) because
    the strip's WIDTH is the radial profile (inner on the left) and its
    HEIGHT is the azimuth (all rows are identical)."""
    positions, uvs, indices = [], [], []
    for i in range(segs + 1):
        a = math.radians(360.0 * i / segs)
        ca, sa = math.cos(a), math.sin(a)
        positions.append((rom * ca, 0.0, rom * sa))
        positions.append((rim * ca, 0.0, rim * sa))
        uvs.append((i / segs, 0.0))
        uvs.append((i / segs, 1.0))
    for i in range(segs):
        j = i * 2
        indices += [j, j + 1, j + 2, j + 1, j + 3, j + 2]
    pos = np.array(positions, dtype=np.float32)
    uv = np.array(uvs, dtype=np.float32)
    idx = np.array(indices, dtype=np.uint32)
    return np.hstack([pos, uv]).ravel(), idx


def circle_line_arrays(rings):
    """rings: list of (radius, r, g, b) -> one GL_LINES buffer of polylines."""
    pts = []
    for radius, r, g, b in rings:
        for i in range(128):
            a0 = math.radians(360.0 * i / 128)
            a1 = math.radians(360.0 * (i + 1) / 128)
            c0, s0 = math.cos(a0), math.sin(a0)
            c1, s1 = math.cos(a1), math.sin(a1)
            pts.append((c0 * radius, 0.0, s0 * radius, r, g, b))
            pts.append((c1 * radius, 0.0, s1 * radius, r, g, b))
    return np.array(pts, dtype=np.float32).ravel()


def surface_to_gl(surf):
    import pygame
    flip = pygame.transform.flip(surf, False, True)
    if flip.get_bytesize() == 4:
        return pygame.image.tostring(flip, "RGBA", True), 4
    return pygame.image.tostring(flip, "RGB", True), 3


def look_at(eye, target, up):
    f = np.array(target, dtype=np.float32) - np.array(eye, dtype=np.float32)
    f = f / np.linalg.norm(f)
    s = np.cross(f, np.array(up, dtype=np.float32))
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[:3, 3] = [-np.dot(s, eye), -np.dot(u, eye), np.dot(f, eye)]
    return m


def perspective(fovy, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fovy) / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2.0 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def model_matrix(pos, scale, rot_y=0.0, tilt=0.0):
    m = np.eye(4, dtype=np.float32)
    if rot_y:
        c, s = math.cos(rot_y), math.sin(rot_y)
        ry = np.eye(4, dtype=np.float32)
        ry[0, 0], ry[0, 2] = c, s
        ry[2, 0], ry[2, 2] = -s, c
        m = ry @ m
    if tilt:
        c, s = math.cos(tilt), math.sin(tilt)
        rx = np.eye(4, dtype=np.float32)
        rx[1, 1], rx[1, 2] = c, -s
        rx[2, 1], rx[2, 2] = s, c
        m = rx @ m
    m[:3, :3] *= scale
    m[:3, 3] = pos
    return m


def rot_translate_matrix(pos, rot_y=0.0, tilt=0.0):
    m = np.eye(4, dtype=np.float32)
    if rot_y:
        c, s = math.cos(rot_y), math.sin(rot_y)
        ry = np.eye(4, dtype=np.float32)
        ry[0, 0], ry[0, 2] = c, s
        ry[2, 0], ry[2, 2] = -s, c
        m = ry @ m
    if tilt:
        c, s = math.cos(tilt), math.sin(tilt)
        rx = np.eye(4, dtype=np.float32)
        rx[1, 1], rx[1, 2] = c, -s
        rx[2, 1], rx[2, 2] = s, c
        m = rx @ m
    m[:3, 3] = pos
    return m


def billboard_model(pos, scale, cam):
    eye = cam.eye_vec()
    f = -eye / np.linalg.norm(eye)
    up = np.array((0.0, 1.0, 0.0), dtype=np.float32)
    r = np.cross(f, up)
    r = r / np.linalg.norm(r)
    u = np.cross(r, f)
    m = np.eye(4, dtype=np.float32)
    m[:3, 0] = r * scale
    m[:3, 1] = u * scale
    m[:3, 2] = -f
    m[:3, 3] = pos
    return m


def set_mat(prog, name, m):
    prog[name].write(np.ascontiguousarray(m.T).tobytes())


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

class Camera:
    def __init__(self, aspect, dist_min=4.0, dist_max=520.0,
                 start_dist=62.0):
        self.aspect = aspect
        self.dist_min = dist_min
        self.dist_max = dist_max
        self.reset(start_dist)

    def reset(self, start_dist=None):
        self.yaw = 0.55
        self.pitch = 0.33
        self.dist = start_dist or self.dist
        self.target = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def set_aspect(self, aspect):
        self.aspect = aspect

    def eye_vec(self):
        cp = math.cos(self.pitch)
        d = np.array([cp * math.cos(self.yaw), math.sin(self.pitch),
                      cp * math.sin(self.yaw)], dtype=np.float32)
        return self.target + d * self.dist

    def frame(self):
        eye = self.eye_vec()
        f = self.target - eye
        f = f / np.linalg.norm(f)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        r = np.cross(f, up)
        r = r / np.linalg.norm(r)
        u = np.cross(r, f)
        return f, r, u

    def view_matrix(self):
        return look_at(self.eye_vec(), self.target, (0, 1, 0))

    def proj_matrix(self):
        return perspective(FOVY, self.aspect, NEAR, FAR)

    def orbit(self, dx, dy):
        self.yaw += dx * 0.006
        self.pitch = max(-1.45, min(1.45, self.pitch + dy * 0.006))

    def pan(self, dx, dy, vh):
        _, r, u = self.frame()
        wpp = 2.0 * self.dist * math.tan(math.radians(FOVY) / 2.0) / max(1, vh)
        self.target = self.target - r * (dx * wpp) + u * (dy * wpp)

    def zoom_to(self, px, py, factor, vw, vh, move_target=True):
        old = self.dist
        new = max(self.dist_min, min(self.dist_max, old * factor))
        if abs(new - old) < 1e-6:
            return
        self.dist = new
        if not move_target:
            return
        hit = self.screen_to_plane(px, py, vw, vh, 0.0)
        if hit is not None:
            k = 1.0 - new / old
            if 0.0 < k < 1.0:
                self.target = self.target + (hit - self.target) * k

    def _unproject(self, px, py, vw, vh):
        nx = 2.0 * px / vw - 1.0
        ny = 1.0 - 2.0 * py / vh
        inv = np.linalg.inv(self.proj_matrix() @ self.view_matrix())
        near = inv @ np.array([nx, ny, -1.0, 1.0])
        near /= near[3]
        far = inv @ np.array([nx, ny, 1.0, 1.0])
        far /= far[3]
        return near, far

    def screen_to_plane(self, px, py, vw, vh, plane_y=0.0):
        near, far = self._unproject(px, py, vw, vh)
        d = far - near
        if abs(d[1]) < 1e-9:
            return None
        t = (plane_y - near[1]) / d[1]
        if t < 0.0:
            return None
        return (near + d * t)[:3]

    def ray(self, px, py, vw, vh):
        near, far = self._unproject(px, py, vw, vh)
        d = far - near
        d = d / np.linalg.norm(d)
        return np.ascontiguousarray(near[:3]), np.ascontiguousarray(d[:3])


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

class Scene:
    def __init__(self, ctx, vw, vh, ring_radii=(1.45, 2.6)):
        self.ctx = ctx
        self.vw, self.vh = vw, vh
        self.textures = {}

        self.prog_sphere = ctx.program(
            vertex_shader=SPHERE_VERT, fragment_shader=SPHERE_FRAG)
        self.prog_line = ctx.program(
            vertex_shader=LINE_VERT, fragment_shader=LINE_FRAG)
        self.prog_halo = ctx.program(
            vertex_shader=QUAD_VERT, fragment_shader=HALO_FRAG)
        self.prog_ring = ctx.program(
            vertex_shader=QUAD_VERT, fragment_shader=RING_FRAG)
        self.prog_star = ctx.program(
            vertex_shader=STAR_VERT, fragment_shader=STAR_FRAG)

        buf, idx = sphere_arrays()
        self.sphere_vao = ctx.vertex_array(
            self.prog_sphere,
            [(ctx.buffer(buf), "3f 3f 2f",
              "in_position", "in_normal", "in_uv")],
            ctx.buffer(idx))

        q, qi = quad_arrays()
        self.halo_vao = ctx.vertex_array(
            self.prog_halo, [(ctx.buffer(q), "3f 2f",
                              "in_position", "in_uv")], ctx.buffer(qi))
        rim, rom = ring_radii
        rq, rqi = flat_ring_arrays(rim, rom)
        self.ring_vao = ctx.vertex_array(
            self.prog_ring, [(ctx.buffer(rq), "3f 2f",
                              "in_position", "in_uv")], ctx.buffer(rqi))

        # Ring buffers are pre-allocated at their full capacity so the VAOs
        # keep the right vertex count (moderngl caches it at creation).
        # Only the leading orbit_vert_count vertices are ever drawn, so a
        # smaller ring set can never leak stale geometry into the frame.
        orbit_zeros = np.zeros(
            ORBIT_RINGS_MAX * ORBIT_SEGS * 2 * 6, dtype=np.float32)
        moon_zeros = np.zeros(MOON_SEGS * 2 * 6, dtype=np.float32)
        self.orbit_buf = ctx.buffer(orbit_zeros.tobytes())
        self.orbit_vao = ctx.vertex_array(
            self.prog_line, [(self.orbit_buf, "3f 3f",
                              "in_position", "in_color")])
        self.orbit_vert_count = 0

        self.moon_ring_buf = ctx.buffer(moon_zeros.tobytes())
        self.moon_ring_vao = ctx.vertex_array(
            self.prog_line, [(self.moon_ring_buf, "3f 3f",
                              "in_position", "in_color")])

        self.star_vao = ctx.vertex_array(
            self.prog_star,
            [(ctx.buffer(self.build_stars()), "3f 3f f",
              "in_position", "in_color", "in_size")])

        self.make_fbo(vw, vh)

        ctx.enable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        try:
            ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        except Exception:
            pass

    def make_fbo(self, vw, vh):
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((vw, vh), 4)],
            depth_attachment=self.ctx.depth_texture((vw, vh)))

    def resize(self, vw, vh):
        if (vw, vh) != (self.vw, self.vh):
            self.vw, self.vh = vw, vh
            self.make_fbo(vw, vh)

    def upload_textures(self, surfaces):
        for name, surf in surfaces.items():
            if surf is None:
                continue
            data, comp = surface_to_gl(surf)
            tw, th = surf.get_size()
            tex = self.ctx.texture((tw, th), comp, data)
            tex.build_mipmaps()
            tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            try:
                max_aniso = self.ctx.info.get(
                    "GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT")
                if max_aniso:
                    tex.anisotropy = min(8.0, float(max_aniso))
            except Exception:
                pass
            self.textures[name] = tex

    def set_orbit_rings(self, rings):
        """rings: list of (radius, rgb-tuple).

        Writes the rings at the front of the pre-allocated buffer and records
        how many vertices belong to them; render() only draws that prefix,
        so leftover data from a previous, larger ring set is never drawn."""
        if rings:
            data = circle_line_arrays(
                [(r, c[0] / 255.0, c[1] / 255.0, c[2] / 255.0)
                 for r, c in rings])
            self.orbit_buf.write(data.tobytes())
        self.orbit_vert_count = len(rings) * ORBIT_SEGS * 2

    def set_moon_ring(self, center, offsets, color):
        """Draw the Moon's orbit ring as a closed polyline through Earth.

        offsets: MOON_SEGS unit direction vectors (scene frame) describing
        the true (inclined) orbital plane; each is scaled by the app to the
        stylised orbit radius before reaching this point."""
        r, g, b = color[0] / 255.0, color[1] / 255.0, color[2] / 255.0
        n = min(len(offsets), MOON_SEGS)
        pts = []
        for i in range(n):
            a = np.asarray(offsets[i], dtype=np.float32)
            b2 = np.asarray(offsets[(i + 1) % n], dtype=np.float32)
            for off in (a, b2):
                pts.append((center[0] + off[0], center[1] + off[1],
                            center[2] + off[2], r, g, b))
        while len(pts) < MOON_SEGS * 2:
            pts.append((center[0], center[1], center[2], 0.0, 0.0, 0.0))
        data = np.array(pts, dtype=np.float32).ravel()
        self.moon_ring_buf.write(data.tobytes())

    def build_stars(self):
        rng = np.random.default_rng(7)
        pts = []
        for _ in range(STARFIELD):
            v = rng.normal(size=3)
            v /= np.linalg.norm(v)
            v *= rng.uniform(120.0, 300.0)
            b = rng.uniform(0.25, 1.0)
            s = rng.uniform(1.5, 3.0)
            pts.append((v[0], v[1], v[2], b, b, b, s))
        return np.array(pts, dtype=np.float32).ravel()


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def render(ctx, scene, cam, bodies, sun_glow=6.5,
           sun_glow_color=(1.0, 0.62, 0.20)):
    """Render the scene.  bodies: list of dicts with
    name / pos (vec3) / radius / rot / tilt / tex / emissive / night
    and optionally ring (rim, rom) for Saturn's ring and
    ring_tex for the ring texture name.
    """
    vw, vh = scene.vw, scene.vh
    fbo = scene.fbo
    fbo.use()
    fbo.clear(color=(0.01, 0.02, 0.05, 1.0), depth=1.0)
    ctx.viewport = (0, 0, vw, vh)

    view = cam.view_matrix()
    proj = cam.proj_matrix()

    sun_pos = np.zeros(3, dtype=np.float32)
    for b in bodies:
        if b.get("emissive"):
            sun_pos = np.array(b["pos"], dtype=np.float32)
            break

    sp = scene.prog_sphere
    set_mat(sp, "u_view", view)
    set_mat(sp, "u_proj", proj)
    sp["u_view_pos"].value = cam.eye_vec()
    sp["u_sun_pos"].value = sun_pos
    sp["u_ambient"].value = (0.13, 0.15, 0.19)

    ring_prog = scene.prog_ring
    set_mat(ring_prog, "u_view", view)
    set_mat(ring_prog, "u_proj", proj)

    for b in bodies:
        pos = np.array(b["pos"], dtype=np.float32)
        scale = b["radius"]
        m = model_matrix(pos, scale, rot_y=b.get("rot", 0.0),
                         tilt=b.get("tilt", 0.0))
        set_mat(sp, "u_model", m)
        sp["u_emissive"] = 1.0 if b.get("emissive") else 0.0
        night = bool(b.get("night"))
        sp["u_has_night"] = 1.0 if night else 0.0
        tex = scene.textures.get(b.get("tex"))
        if tex is None:
            continue
        tex.use(0)
        sp["u_tex"] = 0
        if night:
            night_tex = scene.textures.get("EarthNight")
            if night_tex is not None:
                night_tex.use(1)
                sp["u_tex2"] = 1
        scene.sphere_vao.render(moderngl.TRIANGLES)

        ring = b.get("ring")
        if ring and "SaturnRing" in scene.textures:
            rim, rom = ring
            rm = rot_translate_matrix(pos, rot_y=b.get("rot", 0.0),
                                      tilt=b.get("tilt", 0.0))
            set_mat(ring_prog, "u_model", rm)
            ring_prog["u_alpha"].value = 1.0
            ring_prog["u_tex"] = 0
            scene.textures["SaturnRing"].use(0)
            scene.ring_vao.render(moderngl.TRIANGLES)

    line = scene.prog_line
    set_mat(line, "u_view", view)
    set_mat(line, "u_proj", proj)
    ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE)
    scene.orbit_vao.render(moderngl.LINES,
                           vertices=scene.orbit_vert_count)
    scene.moon_ring_vao.render(moderngl.LINES)
    ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

    ctx.depth_mask = False
    star = scene.prog_star
    set_mat(star, "u_view", view)
    set_mat(star, "u_proj", proj)
    ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE)
    scene.star_vao.render(moderngl.POINTS)

    halo = scene.prog_halo
    set_mat(halo, "u_view", view)
    set_mat(halo, "u_proj", proj)
    halo["u_color"].value = sun_glow_color
    ctx.blend_func = (moderngl.ONE, moderngl.ONE)
    hm = billboard_model(sun_pos, sun_glow, cam)
    set_mat(halo, "u_model", hm)
    scene.halo_vao.render(moderngl.TRIANGLES)

    ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
    ctx.depth_mask = True

    return fbo.read(viewport=(0, 0, vw, vh), components=3)


def to_surface(data, w, h):
    import pygame
    surf = pygame.image.frombuffer(data, (w, h), "RGB").convert()
    return pygame.transform.flip(surf, False, True).convert()