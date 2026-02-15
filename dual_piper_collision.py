"""
dual_piper_collision.py

Two animated Piper robots with real-time collision detection using
GJK on the Minkowski difference of their convex superquadric primitives.
Colliding primitives are highlighted red in the viser viewer.

Usage:
    cd superdec
    python dual_piper_collision.py
"""

import numpy as np
import trimesh
import viser
import time
from scipy.spatial.transform import Rotation as Rot
from superdec.utils.predictions_handler import PredictionHandler


# ── Config ──────────────────────────────────────────────────────────
SQ_PATH = "data/robots/piper/superquadrics/piper.npz"
GJK_SAMPLE_N = 14          # n×n surface points per primitive for GJK
MESH_RES = 8               # mesh resolution for viser display
DT = 0.03                  # animation timestep (seconds)

COLOR_A = [100, 150, 255, 255]      # blue
COLOR_B = [100, 255, 150, 255]      # green
COLOR_COLLIDE = [255, 40, 40, 255]  # red


# ════════════════════════════════════════════════════════════════════
#  Superquadric surface sampling (for GJK point sets)
# ════════════════════════════════════════════════════════════════════

def sample_sq_surface(scale, exponents, n=14):
    """Sample n*n points on a superquadric surface in its local frame."""
    e1, e2 = float(exponents[0]), float(exponents[1])
    u = np.linspace(-np.pi, np.pi, n, endpoint=False)
    v = np.linspace(-np.pi / 2 + 0.02, np.pi / 2 - 0.02, n)
    uu, vv = np.meshgrid(u, v)
    uu, vv = uu.ravel(), vv.ravel()

    def sp(x, e):
        return np.sign(x) * np.abs(x) ** e

    x = scale[0] * sp(np.cos(vv), e1) * sp(np.cos(uu), e2)
    y = scale[1] * sp(np.cos(vv), e1) * sp(np.sin(uu), e2)
    z = scale[2] * sp(np.sin(vv), e1)
    return np.column_stack([x, y, z])


# ════════════════════════════════════════════════════════════════════
#  GJK  –  collision via Minkowski difference
# ════════════════════════════════════════════════════════════════════

def _support(verts, d):
    """Furthest vertex along direction d."""
    return verts[np.argmax(verts @ d)].copy()


def _mink_support(va, vb, d):
    """Support point on the Minkowski difference A - B."""
    return _support(va, d) - _support(vb, -d)


def _triple_cross(a, b, c):
    """(a x b) x c  –  used to get perpendicular in the plane of ab."""
    return np.cross(np.cross(a, b), c)


def gjk(va, vb, max_iter=32):
    """
    GJK intersection test.

    Implicitly builds a simplex on the Minkowski difference A - B.
    Returns True iff the origin lies inside that difference, meaning
    the two convex shapes overlap.
    """
    d = va.mean(0) - vb.mean(0)
    if np.dot(d, d) < 1e-12:
        d = np.array([1.0, 0.0, 0.0])

    s = [_mink_support(va, vb, d)]
    d = -s[0]

    for _ in range(max_iter):
        dd = np.dot(d, d)
        if dd < 1e-20:
            return True                       # degenerate → overlap
        a = _mink_support(va, vb, d / np.sqrt(dd))
        if np.dot(a, d) < 0:
            return False                      # can't reach origin
        s.append(a)
        hit, d = _evolve(s)
        if hit:
            return True
    return False


def _evolve(s):
    """Advance the simplex toward the origin; return (contains, new_dir)."""
    n = len(s)

    # ── line (2-simplex) ──
    if n == 2:
        b, a = s[0], s[1]
        ab, ao = b - a, -a
        if np.dot(ab, ao) > 0:
            return False, _triple_cross(ab, ao, ab)
        s[:] = [a]
        return False, ao

    # ── triangle (3-simplex) ──
    if n == 3:
        c, b, a = s[0], s[1], s[2]
        ab, ac, ao = b - a, c - a, -a
        abc = np.cross(ab, ac)

        if np.dot(np.cross(abc, ac), ao) > 0:
            if np.dot(ac, ao) > 0:
                s[:] = [c, a]
                return False, _triple_cross(ac, ao, ac)
            s[:] = [b, a]
            return _evolve(s)

        if np.dot(np.cross(ab, abc), ao) > 0:
            s[:] = [b, a]
            return _evolve(s)

        if np.dot(abc, ao) > 0:
            return False, abc
        s[:] = [b, c, a]
        return False, -abc

    # ── tetrahedron (4-simplex) ──
    if n == 4:
        dp, c, b, a = s[0], s[1], s[2], s[3]
        ab, ac, ad, ao = b - a, c - a, dp - a, -a

        abc = np.cross(ab, ac)
        if np.dot(abc, ad) > 0:
            abc = -abc
        acd = np.cross(ac, ad)
        if np.dot(acd, ab) > 0:
            acd = -acd
        adb = np.cross(ad, ab)
        if np.dot(adb, ac) > 0:
            adb = -adb

        if np.dot(abc, ao) > 0:
            s[:] = [c, b, a]
            return _evolve(s)
        if np.dot(acd, ao) > 0:
            s[:] = [dp, c, a]
            return _evolve(s)
        if np.dot(adb, ao) > 0:
            s[:] = [b, dp, a]
            return _evolve(s)
        return True, np.zeros(3)              # origin inside!

    return False, np.array([1.0, 0.0, 0.0])


# ════════════════════════════════════════════════════════════════════
#  AABB broad-phase
# ════════════════════════════════════════════════════════════════════

def aabb_of(pts):
    return pts.min(0), pts.max(0)


def aabb_hit(a, b):
    return np.all(a[1] >= b[0]) and np.all(b[1] >= a[0])


# ════════════════════════════════════════════════════════════════════
#  Robot wrapper
# ════════════════════════════════════════════════════════════════════

class PiperRobot:
    """Load a Piper from its superquadric npz and re-pose it rigidly."""

    def __init__(self, path, name):
        self.name = name
        self.h = PredictionHandler.from_npz(path)
        B, P = self.h.exist.shape[:2]

        # reference (default) pose
        self._ref_rot = self.h.rotation.copy()
        self._ref_trans = self.h.translation.copy()

        # existing primitives
        self.prims = [
            (b, p) for b in range(B) for p in range(P)
            if self.h.exist[b, p] > 0.5
        ]

        # pre-sample surfaces in each primitive's local frame
        self._local_pts = {}
        for b, p in self.prims:
            self._local_pts[(b, p)] = sample_sq_surface(
                self.h.scale[b, p], self.h.exponents[b, p], GJK_SAMPLE_N
            )

        self.world_pts = {}
        self.bboxes = {}
        self.set_pose(np.eye(3), np.zeros(3))

    # ── posing ──────────────────────────────────────────────────────
    def set_pose(self, R, t):
        """Apply a rigid-body transform (rotation R, translation t)."""
        for b, p in self.prims:
            self.h.rotation[b, p] = R @ self._ref_rot[b, p]
            self.h.translation[b, p] = R @ self._ref_trans[b, p] + t
        self._update_world()

    def _update_world(self):
        for b, p in self.prims:
            rot = self.h.rotation[b, p]
            trans = self.h.translation[b, p]
            w = (rot @ self._local_pts[(b, p)].T).T + trans
            self.world_pts[(b, p)] = w
            self.bboxes[(b, p)] = aabb_of(w)

    def extent(self):
        """Axis-aligned bounding box of the whole robot."""
        all_pts = np.vstack(list(self.world_pts.values()))
        return all_pts.min(0), all_pts.max(0)


# ════════════════════════════════════════════════════════════════════
#  Collision query
# ════════════════════════════════════════════════════════════════════

def detect_collisions(ra: PiperRobot, rb: PiperRobot):
    """
    Broad phase  :  AABB overlap per primitive pair
    Narrow phase :  GJK on sampled superquadric surfaces
                    (implicitly tests Minkowski difference for origin)

    Returns (colliding prims in A, colliding prims in B).
    """
    ca, cb = set(), set()
    for pa in ra.prims:
        ba = ra.bboxes[pa]
        for pb in rb.prims:
            if not aabb_hit(ba, rb.bboxes[pb]):
                continue
            if gjk(ra.world_pts[pa], rb.world_pts[pb]):
                ca.add(pa)
                cb.add(pb)
    return ca, cb


# ════════════════════════════════════════════════════════════════════
#  Viser helpers
# ════════════════════════════════════════════════════════════════════

def rot_to_wxyz(mat):
    q = Rot.from_matrix(mat).as_quat()          # [x, y, z, w]
    return np.array([q[3], q[0], q[1], q[2]])   # viser wants [w, x, y, z]


def make_prim_mesh(handler, b, p, color, res=MESH_RES):
    """Trimesh for one superquadric in its local frame."""
    v, f = handler._superquadric_mesh(
        handler.scale[b, p], handler.exponents[b, p],
        np.eye(3), np.zeros(3), res,
    )
    m = trimesh.Trimesh(vertices=v, faces=f)
    m.visual.face_colors = np.tile(color, (len(f), 1))
    return m


def add_robot_handles(server, robot, color, collision_color):
    """
    Create two viser mesh handles per primitive:
      normal  –  always visible
      overlay –  shown only when that primitive is colliding
    Returns dict  {(b,p): (normal_handle, overlay_handle)}
    """
    handles = {}
    for b, p in robot.prims:
        m_norm = make_prim_mesh(robot.h, b, p, color)
        h_norm = server.scene.add_mesh_trimesh(
            f"/{robot.name}/n_{b}_{p}", m_norm
        )

        m_coll = make_prim_mesh(robot.h, b, p, collision_color)
        h_coll = server.scene.add_mesh_trimesh(
            f"/{robot.name}/c_{b}_{p}", m_coll
        )
        h_coll.visible = False

        handles[(b, p)] = (h_norm, h_coll)
    return handles


def sync_handles(robot, handles, collisions):
    """Push current transforms + collision visibility to viser."""
    for bp, (h_norm, h_coll) in handles.items():
        b, p = bp
        wxyz = rot_to_wxyz(robot.h.rotation[b, p])
        pos = robot.h.translation[b, p]
        h_norm.position = pos
        h_norm.wxyz = wxyz
        h_coll.position = pos
        h_coll.wxyz = wxyz
        h_coll.visible = bp in collisions


# ════════════════════════════════════════════════════════════════════
#  Main loop
# ════════════════════════════════════════════════════════════════════

def main():
    # ── load robots ────────────────────────────────────────────────
    print("Loading Piper A ...")
    robot_a = PiperRobot(SQ_PATH, "piper_A")
    print("Loading Piper B ...")
    robot_b = PiperRobot(SQ_PATH, "piper_B")

    lo, hi = robot_a.extent()
    span = hi - lo
    center = (lo + hi) / 2.0
    sep = float(span.max()) * 0.7          # initial separation

    print(f"Robot bounding box: {lo} → {hi}  (span {span})")
    print(f"Primitives per robot: {len(robot_a.prims)}")

    # ── viser scene ────────────────────────────────────────────────
    server = viser.ViserServer()

    handles_a = add_robot_handles(server, robot_a, COLOR_A, COLOR_COLLIDE)
    handles_b = add_robot_handles(server, robot_b, COLOR_B, COLOR_COLLIDE)

    # GUI panel
    with server.gui.add_folder("Collision Detection"):
        gui_status = server.gui.add_text("Status", initial_value="---", disabled=True)
        gui_pairs = server.gui.add_text("GJK pairs", initial_value="0", disabled=True)
        gui_ms = server.gui.add_text("Frame ms", initial_value="0", disabled=True)
        gui_sep = server.gui.add_slider(
            "Separation", min=0.0, max=float(sep * 3), step=0.01,
            initial_value=float(sep),
        )
        gui_speed = server.gui.add_slider(
            "Speed", min=0.1, max=3.0, step=0.1, initial_value=1.0,
        )

    print("Viser ready — open the URL above in a browser.")
    t = 0.0

    while True:
        t0 = time.perf_counter()
        t += DT * gui_speed.value

        # ── animate poses ──────────────────────────────────────────
        half = gui_sep.value / 2.0
        phase = t * 0.6

        # Robot A swings from the left
        Ra = Rot.from_euler("y", 0.4 * np.sin(phase)).as_matrix()
        ta = np.array([-half + 0.25 * np.sin(phase), 0.0, 0.1 * np.cos(phase * 0.7)])

        # Robot B swings from the right (mirrored)
        Rb = Rot.from_euler("y", np.pi + 0.4 * np.sin(phase + 1.0)).as_matrix()
        tb = np.array([half - 0.25 * np.sin(phase + 1.0), 0.0, -0.1 * np.cos(phase * 0.7)])

        robot_a.set_pose(Ra, ta)
        robot_b.set_pose(Rb, tb)

        # ── collision detection ────────────────────────────────────
        ca, cb = detect_collisions(robot_a, robot_b)

        # ── update viser ───────────────────────────────────────────
        sync_handles(robot_a, handles_a, ca)
        sync_handles(robot_b, handles_b, cb)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        n_coll = len(ca) + len(cb)
        gui_status.value = f"COLLISION ({n_coll} prims)" if n_coll else "Clear"
        gui_ms.value = f"{elapsed_ms:.1f}"

        # count narrow-phase calls for stats
        broad = sum(
            1 for pa in robot_a.prims
            for pb in robot_b.prims
            if aabb_hit(robot_a.bboxes[pa], robot_b.bboxes[pb])
        )
        gui_pairs.value = str(broad)

        time.sleep(max(0, DT - elapsed_ms / 1000.0))


if __name__ == "__main__":
    main()
