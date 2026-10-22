"""Synthetic SAR scene + AIS feed with known ground truth.

What is modelled, because each one breaks naive SAR-AIS matching:
  * K-distributed sea clutter (spikier as sea state rises)
  * ships as extended targets whose brightness grows with length
  * Doppler azimuth displacement of ships moving in range
  * azimuth ambiguity ghosts of bright ships
  * AIS reports that are minutes old, noisy, sometimes from a turning ship,
    sometimes missing (dark) and sometimes spoofed
"""
from dataclasses import dataclass, field
import math
import numpy as np
from scipy import ndimage

from .geometry import SarGeometry

KNOT = 0.514444

SEA_STATES = {
    # texture shape (lower = spikier), ship-to-clutter penalty in dB
    "calm": dict(nu=12.0, scr_penalty_db=0.0),
    "moderate": dict(nu=4.0, scr_penalty_db=2.0),
    "rough": dict(nu=1.5, scr_penalty_db=4.0),
}


@dataclass
class Ship:
    id: int
    kind: str
    length_m: float
    speed_ms: float
    course_deg: float          # clockwise from +y (azimuth) towards +x (range)
    turn_rate_dps: float
    x_m: float                 # true position at image time
    y_m: float
    has_ais: bool
    spoofed: bool = False
    app_x_m: float = 0.0       # where it appears in the SAR image
    app_y_m: float = 0.0
    in_image: bool = True

    @property
    def dark(self) -> bool:
        return (not self.has_ais) or self.spoofed

    def velocity(self):
        c = math.radians(self.course_deg)
        return self.speed_ms * math.sin(c), self.speed_ms * math.cos(c)


@dataclass
class AisReport:
    ship_id: int
    t_offset_s: float          # report time minus image time (negative = before)
    x_m: float
    y_m: float
    sog_ms: float
    cog_deg: float


@dataclass
class Scene:
    image: np.ndarray          # linear intensity, clutter mean ~1
    geom: SarGeometry
    ships: list
    ais: list
    ghosts: list = field(default_factory=list)   # (x_m, y_m, parent_id)
    sea_state: str = "moderate"

    @property
    def size_m(self):
        h, w = self.image.shape
        return w * self.geom.pixel_m, h * self.geom.pixel_m


def _propagate(x, y, speed, course_deg, turn_dps, dt):
    """Constant-turn-rate motion over dt seconds (dt may be negative)."""
    steps = max(1, int(abs(dt) // 5))
    h = dt / steps
    c = course_deg
    for _ in range(steps):
        r = math.radians(c)
        x += speed * math.sin(r) * h
        y += speed * math.cos(r) * h
        c += turn_dps * h
    return x, y, c % 360


def _ship_patch(length_px, width_px, course_deg, rng):
    """Small intensity patch (peak ~1) of an extended ship target."""
    half = int(math.ceil(length_px / 2 + 3))
    size = 2 * half + 1
    patch = np.zeros((size, size))
    n = max(3, int(length_px * 3))
    r = math.radians(course_deg)
    ux, uy = math.sin(r), math.cos(r)       # along-hull unit vector (x, y)
    vx, vy = uy, -ux                        # across-hull
    for s in np.linspace(-length_px / 2, length_px / 2, n):
        for w in np.linspace(-width_px / 2, width_px / 2, max(1, int(width_px * 2))):
            px = half + s * ux + w * vx
            py = half + s * uy + w * vy
            i0, j0 = int(math.floor(py)), int(math.floor(px))
            fy, fx = py - i0, px - j0
            for di, dj, wt in ((0, 0, (1 - fy) * (1 - fx)), (0, 1, (1 - fy) * fx),
                               (1, 0, fy * (1 - fx)), (1, 1, fy * fx)):
                ii, jj = i0 + di, j0 + dj
                if 0 <= ii < size and 0 <= jj < size:
                    patch[ii, jj] += wt
    patch = ndimage.gaussian_filter(patch, 0.6)
    # bright scatterers (superstructure, containers) fluctuate strongly
    patch *= rng.gamma(1.0, 1.0, patch.shape)
    m = patch.max()
    return patch / m if m > 0 else patch


def _add_patch(img, patch, cx_px, cy_px, scale):
    h, w = img.shape
    ph, pw = patch.shape
    i0 = int(round(cy_px)) - ph // 2
    j0 = int(round(cx_px)) - pw // 2
    ia, ja = max(0, i0), max(0, j0)
    ib, jb = min(h, i0 + ph), min(w, j0 + pw)
    if ia >= ib or ja >= jb:
        return False
    img[ia:ib, ja:jb] += scale * patch[ia - i0:ib - i0, ja - j0:jb - j0]
    return True


def make_scene(seed=0, n_ships=45, size_px=2048, sea_state="moderate",
               geom=SarGeometry(), p_dark_fishing=0.25, p_dark_other=0.06,
               p_spoof=0.03, looks=3):
    rng = np.random.default_rng(seed)
    ss = SEA_STATES[sea_state]
    pix = geom.pixel_m
    size_m = size_px * pix

    # --- clutter: texture (gamma, mean 1) x speckle (gamma, L looks) ---
    texture = rng.gamma(ss["nu"], 1.0 / ss["nu"], (size_px // 8, size_px // 8))
    texture = ndimage.zoom(texture, 8, order=1)[:size_px, :size_px]
    texture /= texture.mean()
    speckle = rng.gamma(looks, 1.0 / looks, (size_px, size_px))
    # slow wind gradient across the swath
    ramp = np.linspace(0.8, 1.2, size_px)[None, :]
    img = texture * speckle * ramp

    ships, ais, ghosts = [], [], []
    signal = np.zeros_like(img)
    for i in range(n_ships):
        kind = rng.choice(["fishing", "small_craft", "cargo", "tanker"],
                          p=[0.45, 0.15, 0.28, 0.12])
        if kind == "fishing":
            L = rng.uniform(12, 35); v = rng.choice([0.0, rng.uniform(2, 9)], p=[0.25, 0.75])
        elif kind == "small_craft":
            L = rng.uniform(10, 25); v = rng.uniform(5, 25)
        elif kind == "cargo":
            L = rng.uniform(90, 230); v = rng.choice([0.0, rng.uniform(9, 16)], p=[0.2, 0.8])
        else:
            L = rng.uniform(150, 330); v = rng.choice([0.0, rng.uniform(8, 14)], p=[0.3, 0.7])
        v *= KNOT
        course = rng.uniform(0, 360)
        turn = rng.choice([0.0, rng.uniform(-0.06, 0.06)], p=[0.7, 0.3]) if v > 0 else 0.0
        margin = 300
        x, y = rng.uniform(margin, size_m - margin, 2)
        p_dark = p_dark_fishing if kind in ("fishing", "small_craft") else p_dark_other
        has_ais = rng.random() > p_dark
        spoofed = has_ais and rng.random() < p_spoof
        s = Ship(i, str(kind), float(L), float(v), float(course), float(turn),
                 float(x), float(y), bool(has_ais), bool(spoofed))
        vxr, _ = s.velocity()
        s.app_x_m = s.x_m
        s.app_y_m = s.y_m + geom.azimuth_shift_m(vxr)
        s.in_image = 0 <= s.app_y_m < size_m

        # brightness: ~ +15 dB per decade of length, small boats near clutter
        scr_db = 9.0 + 16.0 * math.log10(L / 10.0) - ss["scr_penalty_db"]
        scr_db += rng.normal(0, 2.0)
        patch = _ship_patch(L / pix, max(1.0, L / 7 / pix), course, rng)
        if s.in_image:
            _add_patch(signal, patch, s.app_x_m / pix, s.app_y_m / pix, 10 ** (scr_db / 10))
            # azimuth ambiguities: weaker, smeared copies +/- one ambiguity offset
            g = ndimage.gaussian_filter1d(patch, 2.5, axis=0)
            g /= max(g.max(), 1e-9)
            gscale = 10 ** ((scr_db + geom.aasr_db) / 10)
            for sgn in (-1, 1):
                gy = s.app_y_m + sgn * geom.ambiguity_offset_m
                if _add_patch(signal, g, s.app_x_m / pix, gy / pix, gscale):
                    ghosts.append((s.app_x_m, gy, s.id))
        ships.append(s)

        if has_ais:
            dt = rng.uniform(-720, 180)
            rx, ry, rc = _propagate(s.x_m, s.y_m, s.speed_ms, s.course_deg, s.turn_rate_dps, dt)
            rx += rng.normal(0, 10); ry += rng.normal(0, 10)
            if spoofed:
                ang = rng.uniform(0, 2 * math.pi); d = rng.uniform(2000, 5000)
                rx += d * math.cos(ang); ry += d * math.sin(ang)
            sog = max(0.0, s.speed_ms + rng.normal(0, 0.1))
            cog = (rc + rng.normal(0, 3.0)) % 360
            ais.append(AisReport(i, float(dt), float(rx), float(ry), float(sog), float(cog)))

    img = img + signal
    return Scene(img.astype(np.float32), geom, ships, ais, ghosts, sea_state)
