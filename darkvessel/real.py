"""Association and scoring on Sentinel-1 frames processed by scripts/run_real.py.

There is no ground truth for which radar targets are dark, so the scores are
the things real data can check:
  * pairing rate: share of AIS vessels on open water that end up paired with a
    radar detection (a lower bound on detector recall, by length);
  * orphan alerts: alerts within 1.5 km of an AIS vessel the matcher left
    unpaired, i.e. an AIS ship drawn as dark because it was mispositioned;
  * ghosts and fixed structures removed, and what is left per 1000 km2.
"""
from dataclasses import dataclass, replace
import math

import numpy as np
from scipy.spatial import cKDTree

from .associate import Config, find_ghosts, match
from .detect import Detection

ORPHAN_M = 1500.0
LENGTH_BINS = [(0, 20), (20, 40), (40, 100), (100, 1e9)]


@dataclass
class RealConfig(Config):
    fix: str = "dr"                 # which AIS fix: "last", "dr" or "interp"
    fixed_structures: bool = False  # drop cross-pass persistent targets and AtoN
    pos_sigma_m: float = 60.0


REAL_PRESETS = {
    "naive": RealConfig("naive", fix="last", dead_reckon=False, doppler=False,
                        adaptive_gate=False, ghosts=False, optimal=False),
    "dead_reckoning": RealConfig("dead_reckoning", doppler=False, adaptive_gate=False,
                                 ghosts=False),
    "+doppler": RealConfig("+doppler", adaptive_gate=False, ghosts=False),
    "+adaptive_gate": RealConfig("+adaptive_gate", ghosts=False),
    "+ghosts": RealConfig("+ghosts"),
    "+fixed_structures": RealConfig("+fixed_structures", fixed_structures=True),
}


def to_detections(frame):
    return [Detection(d["pixel"] * 10.0, d["line"] * 10.0, d["peak"], 10 ** (d["peak_db"] / 10),
                      d["area_px"], d["length_m"]) for d in frame["detections"]]


def predictions(fixes, cfg: RealConfig):
    out = []
    for f in fixes:
        x, y = f["pixel"] * 10.0, f["line"] * 10.0
        if cfg.doppler:
            y += f["shift_m"]
        if cfg.adaptive_gate:
            v = f["sog_kn"] * 0.514444
            age = f["age_s"]
            heading = v * age * math.radians(cfg.heading_sigma_deg)
            manoeuvre = 0.5 * cfg.accel_sigma * age * age if v > 0.25 else 0.0
            sigma = math.sqrt(cfg.pos_sigma_m ** 2 + heading ** 2 + manoeuvre ** 2)
        else:
            sigma = cfg.gate_m / cfg.gate_sigmas
        out.append((x, y, sigma))
    return out


def run_frame(frame, cfg: RealConfig, fixed_mask=None):
    dets = to_detections(frame)
    fixes = frame["ais"][cfg.fix]
    ghosts = find_ghosts(dets, [d["amb_m"] for d in frame["detections"]],
                         along_tol_m=150.0, cross_tol_m=40.0, min_ratio_db=12.0) if cfg.ghosts else []
    drop = set(ghosts)
    fixed = []
    if cfg.fixed_structures and fixed_mask is not None:
        fixed = [i for i in range(len(dets)) if fixed_mask[i] and i not in drop]
        drop |= set(fixed)
    keep = [i for i in range(len(dets)) if i not in drop]
    preds = predictions(fixes, cfg)
    pairs, dark, unmatched = match(dets, preds, [f["mmsi"] for f in fixes], cfg, keep)
    return dict(pairs=pairs, dark=dark, unmatched=unmatched, ghosts=ghosts, fixed=fixed)


def orphan_alerts(frame, res, fix="interp"):
    """Alerts that sit within ORPHAN_M of an AIS vessel left unpaired."""
    un = set(res["unmatched"])
    pts = [(f["pixel"] * 10, f["line"] * 10 + f["shift_m"]) for f in frame["ais"][fix] if f["mmsi"] in un]
    if not pts or not res["dark"]:
        return []
    t = cKDTree(pts)
    D = frame["detections"]
    return [i for i in res["dark"]
            if t.query([D[i]["pixel"] * 10, D[i]["line"] * 10])[0] <= ORPHAN_M]


def ghost_offset_control(frames, factors=(0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4)):
    """Count bright-faint detection pairs at k times the predicted ambiguity
    offset. If the ghosts are real, the count spikes at k = 1."""
    counts = {k: 0 for k in factors}
    for fr in frames:
        D = fr["detections"]
        if not D:
            continue
        xy = np.array([[d["pixel"] * 10, d["line"] * 10] for d in D])
        pk = np.array([d["peak_db"] for d in D])
        t = cKDTree(xy)
        for j, d in enumerate(D):
            for k in factors:
                for sgn in (-1, 1):
                    q = xy[j] + [0, sgn * k * d["amb_m"]]
                    for i in t.query_ball_point(q, 150.0):
                        if abs(xy[i, 0] - xy[j, 0]) <= 40 and pk[j] - pk[i] >= 12:
                            counts[k] += 1
    return counts
