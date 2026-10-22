"""SAR detection <-> AIS association and dark-vessel flagging.

Each step can be switched on separately so its effect can be measured:
  dead_reckon    move each AIS report to the image time using SOG/COG
  doppler        move the predicted position to where the SAR *shows* the ship
                 (azimuth displacement from its range velocity)
  adaptive_gate  gate radius grows with report age and ship length
  ghosts         drop detections that sit one ambiguity offset from a much
                 brighter detection
  optimal        one-to-one Hungarian assignment instead of greedy nearest
"""
from dataclasses import dataclass, field
import math
import numpy as np
from scipy.optimize import linear_sum_assignment

from .geometry import SarGeometry

BIG = 1e9


@dataclass
class Config:
    name: str = "full"
    dead_reckon: bool = True
    doppler: bool = True
    adaptive_gate: bool = True
    ghosts: bool = True
    optimal: bool = True
    gate_m: float = 300.0            # fixed gate when adaptive_gate is off
    pos_sigma_m: float = 40.0        # base 1-sigma position uncertainty
    heading_sigma_deg: float = 3.0   # COG noise
    accel_sigma: float = 0.0006      # m/s^2, unmodelled manoeuvre
    gate_sigmas: float = 3.5


PRESETS = {
    "naive": Config("naive", dead_reckon=False, doppler=False, adaptive_gate=False,
                    ghosts=False, optimal=False),
    "dead_reckoning": Config("dead_reckoning", doppler=False, adaptive_gate=False,
                             ghosts=False, optimal=True),
    "+doppler": Config("+doppler", adaptive_gate=False, ghosts=False),
    "+adaptive_gate": Config("+adaptive_gate", ghosts=False),
    "full": Config("full"),
}


@dataclass
class Result:
    pairs: dict = field(default_factory=dict)        # det index -> ship_id (AIS)
    dark: list = field(default_factory=list)         # det indices flagged dark
    ghosts: list = field(default_factory=list)       # det indices dropped as ghosts
    ais_unmatched: list = field(default_factory=list)
    predicted: dict = field(default_factory=dict)    # ship_id -> (x, y, sigma)


def predict_ais(report, geom: SarGeometry, cfg: Config):
    x, y = report.x_m, report.y_m
    c = math.radians(report.cog_deg)
    vx, vy = report.sog_ms * math.sin(c), report.sog_ms * math.cos(c)
    dt = -report.t_offset_s          # time to move forward to image time
    if cfg.dead_reckon:
        x += vx * dt
        y += vy * dt
    if cfg.doppler:
        y += geom.azimuth_shift_m(vx)
    if cfg.adaptive_gate:
        # heading error grows linearly with age, manoeuvres like a t^2 / 2
        heading = report.sog_ms * abs(dt) * math.radians(cfg.heading_sigma_deg)
        manoeuvre = 0.5 * cfg.accel_sigma * dt * dt if report.sog_ms > 0.5 else 0.0
        sigma = math.sqrt(cfg.pos_sigma_m ** 2 + heading ** 2 + manoeuvre ** 2)
    else:
        sigma = cfg.gate_m / cfg.gate_sigmas
    return x, y, sigma


def find_ghosts(dets, geom: SarGeometry, along_tol_m=60.0, cross_tol_m=30.0,
                min_ratio_db=10.0):
    off = geom.ambiguity_offset_m
    ghost = set()
    for i, d in enumerate(dets):
        for j, p in enumerate(dets):
            if i == j:
                continue
            # a ghost of a long hull can be centred anywhere along that hull
            slack = 0.4 * p.length_m
            if abs(d.x_m - p.x_m) > cross_tol_m + slack:
                continue
            if abs(abs(d.y_m - p.y_m) - off) > along_tol_m + slack:
                continue
            if 10 * math.log10(p.peak_raw / max(d.peak_raw, 1e-9)) >= min_ratio_db:
                ghost.add(i)
    return sorted(ghost)


def associate(dets, ais, geom: SarGeometry, cfg: Config) -> Result:
    res = Result()
    res.ghosts = find_ghosts(dets, geom) if cfg.ghosts else []
    keep = [i for i in range(len(dets)) if i not in set(res.ghosts)]

    preds = [predict_ais(r, geom, cfg) for r in ais]
    for r, p in zip(ais, preds):
        res.predicted[r.ship_id] = p

    if not keep or not ais:
        res.dark = keep
        res.ais_unmatched = [r.ship_id for r in ais]
        return res

    cost = np.full((len(keep), len(ais)), BIG)
    for a, i in enumerate(keep):
        d = dets[i]
        for b, (px, py, sig) in enumerate(preds):
            dist = math.hypot(d.x_m - px, d.y_m - py)
            # a long hull lets the bright centroid sit anywhere along it
            s = math.hypot(sig, 0.25 * d.length_m) if cfg.adaptive_gate else sig
            z = dist / s
            if z <= cfg.gate_sigmas:
                cost[a, b] = z * z

    if cfg.optimal:
        rows, cols = linear_sum_assignment(cost)
        matched = [(r, c) for r, c in zip(rows, cols) if cost[r, c] < BIG]
    else:
        matched, used = [], set()
        for a in range(len(keep)):
            order = np.argsort(cost[a])
            for b in order:
                if cost[a, b] >= BIG:
                    break
                if b not in used:
                    matched.append((a, b)); used.add(b)
                    break

    matched_dets = set()
    matched_ais = set()
    for a, b in matched:
        res.pairs[keep[a]] = ais[b].ship_id
        matched_dets.add(keep[a]); matched_ais.add(b)
    res.dark = [i for i in keep if i not in matched_dets]
    res.ais_unmatched = [ais[b].ship_id for b in range(len(ais)) if b not in matched_ais]
    return res
