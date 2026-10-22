from datetime import datetime, timedelta
import math

import numpy as np
import pandas as pd
import pytest

from darkvessel.s1 import S1Scene, Swath
from darkvessel import ais
from darkvessel.associate import find_ghosts
from darkvessel.detect import Detection
from darkvessel.real import REAL_PRESETS, run_frame, orphan_alerts

T0 = datetime(2026, 10, 14, 5, 32, 4)
M_PER_DEG = math.radians(1) * 6_371_000


def flat_scene(srt=5.83e-3, inc=38.5):
    """A north-up scene: line runs south, pixel runs east, 10 m pixels."""
    lines = np.arange(0, 2001, 500.0)
    pixels = np.arange(0, 3001, 500.0)
    L, P = np.meshgrid(lines, pixels, indexing="ij")
    lat0, lon0 = 56.0, 11.0
    lat = lat0 - L * 10 / M_PER_DEG
    lon = lon0 + P * 10 / (M_PER_DEG * math.cos(math.radians(lat0)))
    grids = dict(lat=lat, lon=lon, srt=np.full_like(L, srt), inc=np.full_like(L, inc),
                 azt=L * 1.5e-3)
    return S1Scene("test", "", "Descending", T0, 1.5e-3, 10.0, 2001, 3001, 0.05546576, 7588.4,
                   [Swath("IW1", 0, 1500, 1717.1), Swath("IW2", 1501, 3000, 1451.6)],
                   lines, pixels, grids)


def test_pixel_latlon_round_trip():
    s = flat_scene()
    L, P = np.array([10.0, 777.0, 1990.0]), np.array([5.0, 1234.0, 2950.0])
    L2, P2 = s.to_pixel(*s.to_latlon(L, P))
    assert np.allclose(L2, L, atol=1e-3) and np.allclose(P2, P, atol=1e-3)


def test_velocity_projection_on_north_up_scene():
    s = flat_scene()
    v_az, v_rg = s.velocity_image(1000.0, 1000.0, 3.0, 4.0)   # 3 m/s east, 4 m/s north
    assert v_rg == pytest.approx(3.0, abs=0.01)     # pixel runs east
    assert v_az == pytest.approx(-4.0, abs=0.01)    # line runs south


def test_azimuth_shift_matches_hand_calculation():
    s = flat_scene()
    R = 5.83e-3 * 299_792_458 / 2
    expected = -R / 7588.4 * 5.0 * math.sin(math.radians(38.5))
    assert float(s.azimuth_shift_m(1000, 1000, 5.0)) == pytest.approx(expected, rel=1e-6)
    assert -370 < expected < -345


def test_ambiguity_offset_uses_the_subswath_prf():
    s = flat_scene()
    R = 5.83e-3 * 299_792_458 / 2
    near, far = s.ambiguity_offset_m(1000, np.array([100.0, 2500.0]))
    assert near == pytest.approx(0.05546576 * R * 1717.1 / (2 * 7588.4), rel=1e-6)
    assert far == pytest.approx(0.05546576 * R * 1451.6 / (2 * 7588.4), rel=1e-6)


def _track(rows):
    df = pd.DataFrame(rows, columns=["t", "lat", "lon", "sog", "cog", "heading"])
    df["mmsi"], df["kind"], df["status"] = 219000001, "Class A", "Under way"
    df["name"], df["ship_type"] = "TEST", "Cargo"
    for k in ("length", "width", "a", "b", "c", "d"):
        df[k] = np.nan
    return df


def test_fix_interpolates_between_reports():
    tr = _track([(T0, 56.0, 11.0, 10, 0, 0), (T0 + timedelta(seconds=20), 56.001, 11.0, 10, 0, 0)])
    f = ais.fix_at(tr, T0 + timedelta(seconds=5))
    assert f.method == "interp" and f.lat == pytest.approx(56.00025)


def test_dead_reckoning_moves_along_course():
    tr = _track([(T0, 56.0, 11.0, 10, 90, 90)])
    f = ais.fix_at(tr, T0 + timedelta(seconds=60), mode="dr")
    east_m = (f.lon - 11.0) * M_PER_DEG * math.cos(math.radians(56.0))
    assert east_m == pytest.approx(10 * 0.514444 * 60, rel=1e-3)
    assert f.age_s == pytest.approx(60)


def test_last_mode_does_not_move_the_ship():
    tr = _track([(T0, 56.0, 11.0, 10, 90, 90)])
    f = ais.fix_at(tr, T0 + timedelta(seconds=60), mode="last")
    assert (f.lat, f.lon) == (56.0, 11.0)


def test_stale_reports_are_dropped():
    tr = _track([(T0, 56.0, 11.0, 10, 90, 90)])
    assert ais.fix_at(tr, T0 + timedelta(minutes=30), mode="dr") is None


def test_hull_centre_offset_moves_towards_bow():
    tr = _track([(T0, 56.0, 11.0, 0, 0, 0)])
    tr[["a", "b", "c", "d"]] = [20, 180, 10, 10]     # antenna 20 m from the bow
    f = ais.fix_at(tr, T0, mode="last")
    north_m = (f.lat - 56.0) * M_PER_DEG
    assert north_m == pytest.approx(-80, abs=0.5)     # centre is 80 m aft


def test_ghost_offset_can_differ_per_detection():
    parent = Detection(1000, 2000, 300.0, 300.0, 40, 200.0)
    ghost = Detection(1010, 2000 + 4600, 8.0, 8.0, 6, 60.0)
    assert find_ghosts([parent, ghost], [5800, 4600]) == []
    assert find_ghosts([parent, ghost], [4600, 4600]) == [1]


def _frame(det_line, ship_line, shift):
    det = dict(line=det_line, pixel=500.0, peak=40.0, peak_db=40.0, area_px=20, length_m=100.0,
               amb_m=5000.0)
    fix = dict(mmsi=1, line=ship_line, pixel=500.0, shift_m=shift, sog_kn=12.0, age_s=5.0)
    return dict(detections=[det], ais={"dr": [fix], "last": [fix], "interp": [fix]})


def test_doppler_correction_pairs_a_displaced_ship():
    fr = _frame(det_line=1000.0 - 35.0, ship_line=1000.0, shift=-350.0)
    assert run_frame(fr, REAL_PRESETS["dead_reckoning"])["dark"] == [0]
    assert orphan_alerts(fr, run_frame(fr, REAL_PRESETS["dead_reckoning"])) == [0]
    assert run_frame(fr, REAL_PRESETS["+doppler"])["pairs"] == {0: 1}


def test_fixed_structures_are_not_alerts():
    fr = _frame(det_line=1000.0, ship_line=5000.0, shift=0.0)
    assert run_frame(fr, REAL_PRESETS["+ghosts"])["dark"] == [0]
    res = run_frame(fr, REAL_PRESETS["+fixed_structures"], fixed_mask=np.array([True]))
    assert res["dark"] == [] and res["fixed"] == [0]
