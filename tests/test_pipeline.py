import math
import numpy as np
import pytest

from darkvessel.geometry import SarGeometry
from darkvessel.simulate import make_scene, _propagate, AisReport
from darkvessel.detect import cfar, Detection, estimate_texture_shape, k_threshold
from darkvessel.associate import associate, find_ghosts, predict_ais, PRESETS, Config
from darkvessel.evaluate import run_scene, summarise

G = SarGeometry()


def test_azimuth_shift_magnitude():
    # 12 kn straight across the look direction at 30 deg incidence
    shift = G.azimuth_shift_m(12 * 0.514444)
    assert 230 < abs(shift) < 260


def test_no_shift_for_along_track_motion():
    assert G.azimuth_shift_m(0.0) == 0.0


def test_ambiguity_offset_is_kilometres():
    assert 3000 < G.ambiguity_offset_m < 6000


def test_propagate_straight_line():
    x, y, c = _propagate(0, 0, 10.0, 90.0, 0.0, 100.0)
    assert x == pytest.approx(1000.0, abs=1e-6) and y == pytest.approx(0.0, abs=1e-6)


def test_propagate_backwards_is_inverse():
    x, y, c = _propagate(0, 0, 5.0, 30.0, 0.02, 300.0)
    x2, y2, _ = _propagate(x, y, 5.0, c, 0.02, -300.0)
    assert math.hypot(x2, y2) < 5.0


def test_texture_shape_estimate_orders_sea_states():
    calm = make_scene(seed=3, sea_state="calm", n_ships=0, size_px=512).image
    rough = make_scene(seed=3, sea_state="rough", n_ships=0, size_px=512).image
    assert estimate_texture_shape(calm, 3) > estimate_texture_shape(rough, 3)


def test_k_threshold_rises_with_spikier_clutter():
    assert k_threshold(1.5, 12, 1e-6) > k_threshold(12.0, 12, 1e-6)


def test_cfar_finds_isolated_bright_target():
    rng = np.random.default_rng(0)
    img = rng.gamma(3, 1 / 3, (256, 256)).astype(np.float32)
    img[128:131, 100:103] += 60.0
    dets, _ = cfar(img, 5.0)
    assert any(abs(d.x_m - 505) < 15 and abs(d.y_m - 645) < 15 for d in dets)


def test_cfar_few_false_alarms_on_pure_clutter():
    clutter = make_scene(seed=7, sea_state="moderate", n_ships=0, size_px=1024).image
    dets, _ = cfar(clutter, 5.0)
    assert len(dets) <= 3


def test_doppler_correction_moves_prediction_to_apparent_position():
    rep = AisReport(0, 0.0, 1000.0, 1000.0, 6.0, 90.0)   # moving purely in range
    with_d = predict_ais(rep, G, PRESETS["full"])
    without = predict_ais(rep, G, PRESETS["dead_reckoning"])
    assert with_d[1] - without[1] == pytest.approx(G.azimuth_shift_m(6.0), rel=1e-6)


def test_ghost_is_flagged_and_parent_kept():
    parent = Detection(1000, 2000, 300.0, 300.0, 40, 200.0)
    ghost = Detection(1010, 2000 + G.ambiguity_offset_m, 8.0, 8.0, 6, 60.0)
    assert find_ghosts([parent, ghost], G) == [1]


def test_hungarian_assignment_is_one_to_one():
    dets = [Detection(0, 0, 50, 50, 4, 20), Detection(60, 0, 50, 50, 4, 20)]
    ais = [AisReport(0, 0, 10, 0, 0, 0), AisReport(1, 0, 55, 0, 0, 0)]
    res = associate(dets, ais, G, PRESETS["full"])
    assert sorted(res.pairs.values()) == [0, 1] and res.dark == []


def test_unmatched_detection_is_dark():
    dets = [Detection(0, 0, 50, 50, 4, 20), Detection(5000, 5000, 50, 50, 4, 20)]
    ais = [AisReport(0, 0, 10, 0, 0, 0)]
    res = associate(dets, ais, G, PRESETS["full"])
    assert res.dark == [1]


def test_full_pipeline_beats_dead_reckoning_on_false_alarms():
    rows = {"dead_reckoning": [], "full": []}
    for seed in range(3):
        _, _, out = run_scene(make_scene(seed=100 + seed, sea_state="moderate"),
                              {k: PRESETS[k] for k in rows})
        for k in rows:
            rows[k].append(out[k][1])
    a, b = summarise(rows["dead_reckoning"]), summarise(rows["full"])
    assert b["false_alarms_per_scene"] < a["false_alarms_per_scene"]
    assert b["precision"] > a["precision"]
