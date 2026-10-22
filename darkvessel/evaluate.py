"""Score a run against the simulator's ground truth."""
import math
from collections import Counter

from .associate import associate, PRESETS
from .detect import cfar


def label_detections(dets, scene):
    """Truth label for each detection: ('ship', id) | ('ghost', parent) | ('clutter', None)."""
    labels = []
    for d in dets:
        best, best_d = None, 1e18
        for s in scene.ships:
            if not s.in_image:
                continue
            tol = max(50.0, 0.6 * s.length_m)
            dist = math.hypot(d.x_m - s.app_x_m, d.y_m - s.app_y_m)
            if dist <= tol and dist < best_d:
                best, best_d = ("ship", s.id), dist
        if best is None:
            for gx, gy, pid in scene.ghosts:
                if math.hypot(d.x_m - gx, d.y_m - gy) <= 120.0:
                    best = ("ghost", pid)
                    break
        labels.append(best or ("clutter", None))
    return labels


def score(scene, dets, res, labels):
    ships = {s.id: s for s in scene.ships}
    alarm_kinds = Counter()
    tp_ships = set()
    for i in res.dark:
        kind, ref = labels[i]
        if kind == "ship" and ships[ref].dark:
            alarm_kinds["true_dark"] += 1
            tp_ships.add(ref)
        elif kind == "ship":
            alarm_kinds["ais_ship"] += 1      # association failure
        else:
            alarm_kinds[kind] += 1            # ghost or clutter

    detected_ships = {ref for kind, ref in labels if kind == "ship"}
    dark_in_image = {s.id for s in scene.ships if s.in_image and s.dark}
    ais_detected = {sid for sid in detected_ships if not ships[sid].dark}

    correct_pairs = sum(1 for i, sid in res.pairs.items()
                        if labels[i] == ("ship", sid))
    wrong_pairs = sum(1 for i, sid in res.pairs.items()
                      if labels[i][0] == "ship" and labels[i][1] != sid)

    n_alarms = len(res.dark)
    return dict(
        alarms=n_alarms,
        true_dark=alarm_kinds["true_dark"],
        false_from_ais_ships=alarm_kinds["ais_ship"],
        false_from_ghosts=alarm_kinds["ghost"],
        false_from_clutter=alarm_kinds["clutter"],
        dark_ships=len(dark_in_image),
        dark_ships_detected=len(dark_in_image & detected_ships),
        dark_ships_found=len(tp_ships),
        ais_ships_detected=len(ais_detected),
        correct_pairs=correct_pairs,
        wrong_pairs=wrong_pairs,
    )


def run_scene(scene, configs=None, **cfar_kw):
    configs = configs or PRESETS
    dets, _ = cfar(scene.image, scene.geom.pixel_m, **cfar_kw)
    labels = label_detections(dets, scene)
    out = {}
    for name, cfg in configs.items():
        res = associate(dets, scene.ais, scene.geom, cfg)
        out[name] = (res, score(scene, dets, res, labels))
    return dets, labels, out


def summarise(rows):
    """Aggregate per-scene score dicts into rates."""
    tot = Counter()
    for r in rows:
        tot.update(r)
    alarms = max(tot["alarms"], 1)
    return dict(
        scenes=len(rows),
        alarms=tot["alarms"],
        precision=tot["true_dark"] / alarms,
        recall_of_detected=tot["dark_ships_found"] / max(tot["dark_ships_detected"], 1),
        recall_overall=tot["dark_ships_found"] / max(tot["dark_ships"], 1),
        false_alarms_per_scene=(tot["alarms"] - tot["true_dark"]) / len(rows),
        false_from_ais_ships=tot["false_from_ais_ships"] / len(rows),
        false_from_ghosts=tot["false_from_ghosts"] / len(rows),
        false_from_clutter=tot["false_from_clutter"] / len(rows),
        pairing_accuracy=tot["correct_pairs"] / max(tot["ais_ships_detected"], 1),
        wrong_pairs_per_scene=tot["wrong_pairs"] / len(rows),
        detector_recall_dark=tot["dark_ships_detected"] / max(tot["dark_ships"], 1),
    )
