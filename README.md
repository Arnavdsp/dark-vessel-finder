# dark-vessel-finder

A dark-ship alert is a SAR detection that no AIS report explains. Matched
naively, two radar effects make honest ships look dark:

1. **Doppler displacement.** A ship moving in range is focused at the wrong
   azimuth, offset by -(R/V)·v_g·sin θ. From a 520 km orbit at 30° incidence
   a 12 kn ship crossing the beam shows up about 244 m from its true position.
2. **Azimuth ambiguities.** Bright ships leave a weaker copy λ·R·PRF/(2V)
   along-track (about 4.3 km here). It has no AIS, because nothing is there.

This repo simulates both, detects ships with a K-distribution CFAR, and
measures how much each correction reduces false dark-vessel alerts.

## Results (simulation, 30 scenes per sea state, 45 ships per 10 × 10 km scene)

Moderate sea:

| pipeline | alerts truly dark | dark ships flagged* | false alerts / scene | AIS pairing |
|---|---|---|---|---|
| naive (last report, greedy) | 17% | 91% | 22.70 | 33% |
| dead reckoning | 34% | 95% | 9.47 | 86% |
| + Doppler shift | 36% | 96% | 8.53 | 93% |
| + age-based gate | 40% | 95% | 7.27 | 96% |
| + ghost removal (full) | 81% | 94% | 1.13 | 96% |

\* of the dark ships the detector saw.

Widening the dead-reckoning gate to 800 m only reaches 38% precision and
drops recall to 80%. Calm and rough sea results are in `results/metrics.json`.

Where it still fails: the detector finds 26% of boats under 20 m in moderate
sea and 2% in rough sea. Those are the boats most likely to be fishing with
AIS off, so the next step is a learned small-target detector.

## Layout

```
darkvessel/geometry.py   azimuth shift and ambiguity offset
darkvessel/simulate.py   K-distributed clutter, ships, ghosts, AIS (with spoofing)
darkvessel/detect.py     K-CFAR with scene-estimated texture shape
darkvessel/associate.py  dead reckoning, Doppler prediction, gates, ghosts, Hungarian
darkvessel/evaluate.py   truth labelling and metrics
scripts/run_eval.py      Monte Carlo over sea states -> results/metrics.json
scripts/export_scene.py  one scene for the viewer
scripts/build_page.py    builds web/dark-vessel-finder.html
```

## Run

```
pip install numpy scipy pillow pytest
python -m pytest -q
python scripts/run_eval.py 30
python scripts/export_scene.py 1009 moderate && python scripts/build_page.py
```

## Limits

All numbers come from my simulator, so read them as relative gains. Flat-earth
geometry, one incidence angle per scene, no wakes, no along-track defocus, no
land. Ghost removal can hide a real boat sitting exactly one ambiguity offset
from a large ship.

Next: run on xView3-SAR (Sentinel-1 with AIS-derived labels), take R, V and
incidence per pixel from the product annotation, and link detections across
passes.
