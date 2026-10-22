# dark-vessel-finder

A dark-ship alert is a radar detection that no AIS report explains. This
repo builds those alerts from Sentinel-1 radar and Danish AIS, removes the
ones that radar physics or fixed infrastructure explain, and ranks the rest
so an analyst looks at the ship-like targets first.

It started as a simulator (below) and now runs on 23 real Sentinel-1 IW GRD
frames over Danish waters from 12, 13, 14 and 19 October 2026: 13 passes,
466,000 km² of sea, 45,306 radar detections.

## What goes wrong with naive matching

1. **Moving ships are drawn in the wrong place.** A ship moving towards or
   away from the radar is placed along-track by -(R/V) · v_range · sin θ.
   On Sentinel-1 that is about 360 m for a 10 kn ship crossing the beam, so
   its AIS position and its radar blob don't line up and it looks dark.
2. **Bright ships leave ghosts.** Azimuth ambiguities appear λ·R·PRF/(2V)
   along-track, 4.6 to 5.9 km depending on the sub-swath. A ghost has no AIS.
3. **Fixed things are bright too.** Wind turbines, buoys, bridges and rocks
   have no AIS either.

Slant range, incidence, azimuth time and the PRF of each sub-swath are read
from the product annotation for every pixel, not assumed.

## Results on Sentinel-1 + Danish AIS

Scored on open water (more than 2 km from the coast) inside Danish AIS
reception (within 40 km of Denmark and closer to Denmark than to any other
country). Outside that zone a ship looks dark only because no Danish
receiver heard it.

**The Doppler correction checks out against the ships themselves.** For 640
moving ships with an AIS report less than a minute old and one detection
nearby, the measured along-track offset against the predicted one has slope
1.00 and r = 0.99. The median along-track error falls from 206 m to 31 m.

| pipeline | open-water alerts | of which next to an unpaired AIS ship | AIS ships paired |
|---|---|---|---|
| naive (last report, 300 m gate, greedy) | 8,980 | 349 | 62% |
| dead reckoning | 8,976 | 345 | 62% |
| + Doppler shift | 8,742 | 106 | 79% |
| + age-based gate | 8,740 | 95 | 80% |
| + ghost removal (96 ghosts) | 8,645 | 95 | 80% |
| + known structures (2,932 turbines/AtoN) | 5,713 | 74 | 80% |

"Next to an unpaired AIS ship" means within 1.5 km of an AIS vessel the
matcher left unpaired, which is almost always that ship drawn in the wrong
place. Those false alerts fall by 79%.

AIS ships paired with a radar detection, by AIS length (n = 1,362):

| length | n | naive | full |
|---|---|---|---|
| < 20 m | 270 | 47% | 48% |
| 20-40 m | 161 | 65% | 76% |
| 40-100 m | 288 | 78% | 93% |
| > 100 m | 599 | 62% | 92% |

Big ships gain most because they are fast. Boats under 20 m are found half
the time; that is the detector, not the matcher, and is the main open problem.

**Ghosts are real in this data.** Counting bright/faint pairs (12 dB or more
apart, within 40 m in range) at k times the predicted ambiguity offset:

| k | 0.6 | 0.7 | 0.8 | 0.9 | **1.0** | 1.1 | 1.2 | 1.3 | 1.4 |
|---|---|---|---|---|---|---|---|---|---|
| pairs | 44 | 46 | 32 | 34 | **127** | 16 | 13 | 20 | 19 |

**What I tried that didn't work.** Treating any target seen at the same spot
on another day as infrastructure removed 67% of alerts but also 41% of
AIS-paired ships (anchorages, ferry routes). Re-sighting is now a ranking
feature, and only charted turbines (OpenStreetMap) and AIS aids to
navigation are removed. That list touches 1% of AIS-paired ships.

### Ranking what is left

The 5,713 remaining alerts are mostly coastline edges, land texture and
wakes, with real point targets among them. There is no label for "dark
ship", but every AIS-paired detection is a ship, so a classifier trained to
separate AIS-paired detections from unexplained alerts ranks alerts by how
ship-like they look (positive-unlabelled learning). Features are brightness,
size, background texture, edge strength, compactness, distance to coast and
re-sighting. Trained on 12-14 October, tested on 19 October:

| | AUC | alerts left when 95% of AIS ships are kept |
|---|---|---|
| brightness only | 0.93 | 39% |
| triage model | 0.97 | 15% (311 of 2,068) |

The model is biased towards large ships: at that threshold it keeps 100% of
AIS ships over 40 m but 78% of those under 20 m (n = 58).

Each alert gets a card in `results/alerts.geojson`: score, length,
brightness over the local sea, nearest AIS vessel and its distance, gate
size, the ambiguity offset checked, distance to coast and to the nearest
charted turbine, and an image chip. Several top alerts sit 1-2 km from wind
farms and are probably offshore substations, which OpenStreetMap does not
tag as turbines. The rest are candidates for an analyst, not confirmed dark
vessels.

### Stale AIS

Danish coastal AIS is seconds old; AIS received from orbit over busy water
can be minutes old. `scripts/stale_ais.py` rebuilds each fix from reports at
least N minutes older than the image and matches moving ships (the ones the
full pipeline pairs on fresh data, so the first row is 100% by construction
for the age-based gate):

| AIS report age | ships | fixed 300 m gate | age-based gate |
|---|---|---|---|
| freshest | 1,616 | 99% | 100% |
| at least 2 min | 1,608 | 97% | 99% |
| at least 5 min | 1,598 | 88% | 94% |
| at least 10 min | 1,576 | 74% | 88% |

The age-based gate barely matters on coastal AIS and matters a lot once
reports are minutes old.

## Results in simulation

The simulator came first and is still the controlled experiment: K-distributed
clutter, ships with Doppler displacement and ghosts, and AIS reports that are
old, noisy, turning, missing or spoofed (3%). Moderate sea, 30 scenes, 45
ships per 10 × 10 km X-band scene:

| pipeline | alerts truly dark | dark ships flagged* | false alerts / scene |
|---|---|---|---|
| naive (last report, greedy) | 17% | 91% | 22.70 |
| dead reckoning | 34% | 95% | 9.47 |
| + Doppler shift | 36% | 96% | 8.53 |
| + age-based gate | 40% | 95% | 7.27 |
| + ghost removal | 81% | 94% | 1.13 |

\* of the dark ships the detector saw.

Simulation overstated what ghost removal buys: it had no coastline, turbines
or harbours, which dominate real alerts.

## Layout

```
darkvessel/s1.py         Sentinel-1 annotation: geolocation grid, slant range, PRF, Doppler shift, ambiguity offset
darkvessel/ais.py        AIS window, interpolation to each ship's imaging time, dead reckoning, hull-centre offset
darkvessel/landmask.py   Natural Earth land mask in image coordinates
darkvessel/detect.py     K-distribution CFAR with the texture shape estimated per tile
darkvessel/associate.py  gates, ghosts, Hungarian assignment (shared by simulation and real data)
darkvessel/real.py       real-data presets and scores
darkvessel/triage.py     chip features for alert ranking
darkvessel/simulate.py   simulator;  darkvessel/evaluate.py  its scoring
scripts/run_real.py      detect + place AIS for each frame -> results/real/
scripts/eval_real.py     score all frames -> results/real_metrics.json
scripts/train_triage.py  ranking model, held-out day -> results/triage.json
scripts/build_cards.py   alert cards -> results/alerts.geojson
scripts/stale_ais.py     matching with AIS reports 2-10 minutes old
scripts/run_eval.py      simulation Monte Carlo
```

## Run

```
pip install numpy scipy pandas rasterio pystac planetary-computer pyproj shapely pyshp scikit-learn pillow pytest
python -m pytest -q

# AIS: daily files from http://aisdata.ais.dk/, cut to the pass hours
#   unzip -p aisdk-2026-10-14.zip | awk -F, 'NR==1 || substr($1,12,2) ~ /^(05|06|16|17)$/' > data/ais/ais-2026-10-14-passes.csv
# land:   https://naciscdn.org/naturalearth/10m/physical/ne_10m_land.zip -> data/
#         https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip -> data/
# turbines (optional): OpenStreetMap generator:source=wind nodes -> data/osm_wind.json
python scripts/run_real.py --frames data/frames.json --jobs 4    # ~10 min per frame, streamed from Planetary Computer
python scripts/eval_real.py && python scripts/train_triage.py && python scripts/build_cards.py
```

Raster data is read in windows from Microsoft Planetary Computer; nothing
large is stored. `results/real/` holds detections, AIS fixes and 48 × 48
chips per frame.

## Data

- Contains modified Copernicus Sentinel data (2026), via Microsoft Planetary Computer.
- AIS from the Danish Maritime Authority's public archive. Raw AIS is not
  redistributed here; per-frame fixes are derived from it.
- Wind turbines © OpenStreetMap contributors (ODbL). Land and borders from Natural Earth (public domain).

## Limits

- No ground truth for "dark". Precision on real data is not claimed; the
  scores above are the ones AIS can check.
- AIS here is coastal and seconds old. Satellite AIS over busy water is
  older and patchier, which is what the stale-AIS run approximates.
- Sentinel-1 is C-band at 10 m pixels. Varuna-class X-band at about 1 m will
  find smaller boats and have its own ambiguity numbers; the geometry code
  takes them from the annotation, so the change is data, not code.
- Natural Earth coastlines are coarse; Swedish skerries and Wadden Sea banks
  leak through as coastal alerts. Offshore substations are not in the
  turbine list.
- Ghost removal can hide a real boat sitting exactly one ambiguity offset
  from a large ship.

Next: a learned small-boat detector on chips (tiled inference, Wasserstein
box matching), linking detections across passes, and running on xView3-SAR
for labelled dark/non-dark vessels.
