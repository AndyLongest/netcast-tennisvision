# Developer handoff

This document is the shortest path from a fresh checkout to a safe algorithm change.

Run `.\setup.ps1` once, then read `docs/HANDOFF.md`. Do not rely on a globally installed
Python, an existing Jupyter kernel or files outside this repository.

The two demo videos are versioned under `assets/demo/`. Runtime model weights are excluded
from Git and are prepared by `setup.ps1` according to `assets/MODELS.md`.

## 1. Runtime entry points

- `run_ui.ps1` starts the local product UI.
- `netcast_tennisvision.api.server` receives one uploaded video and launches
  `netcast_tennisvision.pipeline.runner`.
- `pipeline/runner.py` executes code cells from `notebooks/tennis_detection.ipynb` and
  reports progress to `data/job_status.json`.
- `api/server.py` stores durable upload identity separately in `data/current_job.json`.
  Pipeline progress updates are merged with that identity by `/api/status`, allowing a
  refreshed or reopened browser to resume the one active job without uploading it again.
- The notebook orchestrates detection and rendering. Reusable algorithms belong in Python
  modules, not in new notebook-only helper functions.

The browser and server compare a fast content fingerprint made from the file size and its
first/last 256 KiB. Selecting the same video while it is running reattaches to the existing
job; selecting a different video never replaces the active clip. A completed matching job
opens its existing report directly.

## 2. Data flow and ownership

`frames_meta` is the frame-level exchange object. Important trajectory fields are:

| Field | Owner | Meaning |
|---|---|---|
| `candidates` | detector pass | all detector proposals before temporal association |
| `ball_px_raw` | tracker | accepted detector coordinate, or `None` |
| `ball_px` | tracker | constrained display/downstream coordinate |
| `ball_seen` | tracker | true only for detector-backed observations |
| `ball_state` | tracker | `observed`, `occluded_predicted`, or `unacquired` |
| `ball_track_id` | tracker | persistent physical-ball identity |
| `ball_confidence` | tracker | detector and motion agreement, not calibrated probability |
| `ball_motion_mode` | tracker | `flight`, `player_hit`, or `bounce` |

Event modules may read these fields but must not rewrite detection ownership or invent
`ball_seen=True` frames.

## 3. Tracking lifecycle

1. A three-observation hypothesis confirms birth.
2. Kalman state predicts one frame; perspective-aware gates rank reachable candidates.
3. Player proximity opens a shadow launch hypothesis; three coherent outgoing points
   certify a racket direction change.
4. Short missing intervals coast with decaying confidence; compatible fragments may be
   merged using future evidence.
5. RTS smoothing fills the confirmed segment while real measurements remain hard anchors.
6. Offline cleanup repairs proven 1–2 frame reverse spikes.
7. The calibrated gravity model may replace only synthetic points inside short gaps.

## 4. Where to make a change

| Change | File |
|---|---|
| Birth, association, search radius, termination | `tracking/world_tracker.py` |
| Court projection, camera pose, player/racket reach | `tracking/geometry.py` |
| Kalman/RTS output and isolated zigzag cleanup | `tracking/smoothing.py` |
| High-ball/occlusion physics | `tracking/ballistics.py` |
| Purple history trail only | `tracking/trail_rendering.py` and `TRAIL_RENDER_MODE` |
| Contact impulse scoring | `events/landing_event_detector.py` |
| Hit-versus-bounce evidence fusion | `events/contact_hypothesis.py` |
| Sub-frame touchdown location | `events/landing_detector.py` |
| Auto/manual court-confidence policy | `vision/court_calibration.py`, `pipeline/runner.py`, `api/server.py` |
| Tennis sequence audit | `events/bounce_sequence.py` |
| UI only | `web/` and rendering cells; do not alter tracking evidence |

The purple trail experiment is deliberately reversible. Set `TRAIL_RENDER_MODE` to
`"legacy"` in the notebook configuration to restore the original renderer without
removing code or changing any analysis output.

## 5. Safe change procedure

1. Add a minimal synthetic test that fails for the reported behavior.
2. Change one owning module; avoid compensating thresholds in unrelated modules.
3. Run `.\.venv\Scripts\python.exe -m pytest -m "not assets and not integration"`.
4. Run the complete native-30fps baseline sample through
   `python -m netcast_tennisvision.pipeline.runner`; add representative native-rate
   regressions without dropping frames when available.
5. Compare at least ball coverage, real observations, player-hit resets, repaired-frame
   count, and the user-reported timestamp clips.
6. Reject a change that raises one metric by sacrificing detector-backed observations.
7. Update `docs/CURRENT_ARCHITECTURE.md` and the reference mapping when behavior changes.

Current frozen-demo regression: 1220 positioned frames and 1123 detector-backed
observations on the 1737-frame sample, with 29 confirmed bounces and 35 racket hits.
These are preservation counters, not independent accuracy claims.

The optional quasi-realtime research path and its rejected A/B variants are documented
in `docs/QUASI_REALTIME_EXPERIMENT.md`. It is disabled by default because simple player
or court frame skipping changed contact classification.

## 6. Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -m "not assets and not integration"
.\.venv\Scripts\python.exe -m pytest -m assets
.\.venv\Scripts\python.exe -m ruff check .
node --check web\app.js
```

The suite separates tracking, landing impulse detection, and sub-frame touchdown fitting.
New physics behavior should be tested using a synthetic calibrated camera so expected 3D
motion is known rather than eyeballed.

## 7. Research discipline

Research papers and citation metadata live under `docs/references/`. Each adopted idea must
state its observation assumptions. Multi-camera or high-frame-rate results are not evidence
that a method works under this project's single-camera regression baseline. Every valid native frame rate is accepted; higher rates are higher-cost, higher-temporal-resolution inputs, not requirements.
