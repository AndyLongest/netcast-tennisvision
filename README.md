# Netcast TennisVision

> Handoff status: the source tree is being prepared for a new engineering owner. Local
> development is supported; public redistribution is still blocked by the licensing
> decisions in [docs/PUBLICATION_CHECKLIST.md](docs/PUBLICATION_CHECKLIST.md).

Netcast TennisVision analyzes a fixed, elevated, behind-the-baseline tennis video and produces a
tracked ball trajectory, player/court geometry, bounce events, highlighted landing zones,
and an interactive 3D report. The validated regression baseline is **native 29.97/30fps**,
but uploads accept every valid native frame rate. Every frame is analyzed: higher rates
cost proportionally more compute, and frame interpolation is never required.

## Start here

For a new engineer on Windows:

```powershell
.\setup.ps1
.\run_ui.ps1
```

`setup.ps1` creates `.venv`, installs the validated Python/PyTorch stack, downloads or
rebuilds the three exact checksummed models, and runs an environment audit. The fixed
source and annotated demo videos already travel with the repository. For model provenance,
download details and offline installation, read [assets/MODELS.md](assets/MODELS.md).

If the machine is offline, copy the three final model files from an installed machine and
use:

```powershell
.\setup.ps1 -AssetSource D:\Netcast-TennisVision-models
```

For a non-technical operator on Windows:

```powershell
.\run_ui.ps1
```

Then open the shown local address, upload a video, or choose the bundled example. Outputs
are written to `data/outputs/`:

- `annotated_clip.mp4` — annotated review video;
- `scene3d.json` — machine-readable frame and event data;
- `rally3d.html` — self-contained interactive 3D viewer.

For development:

```powershell
.\.venv\Scripts\python.exe -m pytest -m "not assets and not integration"
.\.venv\Scripts\python.exe -m pytest -m assets
.\.venv\Scripts\python.exe -m netcast_tennisvision.pipeline.runner
.\.venv\Scripts\python.exe tools\release_check.py --mode handoff
```

The runner executes the maintained notebook pipeline and updates `data/job_status.json`
for the web interface.

## Current validated sample

The fixed 1280×720 demo contains 1737 frames at 29.91fps. The current frozen
RacketVision pipeline produces:

| Metric | Result |
|---|---:|
| Frames with a ball position | 1220 / 1737 |
| Detector-backed observations | 1123 |
| Short occlusion predictions | 97 |
| Mid-flight reverse spikes repaired | 4 |
| Confirmed ground bounces | 28 |
| Racket hits | 32 |

These are pipeline regression counters, not independently labelled accuracy metrics.
They prevent a refactor from silently losing observations but do not prove line-calling
accuracy.

## Architecture

```text
video
  -> court/player/ball candidates (notebook pass A, cached)
  -> persistent single-ball association (tracking/world_tracker.py)
       -> camera and court geometry (tracking/geometry.py)
       -> Kalman + RTS segment smoothing (tracking/smoothing.py)
       -> calibrated 3D gravity model for missing frames only (tracking/ballistics.py)
  -> contact candidates (events/landing_event_detector.py)
       -> competing racket-hit / ground-contact evidence (events/contact_hypothesis.py)
  -> sub-frame touchdown fit (events/landing_detector.py)
  -> tennis sequence audit (events/bounce_sequence.py)
  -> annotated video + JSON + 3D viewer (notebook pass B)
```

The dependency direction is intentional: landing logic consumes the trajectory and may
not create or move detector observations. The 3D ballistic model is advisory and only
replaces synthetic positions inside short gaps; it never admits or rejects a real ball
candidate.

### Tracking modules

| File | Responsibility |
|---|---|
| `tracking/world_tracker.py` | Stable public API and single-ball lifecycle/association |
| `tracking/geometry.py` | Homographies, perspective scale, player/racket envelopes |
| `tracking/smoothing.py` | Forward Kalman pass, RTS backward pass, conservative cleanup |
| `tracking/ballistics.py` | Robust monocular 3D gravity fit and reprojection checks |
| `tracking/trail_rendering.py` | Reversible, screen-space stabilization of the purple history trail |
| `tracking/types.py` | Diagnostics returned by a tracking pass |

### Event modules

| File | Responsibility |
|---|---|
| `events/landing_event_detector.py` | Finds physically supported contact impulses |
| `events/contact_hypothesis.py` | Fuses competing racket-hit and ground-contact explanations |
| `events/landing_detector.py` | Estimates touchdown time and position between 30fps samples |
| `events/bounce_sequence.py` | Audits tennis hit/bounce order without suppressing visible evidence |

## Invariants for future changes

- A single detection cannot create a ball; birth requires a coherent multi-frame track.
- A missed detection does not make the ball disappear immediately.
- Real detections are hard anchors and remain available as `ball_px_raw`.
- A mid-flight direction reversal requires racket/player evidence; a bounce is a separate
  motion mode.
- Ballistic fitting must not participate in candidate acceptance until a separately
  validated multi-hypothesis tracker is available.
- Yellow court highlighting starts only after a confirmed ground contact.
- Automatic court calibration is silent when confidence is high. Low-confidence clips
  pause for a guided four-corner confirmation instead of failing or guessing.
- One racket strike can arm only one landing highlight; a second bounce before the next
  hit ends the point and cannot create another yellow zone or minimap marker.
- All time windows are defined from the actual video fps; 30fps remains the regression floor, not an upload gate.

## Documentation

- [New engineer/Agent handoff](docs/HANDOFF.md)
- [Current architecture and algorithm rules](docs/CURRENT_ARCHITECTURE.md)
- [Repository structure and dependency rules](docs/PROJECT_STRUCTURE.md)
- [Developer handoff guide](docs/DEVELOPMENT.md)
- [Prioritized next steps](docs/NEXT_STEPS.md)
- [Local API and report schemas](docs/API_AND_SCHEMAS.md)
- [Publication blockers](docs/PUBLICATION_CHECKLIST.md)
- [Local research library and implementation mapping](docs/references/README.md)
- [Runtime output ownership](docs/OUTPUTS.md)
- [Historical experiments](docs/history/)

## Repository layout

```text
Tennis_Vision/
├── README.md
├── pyproject.toml
├── run_ui.ps1
├── setup.ps1
├── src/netcast_tennisvision/
│   ├── api/          # local HTTP service
│   ├── pipeline/     # production orchestration
│   ├── vision/       # inference and court calibration
│   ├── tracking/     # lifecycle, geometry, physics and smoothing
│   └── events/       # contact, touchdown and tennis rules
├── notebooks/
├── tests/
├── tools/           # optional diagnostics and one-off developer utilities
├── web/
├── docs/
├── assets/          # tracked demo videos/report plus checksummed model manifest
├── tests/fixtures/  # small, versioned regression labels and manifests
├── models/          # downloaded/generated weights; ignored by git
├── outputs/         # experiments and benchmarks; ignored by git
└── data/            # latest upload, regression fixtures and caches; ignored by git
```

`Tennis_Vision/` is the complete project boundary. Production code must not read files from
its parent or sibling directories. Moving or copying this folder is supported; recreate
`.venv` with `setup.ps1` on the destination machine rather than copying the virtual
environment.

### Court-registration diagnostic

When a new fixed-camera clip fails before tracking begins, run only the calibration
stage (the ball and person models are not loaded):

```powershell
python tools/diagnose_court_registration.py path\to\clip.mp4
```

The detector combines cross-ratio line hypotheses with a dominant-playing-surface
proposal, strict non-degenerate court geometry, balanced support from all nine ITF
paint lines, temporal-medoid consensus, and a geometry-constrained global refit. A
generic colour-edge score is deliberately not used: walls and floor seams can create
stronger colour discontinuities than a distant painted baseline.

## Known limits

This is a monocular, fixed-camera, single-ball, singles-oriented pipeline. A ball that is
only a few pixels wide, fully hidden for a long interval, or absent outside the frame
cannot be recovered honestly. Drag and Magnus forces are documented in the research
library, but spin is not fitted from a 5–9 frame monocular gap because it is not reliably
observable under the current monocular input conditions.
