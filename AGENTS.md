# Agent handoff rules

Read these files, in order, before changing runtime behavior:

1. `README.md`
2. `docs/HANDOFF.md`
3. `docs/CURRENT_ARCHITECTURE.md`
4. `docs/DEVELOPMENT.md`
5. `docs/NEXT_STEPS.md`
6. `docs/RACKETVISION_PRODUCTION.md`
7. `assets/MODELS.md`

Do not ask the product owner to restate information already recorded in those files.

## Production contract

- Start the product with `run_ui.ps1`.
- Treat this `Tennis_Vision/` directory as the complete project boundary. Runtime code
  must never read a parent or sibling path.
- `server.py` accepts uploads; `pipeline_runner.py` runs the maintained notebook.
- Production ball detection is the frozen RacketVision MS-TrackNetV3 weight at
  `models/racketvision_balltrack_state_v1.pt`. Never train on an uploaded video.
- Preserve native input frame rate. Do not silently drop or interpolate frames.
- Tracking owns ball positions. Landing code may consume the trajectory but may not move,
  create, or delete ball observations.
- Yellow zones and minimap markers appear only after a confirmed touchdown. Out balls use
  a red cross; the minimap shows only the current rally.

## Where work belongs

- Candidate inference: `racketvision_runtime.py`
- Association/lifecycle: `temporal_world_tracker.py`
- Geometry, smoothing, ballistics, trail display: `tracking/`
- Landing/contact logic: `landing_event_detector.py`, `landing_detector.py`,
  `bounce_sequence.py`
- Pipeline orchestration/rendering: `notebooks/tennis_detection.ipynb`
- Local app/API: `server.py`, `web/`
- Regression tests: `tests/`

## Required checks

Run `pytest -q` for every code change. Algorithm changes also require a full native-rate
`assets/demo/demo.mp4` regression and review of the timestamp windows recorded in
`tests/fixtures/manual_landing_annotations_v1.json`.

Do not commit runtime files under `data/`, generated reports under `outputs/`, virtual
environments, caches, or model weights.

Run `python tools/release_check.py --mode handoff` before handing work to another agent.
Public publication is separately blocked until every item in
`docs/PUBLICATION_CHECKLIST.md` is resolved.
