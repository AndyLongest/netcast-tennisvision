# Agent handoff rules

Read these files, in order, before changing runtime behavior:

1. `README.md`
2. `docs/HANDOFF.md`
3. `docs/CURRENT_ARCHITECTURE.md`
4. `docs/PROJECT_STRUCTURE.md`
5. `docs/DEVELOPMENT.md`
6. `docs/NEXT_STEPS.md`
7. `docs/RACKETVISION_PRODUCTION.md`
8. `assets/MODELS.md`

Do not ask the product owner to restate information already recorded in those files.

## Production contract

- Start the product with `run_ui.ps1`.
- Treat this `Tennis_Vision/` directory as the complete project boundary. Runtime code
  must never read a parent or sibling path.
- Do not add Python modules to the repository root. Production code belongs under
  `src/netcast_tennisvision/` and follows `docs/PROJECT_STRUCTURE.md`.
- `src/netcast_tennisvision/api/server.py` accepts uploads;
  `src/netcast_tennisvision/pipeline/runner.py` runs the maintained notebook.
- Production ball detection is the frozen RacketVision MS-TrackNetV3 weight at
  `models/racketvision_balltrack_state_v1.pt`. Never train on an uploaded video.
- Preserve native input frame rate. Do not silently drop or interpolate frames.
- Tracking owns ball positions. Landing code may consume the trajectory but may not move,
  create, or delete ball observations.
- Yellow zones and minimap markers appear only after a confirmed touchdown. Out balls use
  a red cross; the minimap shows only the current rally.

## Where work belongs

- Candidate inference: `src/netcast_tennisvision/vision/racketvision.py`
- Association/lifecycle: `src/netcast_tennisvision/tracking/world_tracker.py`
- Geometry, smoothing, ballistics, trail display: `src/netcast_tennisvision/tracking/`
- Landing/contact logic: `src/netcast_tennisvision/events/`
- Pipeline orchestration/rendering: `notebooks/tennis_detection.ipynb`
- Local app/API: `src/netcast_tennisvision/api/`, `web/`
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
