# Agent handoff rules

Read these files, in order, before changing runtime behavior:

1. `README.md`
2. `docs/README.md`
3. `docs/HANDOFF.md`
4. `docs/CURRENT_ARCHITECTURE.md`

Then read only the task-specific contract routed by `docs/README.md`. In particular,
use `docs/PROJECT_STRUCTURE.md` for file placement, `docs/DEVELOPMENT.md` before a code
change, `docs/RACKETVISION_PRODUCTION.md` for ball inference, and `assets/MODELS.md` for
weights. Do not load the experiment or history archives unless the task needs that evidence.

Do not ask the product owner to restate information already recorded in those files.

## Five-minute orientation

Before editing, answer these four questions from the repository itself:

1. **Is this product behavior or an experiment?** Production behavior is defined only by
   `docs/CURRENT_ARCHITECTURE.md`; files in `docs/experiments/` are evidence.
2. **Who owns the value being changed?** Use `src/netcast_tennisvision/README.md` and do not
   compensate for a defect by tuning a downstream layer.
3. **Which regression protects it?** Use `tests/README.md` to find the narrow suite, then
   run the complete handoff gate before transfer.
4. **Is the browser involved?** Read `web/README.md`; visual overlays consume structured
   results and must not silently reinterpret court coordinates or event timing.

If any answer is unclear, improve the owning contract as part of the change. Do not add a
second implementation merely to avoid understanding the maintained one.

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

`src/netcast_tennisvision/README.md` is the close-to-code responsibility map. Experimental
evidence belongs under `docs/experiments/`; retired behavior belongs under `docs/history/`.
Neither directory defines a production fallback.

## Required checks

Run `pytest -q` for every code change. Algorithm changes also require a full native-rate
`assets/demo/demo.mp4` regression and review of the timestamp windows recorded in
`tests/fixtures/manual_landing_annotations_v1.json`.

Do not commit runtime files under `data/`, generated reports under `outputs/`, virtual
environments, caches, or model weights.

Run `python tools/release_check.py --mode handoff` before handing work to another agent.
Public publication is separately blocked until every item in
`docs/PUBLICATION_CHECKLIST.md` is resolved.
