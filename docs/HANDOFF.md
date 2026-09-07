# Engineering handoff entrypoint

This is the canonical first page for a new engineer or coding agent. The product owner
should not need to explain the system again.

## Product in one paragraph

Netcast TennisVision is a local-first, fixed-camera tennis analysis application. A user uploads a
behind-the-baseline match video. The backend runs frozen computer-vision models at the
video's native frame rate, establishes one stable court geometry, tracks one physical
ball through short occlusions, distinguishes racket hits from ground contacts, maps
confirmed touchdowns into court zones, and produces an annotated replay, current-rally
minimap and interactive 3D report. The browser is intentionally non-technical.

## Supported production path

```text
web/app.js
  -> netcast_tennisvision.api.server (single resumable local job)
  -> netcast_tennisvision.pipeline.runner
  -> notebooks/tennis_detection.ipynb orchestration
  -> vision/racketvision.py + tracking/world_tracker.py
  -> vision/player_identity.py (OSNet-AIN labels only; never feeds ball tracking)
  -> events/contact_hypothesis.py + events/landing_event_detector.py
  -> events/landing_detector.py + events/bounce_sequence.py
  -> data/outputs/{annotated_clip.mp4,scene3d.json,rally3d.html}
```

Only the RacketVision MS-TrackNetV3 candidate path is supported. Uploaded videos are pure
inference. Never restore the retired per-video BallNet training path.

## First commands

Windows, from the repository root:

```powershell
.\setup.ps1
.\run_ui.ps1
.\.venv\Scripts\python.exe -m pytest -m "not assets and not integration"
.\.venv\Scripts\python.exe tools\release_check.py --mode handoff
```

The demo is already bundled. If only the models are supplied as an offline handoff folder:

```powershell
.\setup.ps1 -AssetSource D:\Netcast-TennisVision-models
```

Model source, automatic download/conversion and offline installation are documented in
`assets/MODELS.md`. The project must remain movable as one `Tennis_Vision/` folder; runtime
code may not depend on its parent workspace.

## Frozen facts

- Input regression floor: native 29.97/30fps; every positive native frame rate is accepted.
- Frames are never dropped or interpolated.
- Fixed demo: 1737 frames, 1220 positioned ball frames, 1123 detector anchors,
  29 confirmed bounces and 35 racket hits.
- Court calibration is performed once for a fixed camera and reused for every frame.
- A yellow zone appears only after a confirmed in-court touchdown.
- An out ball creates a red cross; the minimap contains only the current rally.
- One strike arms at most one first landing; volleys do not invent a bounce.
- Refreshing/reopening the browser reattaches to the active backend job.

## Ownership boundaries

| Area | Files | Non-negotiable rule |
|---|---|---|
| Candidate generation | `vision/racketvision.py` | frozen public weight, no upload-time training |
| Player identity | `vision/player_identity.py` | sparse OSNet-AIN embeddings, joint A/B assignment, three-sample side-change hysteresis |
| Association and lifecycle | `tracking/world_tracker.py` | tracking owns ball observations |
| Smoothing and physics | `tracking/` | real detections remain hard anchors |
| Contact classification | `events/contact_hypothesis.py`, `events/landing_event_detector.py` | hit and bounce compete; audio is timing-only |
| Landing position | `events/landing_detector.py` | consumes trajectory, never edits it |
| Tennis ordering | `events/bounce_sequence.py` | a volley is allowed; a bounce is not mandatory |
| Orchestration | `pipeline/runner.py`, notebook | preserve native frame rate and frozen parameters |
| Local product | `api/server.py`, `web/` | non-technical UI and resumable single job |

## Definition of done for a change

1. Change only the owning layer.
2. Add a failing test before or with the fix.
3. Run the unit suite.
4. For algorithm changes, run the complete native-rate demo and review every timestamp in
   `tests/fixtures/manual_landing_annotations_v1.json`.
5. Compare the production counters and detector-backed observations.
6. Update the relevant contract document.
7. Run the handoff release check.

## Where to continue

`docs/NEXT_STEPS.md` is the prioritized backlog. Historical thresholds in `docs/history/`
are evidence, not supported runtime alternatives. `docs/PUBLICATION_CHECKLIST.md` contains
legal/distribution blockers and must be completed before making the repository public.
