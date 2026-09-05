# Current production architecture

This document describes the only supported runtime path. Historical BallNet and tuning
experiments are not runtime fallbacks.

## Input contract

- Fixed, elevated, behind-the-baseline monocular video; singles court and both baselines
  should be visible.
- Every valid native frame rate is accepted. Frames are neither dropped nor interpolated.
- Native 29.97/30fps is the required regression floor; higher rates cost proportionally
  more inference time.

## One-way data flow

```text
native video
  -> one multi-frame court calibration + per-frame player geometry
  -> frozen RacketVision MS-TrackNetV3 candidates
  -> persistent physical-ball association
  -> constrained trajectory and short-gap smoothing
  -> hit/contact candidates
  -> competing racket-hit and ground-contact evidence
  -> sub-frame touchdown and court-zone mapping
  -> annotated video, JSON report and 3D viewer
```

Tracking owns the ball trajectory. Landing logic may read it but must never create,
delete, or move a detector observation. Rendering is downstream of both modules.

## Display-only perspective correction

After selecting a video, the browser offers either the original camera view or a guided
perspective-correction preview. The user can choose 0–70% interactively and, when needed,
remark the four doubles-court baseline corners. The normalized corners and strength are
stored with the durable job identity. `vision/display_correction.py` applies that fixed
homography only in the final rendering pass. Inference, court registration, tracking and
landing classification always receive the untouched source frames. A corrected clean clip
is emitted alongside the corrected annotated replay so the report's overlay toggle does
not jump between two geometries. After the projective transform, a court-shaped safe frame
covering the complete court and both baseline-player bands is automatically fitted back
into the original output resolution. Irrelevant roof or venue corners are allowed to crop;
they are not allowed to shrink the match into a thumbnail near the projective vanishing
line. The browser preview uses the same composed transform. The minimap is composited
after correction and stays crisp.

## Ball lifecycle

- A coherent multi-frame hypothesis is required to create a ball track.
- Perspective-aware search and a Kalman gate associate reachable candidates.
- A missed frame coasts with increasing uncertainty instead of deleting the ball.
- Direction reversal requires player/racket evidence; bounce is a separate motion mode.
- RTS smoothing and the monocular gravity model only repair short confirmed gaps. Raw
  detector coordinates remain available for audit.
- A track ends only after uncertainty is exhausted, a supported net stop, or an outward
  edge exit.

## Landing contract

- Contact evidence combines trajectory impulse, adjacent flight arcs, court geometry,
  player/racket proximity, and optional audio timing.
- `events/contact_hypothesis.py` scores the hit and bounce explanations against each other. A
  player-body overlap alone cannot create a hit, and ambiguous evidence preserves the
  upstream label for audit instead of forcing a new event.
- Audio cannot create a landing or provide its spatial coordinate.
- A rejected main-track fragment may restore a landing only when the public detector's
  candidates form a coherent ground-bounce followed by a receiver-racket contact. The
  fragment remains outside the purple trajectory. A racket contact without the preceding
  ground-turn motif is treated as a possible volley and cannot create a yellow flash.
- Touchdown is estimated between native frames, then mapped through the fixed camera's
  calibrated court homography.
- Yellow highlighting starts at `decision_frame`, after touchdown confirmation.
- One racket strike arms one visible first landing. A second bounce before the next hit
  ends the point and does not create another highlight or minimap marker.
- In-court landings highlight the mapped court zone; out balls show a red cross. The
  minimap contains only the current rally's confirmed landings.

## Current fixed regression

`assets/demo/demo.mp4`: 1280×720, 1737 frames, 29.9139fps.

| Counter | Current value |
|---|---:|
| Public-model candidate frames | 1301 |
| Positioned trajectory frames | 1220 |
| Detector-backed observations | 1123 |
| Smoothed short-gap frames | 97 |
| Confirmed bounces | 28 |
| Racket hits | 32 |
| Fixed calibration court-line error | 1.23px |

These counters detect regressions; they are not manually labelled accuracy metrics.
Manual landing review windows live in
`tests/fixtures/manual_landing_annotations_v1.json`.

## Ownership map

| Concern | Owner |
|---|---|
| RacketVision inference | `src/netcast_tennisvision/vision/racketvision.py` |
| Court registration | `src/netcast_tennisvision/vision/court_registration.py` |
| Ball lifecycle and association | `src/netcast_tennisvision/tracking/world_tracker.py` |
| Geometry, smoothing, ballistics, trail | `src/netcast_tennisvision/tracking/` |
| Contact and touchdown | `src/netcast_tennisvision/events/` |
| Tennis sequence audit | `src/netcast_tennisvision/events/bounce_sequence.py` |
| Orchestration and rendering | `notebooks/tennis_detection.ipynb` |
| Upload service and UI | `src/netcast_tennisvision/api/server.py`, `web/` |

Any algorithm change requires tests, the full native-rate demo regression, and review of
the manual timestamp windows. UI-only work must not alter inference or event data.

## Fixed-camera court policy

The court is calibrated once from a clean plate assembled across multiple sampled frames.
That single geometry is reused for every native video frame; ball detection, player
segmentation, tracking and contact classification still execute at full frame rate. The
legacy per-frame court fitter is opt-in through `TENNISVISION_FIXED_COURT=0` for explicit
moving-camera diagnostics and is not the production default.

On the fixed `demo.mp4` regression and RTX 3050 Ti, the accepted fixed-court path reduced
end-to-end processing from 231.48s to 128.75s. It preserved 1220 positioned trajectory
frames, 27 bounces and 31 racket hits. All touchdown times remained within 0.02s; median
landing displacement was 0.05m and one landing 0.08m from an internal service line changed
the displayed adjacent zone.

The recall layer added afterwards leaves those 1220 main-trajectory frames unchanged. On
the same regression it restores one coherent rejected-candidate fragment: a far-backcourt
touchdown at 25.91s followed by the receiver's racket contact, producing 28 bounces and 32
hits. Three opposite-player contact intervals lacking physical ground evidence remain
landing-free as possible volleys. Set `TENNISVISION_RALLY_RECOVERY=0` for an exact rollback.

## Court-calibration fallback

The nine-line automatic calibration is accepted without user interaction when its
multi-frame support, temporal consensus and paint-fit error produce confidence >= 0.72.
Below that threshold the job pauses and asks for four points in the fixed order near-left,
near-right, far-right, far-left. The submitted quadrilateral passes the same geometry and
homography validation as automatic proposals before analysis resumes.
