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
  -> temporal player-count vote (1+1 singles, 2+2 doubles, otherwise training)
  -> frozen RacketVision MS-TrackNetV3 candidates
  -> persistent physical-ball association
  -> court-aware temporal player tracks + sparse OSNet-AIN identity
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
| Confirmed bounces | 29 |
| Racket hits | 35 |
| Fixed calibration court-line error | 1.23px |

These counters detect regressions; they are not manually labelled accuracy metrics.
Manual landing review windows live in
`tests/fixtures/manual_landing_annotations_v1.json`.

## Ownership map

| Concern | Owner |
|---|---|
| RacketVision inference | `src/netcast_tennisvision/vision/racketvision.py` |
| Court registration | `src/netcast_tennisvision/vision/court_registration.py` |
| Player identity and landing ownership | `src/netcast_tennisvision/vision/player_identity.py` |
| Ball lifecycle and association | `src/netcast_tennisvision/tracking/world_tracker.py` |
| Geometry, smoothing, ballistics, trail | `src/netcast_tennisvision/tracking/` |
| Contact and touchdown | `src/netcast_tennisvision/events/` |
| Tennis sequence audit | `src/netcast_tennisvision/events/bounce_sequence.py` |
| Orchestration and rendering | `notebooks/tennis_detection.ipynb` |
| Upload service and UI | `src/netcast_tennisvision/api/server.py`, `web/` |

Any algorithm change requires tests, the full native-rate demo regression, and review of
the manual timestamp windows. UI-only work must not alter inference or event data.

## Player identity contract

- The feet of every person detection are projected into fixed court coordinates. At most
  one active player is selected on each half; implausible short-term jumps are rejected
  rather than treated as a new player.
- OSNet-AIN appearance is sampled every fifth native frame, but it only proposes A/B.
  A side swap needs three strong paired observations *and* enough elapsed time for both
  people to travel between their measured court positions at a 12m/s upper bound.
- Missing or ambiguous samples preserve the last stable identity and confidence. They do
  not create a new identity and cannot make a landing marker change colour by themselves.
- A confirmed landing inherits the stable identity at the preceding racket contact. Only
  a clip whose contact occurred before its first frame uses the opposite-landing-half
  fallback.

## Play-mode routing

Person-box feet are projected into the accepted court coordinate system every fifth frame.
Counts are aggregated across the whole clip with a 65th-percentile presence vote, so a
short occlusion does not turn doubles into singles and a rare extra box does not turn
singles into training. Stable 1+1 is `singles`, stable 2+2 is `doubles`, and every other
asymmetric or multi-person arrangement is `training`. The selected mode and evidence
distribution are exported as `scene3d.json.play_mode` and shown in the report conclusion.
Contact geometry retains the inferred number of active people on each side; this is what
allows a coach plus two trainees to contribute racket evidence instead of discarding one
trainee under the old one-player-per-half assumption.

Training mode also selects a separate candidate/lifecycle policy without changing the
validated singles or doubles path. RacketVision keeps the historical largest heatmap
component at its public 0.5 threshold, plus at most seven spatially distinct alternatives
down to 0.30. Long-lived fixed components form a loose-ball map and are removed before
association. A training feed still needs three temporally reachable observations to be
born, but its minimum image displacement is perspective-tolerant and a dead feed releases
the active state after 0.45 seconds so the next ball can start. Match mode discards every
alternative and retains its original birth, occlusion and fragment-join parameters.

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
touchdown at 25.91s followed by the receiver's racket contact, producing 29 bounces and 35
hits. Three opposite-player contact intervals lacking physical ground evidence remain
landing-free as possible volleys. Set `TENNISVISION_RALLY_RECOVERY=0` for an exact rollback.

## Court-calibration fallback

The nine-line automatic calibration is accepted without user interaction when its
multi-frame support, temporal consensus and paint-fit error produce confidence >= 0.72.
Below that threshold the job pauses and asks for four points in the fixed order near-left,
near-right, far-right, far-left. The submitted quadrilateral passes the same geometry and
homography validation as automatic proposals before analysis resumes.

The manual preview is selected from the full calibration sample rather than frame zero.
Frames containing a detected court take priority, with visible contrast and spatial detail
used as tie-breakers; if automatic court detection finds nothing, the clearest non-black
sample is shown. This prevents a black intro or fade-in from producing an unusable marking
canvas.

For a fixed-camera job, an accepted automatic or manual quadrilateral is authoritative for
every decoded frame. Per-frame paint contrast may control which line fragments are drawn,
but it cannot revoke the court coordinate system or suppress ball tracking. The rounded
four-corner geometry is part of the Pass-A cache signature, so a corrected calibration
never reuses court metadata produced for different corners.

## Report delivery and video rendering

The interactive report is published as soon as `scene3d.json` and `rally3d.html` exist.
The job enters `report_ready`, the browser opens the report against the original video,
and the annotated MP4 continues rendering in the background. `report_ready` is an active,
resumable job state: refreshing reconnects to it and another upload cannot overwrite its
files. When encoding finishes, the player switches to the annotated video at the same
playback time and enables the original/annotated toggle.

Rendering uses x264 `veryfast`, CRF 20 and `faststart` by default. This affects only MP4
compression; it does not rerun or alter detection, tracking, identity, or landing results.
Set `NETCAST_X264_PRESET=medium` to restore the earlier encoder setting. On the 2880x1620,
2779-frame `deemo3.mp4` benchmark, final rendering fell from 370s to 239s (35.4% faster).
The before/after `scene3d.json` SHA-256 remained identical:
`c93f53b6748cb0f543ebf148202d7b879ed915b5c8dbf2c1e301044d9f2e3f5e`.

## Verified fixed-camera profiles

An accepted calibration is now remembered as runtime data in
`data/camera_profiles.json`. A later upload reuses it only when ORB feature matching and
RANSAC prove that several early frames come from the same unmoved camera. The current
frame must have at least 20 geometric inliers, an inlier ratio of at least 0.55 and less
than 3.5 pixels of mean alignment movement on a 640-pixel-wide verification image.
Otherwise the normal full calibration and optional four-point confirmation remain in
force. `TENNISVISION_CAMERA_PROFILES=0` disables reuse immediately.

The profile stores the trusted normalized four corners and the clean grayscale court
plate losslessly. It does not refit or modify the corners on later clips. On `deemo3`,
the fixed-camera lookup matched with 1,049 inliers, 0.994 inlier ratio and 0.178 pixels
alignment movement. The lookup itself took 1.19s; the notebook's cached calibration path
took approximately 8.3s instead of 119.3s for a fresh multi-frame court search. The
production trajectory/report hash remained exactly
`c93f53b6748cb0f543ebf148202d7b879ed915b5c8dbf2c1e301044d9f2e3f5e`.

## Ball-inference preprocessing

RacketVision now converts every resized input frame from HWC `uint8` to CHW `float32`
once when the frame enters the four-frame window. Previously the same frame was converted
again for each overlapping window. On the 1,737-frame 720p `demo`, model-loop throughput
rose from 23.2 FPS to 24.2 FPS (about 4.3%), while candidate presence, coordinates and
confidence values were exactly equal.

A bounded two-batch prefetch queue additionally prepares the next native-rate batch on
the CPU while the GPU processes the current batch. It does not change frame order, model
inputs or thresholds. On the same `demo`, the inference loop rose from 23.6 FPS to 33.2
FPS, and total ball-stage time including fixed setup fell from 83.0s to 62.6s (24.6%). All
1,737 candidate rows were exactly equal. Set `TENNISVISION_RACKETVISION_PREFETCH=0` for
an immediate rollback to serial preparation.

Full-rate person segmentation uses the same bounded-overlap principle: one worker
prepares the next unchanged YOLO result batch while the main thread expands, dilates and
PNG-compresses the prior masks. A four-run crossover on `demo` reduced this isolated
stage from 42.09s to 28.31s (1.49×), with identical hashes for every player box and mask.
The integrated notebook's person portion of Pass A improved more modestly, from 44.75s to
41.18s, because other per-frame CPU work competes for the same resources. The complete
structured report also remained byte-identical to the serial control. Set
`TENNISVISION_PERSON_PREFETCH=0` for an immediate rollback. End-to-end wall time remains
noisy; timings and rejected alternatives are recorded in
`docs/PERFORMANCE_EXPERIMENT_2026-09-15.md`.

RacketVision's deterministic 180-frame median background is sampled by four independent
decoder instances, then restored to source-frame order before the median. On `demo` this
reduced background construction from 9.44s to 4.70s; the median image and all 1,737 ball
candidate rows remained exact. Set `TENNISVISION_BACKGROUND_WORKERS=1` for the serial
rollback. `TENNISVISION_PARALLEL_DETECTORS=1` remains a server-only experiment: it keeps
the demo outputs exact, but the current 4GB development GPU loses most of the theoretical
gain to ball/person contention.

Person stride-two and TensorRT person experiments are deliberately not production
features. Stride two removed two racket-contact events on `demo`; TensorRT offered only
about 2.5% person-model gain on this RTX 3050 Ti and changed downstream events. Both were
removed. Production remains full-frame PyTorch person inference and native-frame-rate
RacketVision ball inference.

### Uncached end-to-end timing

On 2026-09-15, `deemo3.mp4` (2880x1620, 2,779 frames, 200.3s) was run with an empty
ball/Pass-A/audio cache and an already verified fixed-camera profile. The interactive
report became available after 484.09s (8m04s). The unchanged-resolution annotated MP4
finished after 753.19s (12m33s), including 269s of background rendering. The resulting
scene retained 782 tracked frames, 8 bounces and 4 racket hits, and its SHA-256 remained
exactly `c93f53b6748cb0f543ebf148202d7b879ed915b5c8dbf2c1e301044d9f2e3f5e`.
