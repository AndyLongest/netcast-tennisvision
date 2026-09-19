# Current production architecture

This document describes the only supported runtime path. Historical BallNet and tuning
experiments are not runtime fallbacks.

## Input contract

- Fixed, elevated, behind-the-baseline monocular video; singles court and both baselines
  should be visible.
- Every valid native frame rate is accepted. Frames are neither dropped nor interpolated.
- Native 29.97/30fps is the required regression floor; higher rates cost proportionally
  more inference time.
- The live-lab requires one guided four-corner confirmation before provisioning L40S.
  The normalized near-left, near-right, far-right and far-left points are validated on
  the relay and scaled to the actual stream dimensions by the worker. Uploaded live tests
  must not silently inherit Demo corners or a previous camera profile.
- Manual four-corner confirmation (offline and live) checks only four finite points,
  a strictly convex cyclic boundary and a nonsingular perspective mapping. It imposes
  no screen-height, width, depth, area, spacing, far/near ratio, sideline ratio or
  centre-drift thresholds. Near/far labels follow the user's click order, not screen
  height. Coordinates outside the video frame are also allowed; live transport
  still expresses coordinates relative to frame width and height.
  Automatic court proposals retain their existing false-positive filters.

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
  -> JSON report and 3D viewer
  -> browser event overlay + optional WebGL perspective correction (normal)
  -> annotated video (rollback)
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
- A near-player contact may open a three-observation outgoing-launch hypothesis. Its
  displacement is validated in calibrated court metres with a generous tennis-speed
  ceiling, rather than rejected by one uniform image-space radius. Only a certified
  launch may initialize the high-speed post-contact state.
- RTS smoothing and the monocular gravity model only repair short confirmed gaps. Raw
  detector coordinates remain available for audit.
- A track ends only after uncertainty is exhausted, a supported net stop, or an outward
  edge exit.

This contact-aware court-metric tracker is the only supported association path for both
offline analysis and live inference. Earlier circular-search, image-speed-only and pure
ballistic variants are summarized in `experiments/BALL_TRACKING_PATH.md`; none remains as
a runtime fallback.

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
| Positioned trajectory frames | 1268 |
| Detector-backed observations | 1192 |
| Smoothed short-gap frames | 76 |
| Confirmed bounces | 27 |
| Racket hits | 33 |
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
- A/B-to-side mapping is selected by confidence-weighted tracklet consensus and frozen
  for the complete rally. Appearance noise cannot switch marker colours during a point;
  a different mapping may only take effect after a rally boundary.
- A confirmed first landing uses its court half to corroborate the preceding hitter: a
  normal touchdown was struck from the opposite half. This repairs a single-frame racket
  proximity error without changing the ball track, touchdown position or landing class.
  The correction source and whether the hit side disagreed remain exported for audit.
- Player presentation colour is sampled from the central torso crop, aggregated across
  frames under the stable ReID label, and emitted downstream as `player_color`. It never
  participates in ball tracking, landing detection or near/far court geometry.

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

Every offline play mode now enters association with only the historical largest heatmap
component at the public 0.5 threshold. Low-threshold alternatives remain available to
research APIs but are not a production fallback. The whole-video fixed-component map was
removed because separate rallies revisit the same image regions and were being erased.
Training mode still uses its perspective-tolerant lifecycle: three reachable observations
are required for birth and a dead feed releases after 0.45 seconds. Singles and doubles
retain their longer match occlusion and fragment-join parameters.

## Fixed-camera court policy

The court is calibrated once from a clean plate assembled across multiple sampled frames.
That single geometry is reused for every native video frame; ball detection, player
segmentation, tracking and contact classification still execute at full frame rate. The
legacy per-frame court fitter is opt-in through `TENNISVISION_FIXED_COURT=0` for explicit
moving-camera diagnostics and is not the production default.

The accepted fixed-court mechanism was originally isolated on the RTX 3050 Ti at 128.75s
versus 231.48s for per-frame fitting. Those historical counters are preserved in
`docs/experiments/QUASI_REALTIME_EXPERIMENT.md`; they are not a selectable production
generation.

The current frozen demo uses the single contact-aware court-metric tracker and the
physically supported rejected-fragment recall pass. It contains 1268 positioned frames,
1192 detector-backed anchors, 76 short-gap predictions, 27 confirmed bounces and 33 racket
hits. Three opposite-player contact intervals lacking physical ground evidence remain
landing-free as possible volleys. There is no environment switch back to an earlier
tracker or cache-dependent recall generation.

## In/out line-call policy

Line calls consume the confirmed touchdown and never move it. The legal singles or
doubles rectangle uses the ITF outside edge of the painted lines. A physical ball radius
is included because touching a line is in. Touchdown and court-fit pixel uncertainty are
propagated through the inverse homography at that exact image location, so a far-court
pixel is not treated as the same number of metres as a near-court pixel.

The exported `line_call` is `in`, `out`, or `review`. An automatic call is made only when
the full uncertainty interval agrees. A close call whose interval crosses the boundary is
`review`: it receives neither a yellow in-zone flash nor a red out cross. This preserves
useful automatic calls on limited-quality video without fabricating centimetre precision.
The implementation is isolated in `events/line_call.py` and cannot alter tracking,
touchdown timing, or landing coordinates.

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

For a fixed-camera job, the accepted automatic or manual quadrilateral remains the
reference. `vision/court_motion.py` registers the first native frame and then once
every 300 seconds of source time (offline and live) to that reference using
spatially distributed ORB/RANSAC matches, excluding fixed broadcast graphics. A measured
reframing carries the same confirmed corners into the current image; it never refits or
snaps the user's corners to other court lines. An unchanged view preserves exact reference
coordinates; insufficient matches retain the last geometry and are recorded as unverified.
Between corrections the last geometry and verification state are reused causally;
failed matches also wait 300 seconds before retrying. Camera movement can therefore
leave overlays misaligned until the next correction, as selected by the user. All
ball frames and native presentation timestamps are still retained. Registration is
cached against the video identity, reference image, exact corners and interval.
Per-frame paint contrast cannot revoke manual geometry or suppress tracking. Pass-A also
includes the rounded calibration in its cache signature.

Offline reports export `court_keyframes` so the browser projects zones with the current
frame's geometry, including after seeking. The confirmation dialog previews service and
singles lines before submission. Ball observations remain owned by tracking. Contact fits
express evidence in a common registered image plane, preventing camera translation from
being treated as a ball impulse. Touchdown refinement searches a continuous quadratic
change point within three frames of the proposal, requiring two-sided detector support,
an upward impulse and improvement over smooth flight. Tennis-rule consistency alone is
not physical evidence of a landing.

`events/rallies.py` assigns shared IDs after landing confirmation. Out/net/second-bounce
outcomes close points; discarded candidates cannot bridge inactivity. The exported
`rallies` timeline controls both the browser and baked minimap: the next hit clears old
markers immediately, and dead time expires the preceding map after two seconds. For edited broadcasts, an optional visual score-panel detector also marks persistent
numeric-column changes. It currently recognizes an opaque blue two-row panel at the
lower left, excludes the speed badge/player names and requires 0.35 seconds of stable
evidence. Unsupported layouts use contact segmentation; no OCR or new serve model is
claimed. Missed/false contacts and unsupported score panels can still require review.

## Report delivery and video rendering

The interactive report is published as soon as `scene3d.json` and `rally3d.html` exist.
The normal cloud command uses `TENNISVISION_OUTPUT_MODE=event-overlay`: the browser plays
the local source and draws confirmed current-rally landing points, out crosses and the
latest yellow landing zone from `scene3d.json`. If perspective correction is enabled,
WebGL applies the same saved homography, protected player bands, safe scale and translation
used by `vision/display_correction.py` before the event layer is drawn. Both modes complete
at the former `report_ready` boundary and do not render or download a duplicate MP4.
Set `TENNISVISION_OUTPUT_MODE=annotated-video` for the former behavior; in that mode
`report_ready` remains active and resumable while encoding finishes.

PPIO transport uses four concurrent, independently checksummed 8 MiB upload parts with bounded retry.
Before creating an instance, the relay reads the selected PPIO product's live
`minRootFS`/`maxRootFS` constraints and clamps its conservative 60 GB target into that
range. If inventory changes between lookup and creation, a size-validation rejection is
retried once at the newly reported limit; such a rejected request has not allocated a
GPU and is not billable. A running job is reattached by video fingerprint after a
browser refresh; the temporary GPU itself is released when that job completes or fails
so idle capacity is never kept merely for reuse.
The remote service assembles all verified parts before entering the same native-rate
analysis endpoint. Generated MP4 files return through concurrent 8 MiB byte ranges and
are written at their original offsets before the final size check. This keeps every request
below the provider HTTP gateway's large-body risk boundary and prevents one network
interruption from retransmitting an entire match. Set
`TENNISVISION_TRANSFER_WORKERS=1` for the exact serial rollback.

Production cloud releases use `Dockerfile.release`: a thin code overlay on the audited
`production-v1` ML runtime. The legacy image stores frozen weights in
`/opt/netcast/models`; the release maps that directory to the canonical `/app/models`
path and refuses to publish unless the ball, bounce, player-segmentation and
player-identity checkpoints are all present. The relay currently pins `production-v29`. Short PPIO control-plane TLS
disconnects while an existing instance starts are retried until the startup deadline;
instance creation itself is never blindly retried because that could allocate two GPUs.

The internal live-lab uses a different transport contract from offline uploads:
FFmpeg publishes native-rate RTMP to ZLMediaKit and the GPU worker independently pulls
that stream. In the reverse direction, L40S publishes authenticated JSON snapshots to a
separate result-relay service on the same ECS; the trusted local service reads results
only from that relay. ZLMediaKit never stores inference JSON, and there is deliberately no
direct L40S-to-local result fallback. On L40S, the accepted
`production-v11` worker preserves every native ball frame, batches sixteen unchanged
RacketVision inputs per CUDA call, and runs player segmentation every fourth frame while
causally holding the last real player boxes between samples. It queues decoded ndarrays
directly in a 0.75-second bounded queue and JPEG-encodes only the throttled browser preview.
When inference falls behind, the oldest unprocessed image is discarded and represented as
an empty observation on the original source timeline; memory and wall-clock latency cannot
grow without bound during a 10–30 minute stream. The 1,737-frame demo held a
stable 0.3–0.5 second queue after warm-up and completed in 59.409 seconds for 58.067 seconds
of source; the prior every-frame person/four-frame-batch worker required 102.446 seconds.
`TENNISVISION_LIVE_BATCH_SIZE` and `TENNISVISION_LIVE_PERSON_STRIDE` are bounded rollback
controls. These settings apply only to the causal live worker; the offline production
report continues to use its documented full-frame person path.

The live-lab browser may upload a different test clip to the trusted local relay. The
file stays on that camera-simulator host and is never uploaded to the inference worker.
After a temporary L40S loads its frozen models and reports `awaiting_stream`, local
FFmpeg publishes the clip at native speed to ZLMediaKit. The L40S receives only the
unique stream name and pulls the same RTMP feed a production camera would expose.
Status and events are written to ECS under the local session id and pulled back by the
trusted local service. A relay outage fails visibly instead of bypassing ECS. Completion,
failure and explicit stop terminate the local producer, release the instance, and delete
that session's ECS result snapshot after its final state has been persisted locally. The
relay also removes snapshots older than 24 hours as a crash-only safety net, so the ECS
cannot silently accumulate experiment results.

For an external RTMP source, decoder EOF is treated as a reconnectable signal loss rather
than proof that the camera session ended. In the finite pseudo-camera experiment, the local
relay sends an explicit `/api/live-lab/finish` only after its FFmpeg publisher exits with
status zero; only then may the L40S drain the stream and mark the session complete. A remote
`complete` received while the publisher is still alive is rejected as a premature result.

Rendering first performs a one-frame NVENC preflight. A usable NVIDIA encoder receives the
unchanged rendered frames with the `p4`/CQ20 quality profile; otherwise an on-demand cloud
job falls back to x264 `veryfast`, CRF 22 and `faststart`. This affects only MP4 compression; it does
not rerun or alter detection, tracking, identity, or landing results. Set
`NETCAST_VIDEO_ENCODER=x264` for the exact software-encoder rollback and
`NETCAST_X264_CRF=20` for the previous cloud bitrate. A 30-second 2880x1620 `deemo2`
review segment fell from 28.35 MB to 19.32 MB at CRF22 with SSIM 0.9924; resolution,
frame rate and H.264 compatibility stayed unchanged. On the 2880x1620,
2779-frame `deemo3.mp4` benchmark, final rendering fell from 370s to 239s (35.4% faster).
The before/after `scene3d.json` SHA-256 remained identical:
`c93f53b6748cb0f543ebf148202d7b879ed915b5c8dbf2c1e301044d9f2e3f5e`.

The cloud transport and fixed-camera A/B is recorded in
`docs/experiments/CLOUD_ACCELERATION_2026-09-16.md`. The accepted v8 path reduced the
same `deemo2` complete time from about 598s to 498s. CRF22's complete cloud timing remains
pending because the next provider allocation was rejected for insufficient balance.

## Verified fixed-camera profiles

An accepted calibration is now remembered as runtime data in
`data/camera_profiles.json`. A later upload reuses it only when ORB feature matching and
RANSAC prove that several early frames come from the same unmoved camera. The current
frame must have at least 20 geometric inliers, an inlier ratio of at least 0.55 and less
than 3.5 pixels of mean alignment movement on a 640-pixel-wide verification image.
Otherwise the normal full calibration and optional four-point confirmation remain in
force. `TENNISVISION_CAMERA_PROFILES=0` disables reuse immediately.

The on-demand relay copies the local profile store into a newly cleaned worker before the
video and downloads its updated store afterwards. Transfer is limited to 3 MiB and is
best-effort; failure simply performs a fresh calibration. Camera evidence never enters the
container image and is released with the temporary instance.

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
`docs/experiments/PERFORMANCE_EXPERIMENT_2026-09-15.md`.

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

Ball-backend A/B on the RTX 3050 Ti measured a four-frame-batch PyTorch FP16 call at
124.4 ms. ONNX Runtime CUDA FP16 took 796.6 ms (6.4× slower) with maximum heatmap
difference 0.000381. The advertised TensorRT provider could not load because its runtime
libraries were absent and fell back to CPU. Neither backend is used. The optional harness
is `tools/benchmark_ball_backends.py`; its dependencies are not application dependencies.

Skipping baked replay composition is independent of those rejected inference changes.
On local `demo`, event-overlay execution ended at 134.4s without rendering. On recorded
cloud `deemo2`, the equivalent report boundary was 304.4s versus 497.8s complete: a
projected 193.3s / 38.8% reduction in user-visible wait. The old cache-dependent demo
discrepancy is retired; the shipped manifest and clean production route now share the
1268/1192/76 trajectory counters and 27-bounce/33-hit event result.

### Uncached end-to-end timing

On 2026-09-15, `deemo3.mp4` (2880x1620, 2,779 frames, 200.3s) was run with an empty
ball/Pass-A/audio cache and an already verified fixed-camera profile. The interactive
report became available after 484.09s (8m04s). The unchanged-resolution annotated MP4
finished after 753.19s (12m33s), including 269s of background rendering. The resulting
scene retained 782 tracked frames, 8 bounces and 4 racket hits, and its SHA-256 remained
exactly `c93f53b6748cb0f543ebf148202d7b879ed915b5c8dbf2c1e301044d9f2e3f5e`.

Offline reports preserve decoder presentation timestamps in `frame_times` and contact `decision_t`. Browser frame selection uses this timeline, not only frame index divided by nominal fps. This preserves initial offsets and internal gaps without adding or interpolating ball observations.
