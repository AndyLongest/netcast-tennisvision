# Pose-assisted contact experiment — 2026-09-22

Owner: events/contact_pose.py. Pure opt-in helper (enabled=False by default), no model
imports or I/O when unused. No production notebook, live worker or cloud image enabled.
Shared YOLO11n-pose weights were checked against vision/serve_pose.py's SHA256.
Experiment inference consumed native source crops at roughly 15Hz around contact
windows, imgsz=320. Ball tracking is unchanged from stationary_flat2/result.

## Evidence rule

Visible shoulder/elbow/wrist sequences are normalized by player height and shoulder
motion removed. At least three confident samples spanning both sides of the contact
are required for each arm. Low-confidence/missing joints stay unknown; stationary arms
alone cannot create a bounce. A quiet-arm proposal additionally needs a measured upward
impulse score >=0.75, BIC gain >=10 and a valid fractional touchdown fit. Contact proposal
timing can shift within +/-67ms, with pose rechecked at the selected impulse. A strong
swing near the ball prevents this correction. Racket length is approximated by reach;
this is not a racket detector or a trained tennis action classifier. Fast undersampled
swings and missing people remain risks: do not promote based on these two examples.

## Flat2 first 90 seconds

Cached pose-model outputs: outputs/contact_pose/poses.json. Raw inference took
21.3867s total, 798 person crops, 17.5985s inference; first prediction 2.4418s; warm
per-crop median 16.43ms / P95 32.42ms. This is offline additional work, not live latency.
No new model downloaded and no per-video training.

Five hit-to-bounce proposals were generated in the whole 90 seconds. After unchanged
later rules, report bounce count is 16 versus 14 in the stationary-prior baseline.
Counts do not establish accuracy. Specifically:

- 9.73s candidate -> 9.77s impulse: visible near-player arms quiet, strong ball impulse;
  reclassified upstream, but the existing one-landing-between-hits rule rejects it after
  an earlier 8.34s bounce. It is NOT a recovered final minimap marker.
- 69.8231s: recovered final bounce, decision 70.10s. It remains line_call=review because
  homography-propagated uncertainty is 3.18m. No definite yellow in-zone flash should be
  claimed; recognizing contact does not resolve low-view position uncertainty.
- 70.77s actual return: visible far-side wrist motion is identified as swing and retains
  hit classification.

The original contact audit also scored events before their impulse_score was populated;
this experiment computes impulse evidence before correction, so the effect is combined
pose/trajectory evidence, not an isolated pose-only accuracy gain.

Full native default high-view regression (1737 frames) exactly matches the pre-change
scene, including all 27 bounces, frame data and decision times. This proves off/default
preservation, not enabled-pose high-view accuracy. Synthetic tests cover off, missing
joints, smooth flight, visible swing, quiet arms with a real bounce and input immutability.
Full pytest and Ruff passed. The existing reviewed high-view timestamp windows are
identical. Full experimental MP4 decoded to 2700 frames, original audio, no trail.

## Live compute experiment scope

outputs/contact_pose/live_bench_isolated.py adapts the existing local source-paced
benchmark with causal proximity scheduling (only arrived frames, 15Hz pose crops),
pose_contact_evidence evaluation, fresh ball/person/pose inference and ABBA off/on/on/off.
Four 10-second runs cover source 66–76s. Models warm up before timing. This is a compute
adapter, not deployed RTMP/WebRTC; the causal sampling schedule differs from the offline
candidate-window schedule. It measures GPU contention, not event accuracy or cloud delay.
Initial measurements overlapping video encoding are excluded; use live_bench_isolated/.
The local baseline itself has sustained backlog and dropped frames, so this setup cannot
validate zero-backlog realtime or isolate a stable incremental delay on an unsaturated GPU.

Artifacts: outputs/contact_pose/result/review.html, flat2_contact_pose_90s.mp4,
corrections.json, timing.json, live_bench_isolated/run_*.json, check_targets.jpg.
No cloud deployment. Keep disabled pending more labelled contact cases and an
unsaturated-server live benchmark. Removing the optional caller fully exits this path.


Isolated ABBA measured source-to-result P50 seconds: off 6.993 / 4.863,
on 5.646 / 5.657. P95: off 11.774 / 7.796, on 8.690 / 8.653.
Dropped of 300 incoming frames: off 114 / 76, on 82 / 82. Pose work in on runs
was 1.661 / 1.384s for 85 crops each. Large baseline variability and loss mean no
credible fixed per-event incremental latency can be reported from these runs.
