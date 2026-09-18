# Ball tracking experiment path

This is the single retained record of how the production ball path was selected. It is
evidence, not an alternate runtime. Production has one implementation:
`vision/racketvision.py -> tracking/world_tracker.py -> tracking/smoothing.py`.

## Decision summary

| Route tried | Characteristic | Advantage | Defect / reason for rejection |
|---|---|---|---|
| Per-clip BallNet adaptation | Re-trained or adapted around one clip | Could look strong on its source clip | No trustworthy labels, overfits one camera, upload-time training is slow and irreproducible |
| Public RacketVision candidate only | Uses the fixed MS-TrackNetV3 heatmap maximum independently | Open weights, deterministic, no user-video training | A detector miss deletes continuity; a near-camera strike can outrun a uniform pixel gate |
| Larger circular search | Expands around the previous pixel position | Recovers some brief misses | Perspective makes one pixel radius physically inconsistent; bright objects can hijack the track |
| Speed-adaptive image gate | Predicts from the last two image positions | Better for ordinary flight | A bounce or racket strike legitimately changes velocity; near/far pixel scales differ sharply |
| Pure ballistic extrapolation | Fits a smooth gravity arc through missing frames | Visually continuous for a clean unobstructed arc | Monocular height is ambiguous; cannot certify a hit, bounce, net stop, or an unseen ball by itself |
| Multi-model / low-threshold ensemble | Adds weak RacketVision components and other open detectors | Raises raw candidate coverage | Raised false positives and degraded the one-physical-ball constraint on match footage |
| **Contact-aware court-metric association (accepted)** | Fixed RacketVision candidates, player-contact launch hypotheses, court-metric plausibility and bounded smoothing | Recovers high-speed near-player returns without admitting arbitrary mid-flight reversals | Still depends on visible detector evidence near or shortly after contact; long fully hidden far-court flight remains uncertain |

## Accepted production route

1. RacketVision MS-TrackNetV3 runs as frozen pure inference at the native frame rate.
2. Singles and doubles retain only the public 0.5 primary heatmap component. Weak
   components remain restricted to the separately detected training mode.
3. A ball is born only from a coherent multi-frame hypothesis. One bright point cannot
   create a track.
4. Ordinary flight remains constrained by Kalman uncertainty, recent direction and a
   hard physical displacement budget.
5. A candidate near a tracked player can open a racket-contact launch hypothesis. Three
   coherent outgoing observations must point toward the opponent.
6. The launch step is checked through the fixed court homography in metres per second,
   with a deliberately generous 90 m/s ceiling. The homography is a plausibility guard,
   not a claimed ball-speed measurement.
7. Only after that launch is certified may the tracker accept the abrupt velocity change
   and initialize a high-speed state. The smoother can then preserve that state through a
   short occlusion instead of dragging it back to the pre-contact path.
8. Bounce, racket contact, net termination and outward frame exit remain distinct state
   transitions. A mid-flight reversal without event evidence is rejected.

There is no runtime switch to the retired association rules. The same tracker is imported
by offline reports and the live worker.

## Acceptance evidence

The five-minute 1920×1080, 30fps difficult clip contains 8,835 frames. With the conservative
public candidate only, the accepted tracker produced 5,083 detector-backed observations and
6,570 positioned frames. Manual review found the near-player outgoing flight substantially
more continuous than the prior gate.

On the fixed 1,737-frame demo, the same conservative candidate policy produced 1,243
detector-backed observations and 1,380 positioned screen frames in the focused tracker
harness. The former checked-in trajectory exposed 1,155 screen positions in that harness.
These are coverage counters, not labelled precision or recall; the full pipeline event
regression remains authoritative for bounce and hit counts.

The 2026-09-18 clean-workspace consolidation then deleted the notebook's per-clip BallNet
training/recovery cells and its single-detection legacy tracker rather than merely skipping
them by numeric cell index. With no historical BallNet weight or detection cache available,
the complete native-rate demo retained 1,268 positioned frames: 1,192 detector-backed and
76 bounded predictions, with the same 27 bounces and 33 hits. A warm complete report plus
annotated-video pass took 66.294s on the development GPU. Artifact bytes can still differ
after fresh automatic court calibration, so the handoff gate freezes counters and event
timestamps; byte equality is a separate same-environment check.

## Known limits

- A few near-player strikes still lack enough post-contact detector evidence. Their exact
  timestamps should be added as small reviewed regression fixtures before changing gates.
- Court homography projects an airborne ball onto the court plane. It is suitable for
  rejecting impossible teleports, not for reporting true 3D speed.
- A long interval with no visible candidate cannot be reconstructed honestly. Short-gap
  prediction is rendered separately from detector-backed observations for audit.
- The far court is resolution-limited. Product continuity may use a bounded physical
  bridge, but close line calls still require uncertainty-aware review rather than invented
  precision.

## Change discipline

Future work extends this one route. A proposed change must preserve the public candidate
policy, one-ball lifecycle and native-frame timeline; add a timestamp regression; run the
complete demo and unit suite; and report detector-backed observations separately from
predicted positions. Rejected variants are summarized above and must not return as hidden
fallbacks.
