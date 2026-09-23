# Pose pseudo-live latency experiment — 2026-09-21

## Decision

Do not enable pose in the deployed live worker based on this experiment. The local
baseline is already overloaded. Pose adds measurable work, but queue saturation,
dropped frames and run-to-run variation prevent a reliable estimate of its latency
impact on a healthy live pipeline. No production setting or cloud deployment changed.

## Method

- Windows, RTX 3050 Ti Laptop GPU, 4 GB VRAM, project `.venv`.
- Input: `data/history/2023a642c4a4431bb9dfb45494236965/source.mp4`
  (`demo3_clip`), source 50–75 seconds, 1920×1080 at 30.00667 fps.
- Calibration: the same directory's `scene3d.json`, court corners only.
- Real FFmpeg `-re` source pacing; 750 decoded frames per run. No cached ball or pose
  predictions. Models warmed before source arrival; startup excluded.
- Frozen RacketVision ball inference, batch 16, CUDA FP16; person model
  `yolo11n-seg.pt`, size 640, stride 4. Static median background is prepared from
  the source before timing, not constructed causally from the stream.
- Tracking and landing detection run on an eight-second rolling window. Pose uses
  the maintained scheduler and serve sequence detector on arrived frames only;
  `yolo11n-pose.pt`, player crops size 320, batches up to 8.
- A bounded 90-frame queue discards oldest frames on overload and explicitly counts
  them. This is an experiment overload policy, not a new production frame-drop rule.
- Primary delay is result completion minus the frame's source timeline timestamp,
  anchored to the first decoded arrival. Exclude the first three source seconds;
  include the final drain. It is not network, player-display or serve-confirmation
  latency. Only processed frames contribute to percentiles.
- This adapter does not reproduce RTMP/ZLMediaKit/WebRTC transport or full event
  publication/rendering. The deployed live worker does not yet invoke this pose path.

## Clean reverse pair

Run order: on, then off; no concurrent test suite or other agent-launched GPU work.
Both runs use identical configuration except pose.

| Metric | Pose off | Pose on |
|---|---:|---:|
| Source-to-result median | 4.013 s | 4.044 s |
| Source-to-result p95 | 7.457 s | 7.245 s |
| Maximum | 8.210 s | 7.839 s |
| Processed / received | 441 / 750 | 426 / 750 |
| Dropped | 309 (41.2%) | 324 (43.2%) |
| Pose samples | 0 | 53 |
| Pose scheduling, inference and sequence work | 0 | 0.734 s total |

Raw evidence: `outputs/pose_pseudolive_clean/run_0_True.json` and
`outputs/pose_pseudolive_clean/run_1_False.json`; log
`outputs/pose_pseudolive_clean.log`. Files contain per-frame delay records.

The median difference is +30.9 ms, with 15 fewer frames processed when pose is on.
The lower on-mode p95 is not evidence that pose speeds up analysis: different frames
survive the bounded queue and the run durations vary. These short paired measurements
do not establish statistical significance.

The pose work totals 0.734 seconds for 25 seconds of source, but only 53 crops were
processed. Do not extrapolate that to the cost of complete, lossless pose coverage.
There is no offline whole-video timeout silently turning off the component here.

Neither on run in this experiment confirmed a serve in the tested interval. The
same interval has two previously reviewed serves at approximately 61.19 and 67.88
seconds. This fails functional validation for this adapter; overload and missing
observations require investigation before attributing the failure to one cause.

An earlier exploratory off/on/on/off series produced off medians 4.079/3.957 seconds
and on medians 4.069/4.108 seconds, with 39.7–45.3% drops. Some test-suite execution
overlapped that series, so it is not the clean comparison. Raw outputs remain under
`outputs/pose_pseudolive/`.

## Reproduce

From the repository root in PowerShell, with verified local model weights installed:

```powershell
.venv\Scripts\python.exe tools/benchmark_pose_pseudolive.py `
  --video data/history/2023a642c4a4431bb9dfb45494236965/source.mp4 `
  --calibration data/history/2023a642c4a4431bb9dfb45494236965/scene3d.json `
  --out outputs/pose_pseudolive_clean --order on off
```

Next acceptance experiment must use the intended live GPU and actual transport,
first establish a baseline without accumulating backlog, then compare pose off/on
with serve recall and drops alongside latency. Preserve the existing off switch.
This local experiment makes no claim about cloud L40S performance.
