# L40S RTMP pose A/B — 2026-09-21

## Finding

The L40S kept up in all four short runs. Adding the optional pose sequence component
did not cause sustained queue growth in this test. Pooled batch-last-frame analysis
latency rose from 315.7 to 353.1 ms (37.4 ms), and p95 from 700.2 to 767.6 ms.
Run-to-run variation is substantial; this is not a statistically precise overhead
estimate or a long-duration live SLA. Both pose runs detected only one of two reviewed
serves, so recognition still needs work even though compute capacity appears sufficient.

## Setup

- One temporary PPIO `L40S.22c125g`, actual device `NVIDIA L40S`.
- Base image `production-v30`; current local Python source uploaded into this disposable
  container. No published image, production setting or running app was replaced.
- Source: `demo3_clip`, original seconds 45–80, native 30.00667 fps, 1920×1080.
  Excerpt encoded with x264 CRF 18 without frame-rate conversion.
- Real path: FFmpeg `-re` producer on GPU host -> existing external ZLMediaKit RTMP
  server -> maintained external-stream live worker -> ball/player/track inference.
  The worker uses FFmpeg decoding to 640×360, ball batch 16 and person stride 4.
- Models: frozen RacketVision, YOLO11n-seg, optional YOLO11n-pose with 320-pixel
  crops, at approximately 10 Hz when scheduled. Pose is warmed before streaming.
- Same confirmed court corners for every run; live causal background initialization.
  No ball, player or pose inference cache. Only arrived frames reach the pose detector.
- Order: **off, on, on, off**. On-mode adds scheduling, crops, pose inference and
  serve sequence detection synchronously after the maintained rolling tracker.
  It does not change the worker's existing landing/rally publication behavior.
- Hooks exist only in the experiment process through exact, checked source anchors.
  Production runtime files and interfaces remain unchanged.

## Results

| Run | Pose | Batch-last delay p50 / p95 | Peak backlog | Ingested / inferred | Queue drops | Pose crops | Pose work | Confirmed serves |
|---|---|---|---:|---|---:|---:|---:|---:|
| 1 | off | 263.8 / 543.3 ms | 0.833 s | 1017 / 1015 | 2 | 0 | 0 | — |
| 2 | on | 353.4 / 698.6 ms | 0.833 s | 1017 / 1015 | 2 | 137 | 0.770 s | 1 |
| 3 | on | 352.8 / 815.5 ms | 0.900 s | 1017 / 1015 | 2 | 137 | 0.742 s | 1 |
| 4 | off | 341.4 / 808.2 ms | 0.800 s | 1013 / 1013 | 0 | 0 | 0 | — |

Pooled percentiles combine steady batches across the two runs for each setting.
The first five ingested seconds are excluded from delay percentiles, but full-run
drops and peak backlog are retained. On-mode first/second-half medians were
426/305 ms and 366/331 ms: no evidence of accumulating delay over these excerpts.

The recorded delay ends after tracking and optional pose work for the latest frame
in a batch, before landing publication. Its clock starts at the live worker's causal
stream initialization, not at the original camera's hardware timestamp. Earlier frames
within each 16-frame batch also wait up to approximately 0.5 seconds for batching.
These numbers therefore are **not** every-frame latency, serve-decision latency or
browser glass-to-glass latency. Browser WebRTC rendering and the ECS result relay
were not exercised; result collection used an authenticated experiment endpoint.

All runs also ingest fewer frames than the approximately 1050-frame source excerpt,
because RTMP connection/start/end handling is outside the inference queue counter.
Do not call the experiment lossless: queue drops alone are 0–2, but transport/startup
coverage is incomplete. Neither mode showed the local experiment's 40% queue loss.

The reviewed source has serves near 61.19 and 67.88 seconds. Both on runs confirmed
one event at ingested frame 657, consistent with the later serve after allowing for
startup offset. This is not a verified one-to-one source timestamp mapping and is not
a broader accuracy score. The earlier serve remains unconfirmed; this experiment
does not establish whether resolution, causal tracking, scheduling or the sequence
thresholds caused that miss.

## Evidence and reproduction

- `outputs/pose_cloud_live/result.json`: all four terminal worker snapshots, per-batch
  records, pose counts, serve candidates and remote log tail.
- `outputs/pose_cloud_live/summary.json`: per-run latency summaries.
- `outputs/pose_cloud_live/payload.zip`: exact executed source and uploaded excerpt/model.
- `outputs/pose_cloud_live/instance.json`: `released: true` for the completed instance.
- `outputs/pose_cloud_benchmark.log`: progress and successful resource release.
- `outputs/pose_cloud_tests.log`: complete pytest suite passed; Ruff passed.

Raw serve evidence from this run has absolute `frame` and `decision_frame` but
window-local preparation/release fields. The tool now offsets all four fields for
future runs; this reporting-only correction does not change recognition or timing.

From the repository root, with local model weights, archived input and PPIO credentials:

```powershell
.venv\Scripts\python.exe tools/benchmark_pose_cloud.py
```

The tool provisions one temporary instance, uploads a checksummed payload through a
bearer-authenticated bootstrap, runs the fixed experiment and deletes the instance in
`finally`. It journals the instance ID and refuses to start another if its previous
journal is unreleased. Earlier bootstrap failures were also released. A 30-minute
container process timeout supplements provider cleanup; it does not replace deletion.

Conclusion: keep pose optional. The short cloud test supports real-time compute with
modest additional latency; improve serve recall and run a longer, complete client-visible
live test before making pose the production default.
