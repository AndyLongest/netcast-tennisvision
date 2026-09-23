# Lightweight speed cloud experiment

Experimental `method=linear` fits the existing gravity-constrained ray equations
using one linear least-squares solve. It skips nonlinear optimization and ODE
integration. A finite-difference pixel Jacobian is used only to retain uncertainty
and conditioning gates. Existing pixel residual/height/speed/sensitivity gates
remain. Small epsilon fixes the numerical0.3s boundary. Default offline drag mode
is unchanged. No new models or package dependencies.

Cloud-only performance evaluation on NVIDIA L40S, flat2 first90s, ABBA off/on/on/off,
RTMP path and actual fresh ball/person/pose inference as earlier. Pose on throughout.
Latest16-frame support (~0.5s at30fps), three-frame maturity. One CPU thread handles
speed, at most one outstanding job and no pending queue. Busy submissions skipped.
Snapshots copied before submitting to prevent tracker mutation races. Completed
results drained on next batch and at shutdown. Per-run accepted result IDs are
not merged across runs. This is an experiment process hook, not deployed live code.

`published_at` in raw experimental events records compute completion, while the
next batch drains the result into the stream report. Thus `server_delay_ms` is
compute-complete age relative to the flight midpoint, NOT exact publish time.
`client_snapshot_age_ms` reflects the first remote HTTP snapshot in which the
collector sees it, including drain/progress/poll waits; remaining network transit
and browser paint are excluded. Raw scope string inherited from the prior adapter
incorrectly says synchronous: implementation is asynchronous, as described here.

Actual final artifacts: outputs/speed_cloud_linear/{remote.py,provision.py,payload.zip,
result.json,client_events.json,summary.json,instance.json,review.html}.
No speed ground truth: throughput and accepted-result count do not prove accuracy.
Pure-gravity approximation can bias fast or spinning shots. No production toggle
or cloud image changed. Complete pytest and default high native-rate regression
pass; all non-speed scene fields unchanged.

## Completed results

| Run | Enabled | Batch median/P95 ms | Queue drops | Speeds/delivered |
|---|---|---|---|---|
| 1 | False | 205.5/357.6 | 0 | 0/0 |
| 2 | True | 216.5/354.8 | 0 | 18/18 |
| 3 | True | 212.9/352.5 | 0 | 18/18 |
| 4 | False | 131.7/336.8 | 0 | 0/0 |

Enabled fit-work median2.83/2.93ms, P9575.47/72.82ms under real inference contention.
Speed compute-complete age median602.9/591.0ms, P95752.1/723.7ms. First client HTTP
snapshot age median2225.7/2331.1ms, P952756.1/2706.8ms. No busy skips. Each run ingests
2674 of2700 source frames: zero queue drops is not lossless RTMP coverage.

This short experiment avoids the heavy synchronous version's80/85 queue drops.
Median batch latency varies considerably even between off runs, so do not claim a
precise incremental latency or zero overhead. Only18 valid speeds per enabled run
versus40 for prior heavy experiment; shorter support/linear fit sacrifices coverage.
No paired radar accuracy test. Both settings and background scheduling changed;
this evaluates the combined lightweight proposal, not isolated solver causality.

Verdict: successful productive lightweight cloud experiment, useful for continued
live integration. User-visible subsecond delivery NOT achieved; progress publication,
next-batch drain and polling still add delay. Production defaults unchanged. Temporary
cloud instance released:true. Failed/old experiments retained separately.
