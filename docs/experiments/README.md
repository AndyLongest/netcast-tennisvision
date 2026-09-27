# Experiment archive

These reports preserve measurements and rejected alternatives. They are decision evidence,
not supported runtime modes. Current behavior is defined by
[`../CURRENT_ARCHITECTURE.md`](../CURRENT_ARCHITECTURE.md).

| Report | Scope |
|---|---|
| [`CONTACT_POSE_2026-09-22.md`](CONTACT_POSE_2026-09-22.md) | Optional pose/ball contact correction, two flat2 cases, and local source-paced compute limits |
| [`STATIONARY_PRIOR_2026-09-22.md`](STATIONARY_PRIOR_2026-09-22.md) | Opt-in causal stationary-candidate rejection; controlled first-90-second flat2 render |
| [`CONTACT_WITHOUT_FRAME_2026-09-22.md`](CONTACT_WITHOUT_FRAME_2026-09-22.md) | Shared low/high contact fitting without a detected centre; native regressions and rollout limits |
| [`LOW_VIEW_2026-09-21.md`](LOW_VIEW_2026-09-21.md) | Flat demo far-crop supplementation and unpositioned occluded bounce experiment |
| [`POSE_CLOUD_LIVE_2026-09-21.md`](POSE_CLOUD_LIVE_2026-09-21.md) | L40S RTMP pose ABBA: no sustained backlog in short runs, small delay increase, incomplete serve recall |
| [`POSE_PSEUDOLIVE_2026-09-21.md`](POSE_PSEUDOLIVE_2026-09-21.md) | Pose on/off source-paced local latency; saturated baseline prevents live-readiness conclusions |
| [`BALL_TRACKING_PATH.md`](BALL_TRACKING_PATH.md) | Single consolidated ball-tracking evolution, accepted contact-aware route and limits |
| [`UNCERTAINTY_AWARE_LINE_CALL_2026-09-16.md`](UNCERTAINTY_AWARE_LINE_CALL_2026-09-16.md) | ITF edge convention, homography uncertainty and three-state line calls |
| [`REALTIME_LANDING_SIMULATION_2026-09-16.md`](REALTIME_LANDING_SIMULATION_2026-09-16.md) | Causal landing-display delay and native ball-throughput experiment |
| [`EVENT_PRESERVING_BALL_SAMPLING_2026-09-16.md`](EVENT_PRESERVING_BALL_SAMPLING_2026-09-16.md) | Rejected adaptive-frame and grouped-output BallTrack trials |
| [`CLOUD_ACCELERATION_2026-09-16.md`](CLOUD_ACCELERATION_2026-09-16.md) | PPIO transfer, camera reuse and encoding A/B |
| [`ACCELERATION_DEEP_RESEARCH.md`](ACCELERATION_DEEP_RESEARCH.md) | Survey of end-to-end acceleration options |
| [`OPEN_SOURCE_RUNTIME_BENCHMARK.md`](OPEN_SOURCE_RUNTIME_BENCHMARK.md) | Comparable open-source runtime evidence |
| [`PERFORMANCE_EXPERIMENT_2026-09-15.md`](PERFORMANCE_EXPERIMENT_2026-09-15.md) | Native-rate local acceleration A/B results |
| [`PPIO_BENCHMARK_2026-09-15.md`](PPIO_BENCHMARK_2026-09-15.md) | Cloud GPU benchmark and transfer observations |
| [`PLAYER_IDENTITY_EXPERIMENT.md`](PLAYER_IDENTITY_EXPERIMENT.md) | Player A/B identity evaluation |
| [`QUASI_REALTIME_EXPERIMENT.md`](QUASI_REALTIME_EXPERIMENT.md) | Rejected and accepted near-real-time variants |

Every new benchmark should record the exact input, hardware, configuration, output
equivalence checks and rollback decision. A speed result without accuracy/equivalence
evidence is incomplete.

- [Monocular speed analysis](SPEED_ANALYSIS_2026-09-23.md) — assumptions, validation and limits.

- [Open-source speed selection](SPEED_OPEN_SOURCE_2026-09-23.md) — TT3D-inspired independent tennis drag model.

- [Speed live latency](SPEED_LIVE_LATENCY_2026-09-23.md) — overloaded local A/B and productive component timing.

- [Cloud speed latency](SPEED_CLOUD_LATENCY_2026-09-23.md) — L40S live ABBA and productive-fit timing.

- [Cloud speed end-to-end verification](SPEED_E2E_2026-09-23.md) — homography sign fix, real speed delivery and ABBA.

- [Lightweight cloud speed experiment](SPEED_LINEAR_CLOUD_2026-09-23.md) — short linear fit and bounded background worker.
