# Experiment archive

These reports preserve measurements and rejected alternatives. They are decision evidence,
not supported runtime modes. Current behavior is defined by
[`../CURRENT_ARCHITECTURE.md`](../CURRENT_ARCHITECTURE.md).

| Report | Scope |
|---|---|
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
