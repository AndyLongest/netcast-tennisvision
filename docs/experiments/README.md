# Experiment archive

These reports preserve measurements and rejected alternatives. They are decision evidence,
not supported runtime modes. Current behavior is defined by
[`../CURRENT_ARCHITECTURE.md`](../CURRENT_ARCHITECTURE.md).

| Report | Scope |
|---|---|
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
