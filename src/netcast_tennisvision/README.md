# Package ownership map

Production Python code lives only in this package. Before editing a module, identify the
owner below and keep data flowing downstream.

```text
api -> pipeline -> vision -> tracking -> events -> report/rendering
  \                                      /
   +-> cloud -> streaming/live ---------+
```

The arrows show orchestration order. They do not allow downstream event code to rewrite
upstream observations.

| Package/module | Owns | Must not own |
|---|---|---|
| `api/` | Browser transport, resumable jobs, local/cloud relay | Detection thresholds |
| `cloud/` | Temporary PPIO lifecycle and worker transport | Product algorithms |
| `streaming/` | Causal live sessions, result relay and live-only state | Offline model tuning |
| `pipeline/` | Notebook execution, progress, encoding | Reusable CV logic |
| `vision/racketvision.py` | Frozen ball candidates | Track birth or landings |
| `vision/court_*` | Fixed-camera court geometry | Per-frame ball decisions |
| `vision/player_identity.py` | Stable player A/B labels | Ball candidate acceptance |
| `tracking/` | One physical ball, geometry, smoothing, short gaps | Landing-zone display |
| `events/` | Hit/bounce competition, touchdown and tennis ordering | Creating/moving ball observations |
| `paths.py` | Repository-local paths | Runtime policy |

Public entry points are `python -m netcast_tennisvision` for the product service and
`python -m netcast_tennisvision.pipeline.runner` for direct analysis. Developer-only
commands belong in `tools/` and must never be imported by this package.

The detailed runtime contract is [`../../docs/CURRENT_ARCHITECTURE.md`](../../docs/CURRENT_ARCHITECTURE.md).

## Maintained modules

This is a navigation map, not a second architecture specification. Behavioral rules stay
in `docs/CURRENT_ARCHITECTURE.md`.

| Module | Stable surface | Nearest regression |
|---|---|---|
| `api/server.py` | `main`, HTTP routes, `Handler` | `tests/test_server.py`, `test_chunked_transfer.py` |
| `cloud/ppio_lifecycle.py` | `PPIOJobManager`, `PPIOLiveJobManager` | `tests/test_ppio_lifecycle.py` |
| `pipeline/runner.py` | `main`, progress and court-confirmation bridge | `tests/test_pipeline_status.py` |
| `streaming/live_experiment.py` | `LiveExperimentManager` | `tests/test_live_experiment.py` |
| `streaming/result_relay.py` | relay store/publisher/fetch/delete | `tests/test_result_relay.py` |
| `vision/racketvision.py` | frozen candidate inference functions | `tests/test_racketvision_*.py` |
| `vision/player_identity.py` | identity and landing attribution | `tests/test_player_identity_experiment.py` |
| `tracking/world_tracker.py` | `track_ball_persistent` | `tests/test_temporal_world_tracker.py` |
| `events/landing_event_detector.py` | contact impulses and one-landing ordering | `tests/test_landing_event_detector.py` |
| `events/landing_detector.py` | sub-frame touchdown fit | `tests/test_landing_detector.py` |

The three larger lifecycle modules (`api/server.py`, `cloud/ppio_lifecycle.py`, and
`streaming/live_experiment.py`) deliberately keep stateful orchestration together. Extract
only stateless, independently testable behavior; do not create thin files that merely hide
control flow.
