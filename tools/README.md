# Developer tools

Nothing in this directory is imported by production code. Scripts are kept flat so every
command can be run from the repository root without path or import surprises; this index
provides the logical grouping. Generated files belong under `outputs/evaluations/` or
`outputs/benchmarks/`, never beside source code.

## Installation and release gates

| Script | Purpose |
|---|---|
| `install_assets.py` | Download/rebuild only manifest-listed models and verify SHA-256 |
| `verify_install.py` | Audit Python, libraries, FFmpeg, models and the frozen manifest |
| `release_check.py` | Run CI, handoff or public-publication repository gates |
| `check_e2e_baseline.py` | Compare frozen demo outputs and the accepted runtime envelope |
| `build_demo_assets.py` | Regenerate `web/demo-scene.js` from the frozen demo report |
| `quarantine_legacy_workspace.py` | Reversibly isolate legacy parent-workspace material |

The operator-facing wrappers are `setup.ps1`, `download_models.ps1` and `run_ui.ps1` in
the repository root.

## Diagnostics and review output

| Script | Purpose |
|---|---|
| `diagnose_court_registration.py` | Run court calibration without loading ball/person models |
| `audit_landing_candidates.py` | Inspect landing evidence without changing outputs |
| `evaluate_racketvision.py` | Evaluate detector candidates with the frozen public weight |
| `render_confidence_video.py` | Render observation/confidence overlays from existing metadata |
| `experiment_player_identity.py` | Produce the isolated player A/B identity review artifact |
| `jupyter_terminal_exec.py` | Compatibility helper for controlled notebook execution |

## Benchmarks

| Script | Isolated variable |
|---|---|
| `benchmark_background_sampling.py` | Parallel deterministic background construction |
| `benchmark_detector_parallelism.py` | Concurrent ball/person model execution |
| `benchmark_person_prefetch.py` | Person-result preparation overlap |
| `benchmark_pipeline_variant.py` | End-to-end feature-flag combinations |
| `benchmark_production_batching.py` | Production ball-batch equivalence |
| `benchmark_racketvision_batch.py` | Detector-only batch sizes |
| `benchmark_live_cloud.py` | True RTMP/ZLMediaKit live chain on one auto-released PPIO GPU |

Benchmark scripts may measure rejected variants. A script's presence does not mean its
variant is enabled. Production decisions are recorded in
`docs/CURRENT_ARCHITECTURE.md`; detailed evidence belongs in `docs/experiments/`.
