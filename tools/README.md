# Developer tools

Utilities in this directory are optional diagnostics and are not imported by the product
pipeline. `render_confidence_video.py` renders a review video from existing tracking
metadata; it does not run detection or change algorithm outputs.

Handoff/release utilities:

- `install_assets.py`: installs only manifest-listed assets and rejects checksum mismatch;
- `verify_install.py`: checks Python, libraries, FFmpeg, models and frozen manifest;
- `install_assets.py`: downloads/converts/rebuilds the pinned runtime models and verifies
  every final SHA-256; the root `download_models.ps1` is its operator-friendly entrypoint;
- `check_e2e_baseline.py`: proves the current demo outputs remain byte-identical and the
  warm-cache runtime stays within the accepted 10% performance envelope;
- `quarantine_legacy_workspace.py`: inventories and reversibly isolates parent-workspace
  material without touching the canonical repository or the outer active Git metadata;
- `build_demo_assets.py`: regenerates `web/demo-scene.js` from the frozen JSON report;
- `release_check.py`: CI, handoff and public-publication repository gates.

- `diagnose_court_registration.py`: court-only failure diagnosis.
- `evaluate_racketvision.py`: detector-only blind evaluation using the production weight.
- `benchmark_production_batching.py`: verifies batched inference equivalence.
- `audit_landing_candidates.py`: inspects landing evidence without changing outputs.

Generated files belong under `outputs/evaluations/` or `outputs/benchmarks/`, never beside
the source code.
