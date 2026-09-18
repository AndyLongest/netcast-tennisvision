# Notebook boundary

`tennis_detection.ipynb` is the maintained two-pass production orchestrator and visual
research surface. It is not the owner of reusable algorithms.

- Pass A decodes native-rate frames and caches court, person and frozen ball candidates.
- Package modules associate the physical ball, classify contacts and locate touchdowns.
- Pass B writes `scene3d.json`, the interactive viewer and the optional review video.
- `pipeline/runner.py` executes code cells and bridges progress/manual court confirmation
  to the product API.

New reusable logic belongs under `src/netcast_tennisvision/` with a focused unit test.
Notebook cells may configure, orchestrate and visualize that logic. Keep cell outputs small;
large diagnostic images and videos belong under ignored `outputs/`.

After changing orchestration, run the complete bundled demo at native frame rate and compare
`tests/fixtures/production_manifest.json` plus every human review window in
`tests/fixtures/manual_landing_annotations_v1.json`.
