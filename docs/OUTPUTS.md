# Runtime data and generated outputs

The project keeps generated files in three deliberately separate locations:

| Location | Purpose | Safe to overwrite? |
|---|---|---|
| `data/clip.mp4` | most recently uploaded source | yes, by the single-job server |
| `data/outputs/` | most recent upload report | yes, by `pipeline_runner.py` |
| `assets/demo/` | fixed product demo and its frozen report | only when promoting a validated baseline |
| `data/regression/checks/` | user-reviewed timestamp clips and QA evidence | no |
| `data/cache/` | reproducible inference caches | yes |

The web demo reads only `assets/demo/`. Upload analysis reads only
`data/outputs/`. This separation prevents an upload from changing the bundled example.

Large exploratory videos are intentionally not kept. Algorithm history belongs in small
Markdown/JSON records under `docs/history/`; a result worth preserving must be promoted
to a named regression fixture rather than left in a generic output directory.
