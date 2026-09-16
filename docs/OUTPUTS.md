# Runtime data and generated outputs

The project keeps generated files in three deliberately separate locations:

| Location | Purpose | Safe to overwrite? |
|---|---|---|
| `data/clip.mp4` | most recently uploaded source | yes, by the single-job server |
| `data/outputs/` | most recent upload report | yes, by `pipeline/runner.py` |
| `assets/demo/` | fixed product demo and its frozen report | only when promoting a validated baseline |
| `data/regression/checks/` | user-reviewed timestamp clips and QA evidence | no |
| `data/cache/` | reproducible inference caches | yes |

The web demo reads only `assets/demo/`. Upload analysis reads only
`data/outputs/`. This separation prevents an upload from changing the bundled example.

Large exploratory videos are intentionally not kept. Algorithm history belongs in small
Markdown/JSON records under `docs/history/`; a result worth preserving must be promoted
to a named regression fixture rather than left in a generic output directory.

In normal `event-overlay` output mode the browser always reads untouched `data/clip.mp4`.
It optionally applies the saved perspective homography with WebGL, then draws confirmed
current-rally landings and terminal net-hit crosses from `scene3d.json`. An `annotated_clip.mp4` or
`corrected_clip.mp4` left from an older task is not part of the current result and is never
selected by the UI. Those files are generated only by the explicit `annotated-video`
rollback mode.
