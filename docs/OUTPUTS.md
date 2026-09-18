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
current-rally landings and terminal net-hit crosses from `scene3d.json`. The same report
also carries the normalized image-space geometry needed by the browser overlay:

- `court_image_corners`: four court corners in world-quad order, normalized to the source frame;
- `frames[].bp`: the exact normalized ball observation used for the purple marker;
- `frames[].dp`: the normalized display trajectory point used for the short purple trail.

Landing flashes and minimap points become visible only at each event's `decision_frame`,
never at its retrospectively estimated touchdown frame. The minimap is a fixed-size
bottom-right UI layer matching the original annotated renderer. An `annotated_clip.mp4`
or `corrected_clip.mp4` left from an older task is not part of the current result and is
never selected by the UI. Those files are generated only by the explicit
`annotated-video` rollback mode.
