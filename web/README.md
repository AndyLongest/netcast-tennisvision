# Browser product map

The browser is a consumer of backend truth. It may animate and present structured results,
but it must not infer new hits, landings, court sides or player ownership.

| File | Responsibility |
|---|---|
| `index.html` | Main upload, progress, report and analysis-history shell |
| `styles.css` | Main product design system and responsive layout |
| `app.js` | Upload/resume lifecycle, court confirmation, report playback and history |
| `live-lab.html` | Internal pseudo-live comparison shell |
| `live-lab.css` | Live-lab-only layout and route visualization |
| `live-lab.js` | Live session transport, causal event display and cleanup |
| `demo-scene.js` | Generated bundled-demo data; rebuild with `tools/build_demo_assets.py` |

## Coordinate and timing contracts

- Court confirmation and manual display-correction marking reserve 15% of the source
  size on each side as a clickable gutter. Pointer conversion removes that offset;
  saved points remain in original image coordinates (normalized for live), including
  negative and beyond-frame values. CSS letterboxing is not part of the gutter.

- Ordinary event-overlay replay shows the current ball marker without a trailing path.
  Tracking history remains in structured data for contact analysis and 3D review.

- The minimap bottom is always the video near court; the top is always the far court.
  Player identity changes marker colour only and never flips geometry.
- Replay minimap geometry uses one pixels-per-metre scale on both axes. Markers and
  out crosses scale with map width using the baked renderer's proportions, without
  oversized glow; service lines end at the singles sidelines.
- Show an in-court landing or out cross only after the backend event's decision frame.
- Perspective correction changes display only. Detection continues against original frames.
- A pending analysis shows an explicit analysis state, never the unannotated source as if
  it were a completed result.
- Refreshing the page reattaches through `/api/status`; it must not create a duplicate job.
- Court overlays select `court_keyframes` at the current source PTS, including after
  seeking backwards. HTML/JS/CSS responses require cache revalidation so a newly
  generated report cannot silently retain an older first-frame-only renderer.

Backend endpoints and payload fields are defined in `../docs/API_AND_SCHEMAS.md`. Product
copy and visual rules are defined in `../docs/FRONTEND_PRODUCT.md`.

## Verification

```powershell
node --check web\app.js
node --check web\live-lab.js
.\.venv\Scripts\python.exe -m pytest tests\test_live_lab_frontend.py tests\test_server.py
```

Do not hand-edit `demo-scene.js`. Do not add a second frontend framework or build pipeline
for a local fix; the supported product is deliberately static and served by the Python API.

The upload card offers “使用本机显卡分析”, off by default. It submits `X-Execution-Target: local-gpu` when enabled, shows local progress messaging, and keeps force-stop available. This selects the API host GPU for ordinary uploaded-video analysis; live-lab routing is unchanged.

`viewpoint.js` presents the shared Python viewpoint classifier in both calibration
screens and the initial-camera report label. It owns no thresholds and does not change
coordinates or select an analysis model. Stale preview responses are discarded.
