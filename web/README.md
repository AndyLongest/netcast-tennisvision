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

- The minimap bottom is always the video near court; the top is always the far court.
  Player identity changes marker colour only and never flips geometry.
- Show an in-court landing or out cross only after the backend event's decision frame.
- Perspective correction changes display only. Detection continues against original frames.
- A pending analysis shows an explicit analysis state, never the unannotated source as if
  it were a completed result.
- Refreshing the page reattaches through `/api/status`; it must not create a duplicate job.

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
