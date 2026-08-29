# Frozen product demo

This directory is the only source used by the web application's “真实比赛样例” path.

Tracked product-demo assets:

- `demo.mp4` — bundled 1280×720 source used by “真实比赛样例”;
- `annotated_clip.mp4` — frozen production result;
- `scene3d.json` — 1737 frames, 28 confirmed bounces and 32 racket hits;
- `rally3d.html` — self-contained 3D report.

All four files travel with the repository. The web sample never reads a video or report
from the parent workspace. Their expected byte sizes and SHA-256 values are frozen in
`assets/manifest.json`.

After promoting a new validated report, run:

```powershell
python tools/build_demo_assets.py
python tools/release_check.py
```

This regenerates `web/demo-scene.js` and checks the report against the frozen production
manifest in `tests/fixtures/production_manifest.json`.
