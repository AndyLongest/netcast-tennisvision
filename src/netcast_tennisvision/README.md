# Package ownership map

Production Python code lives only in this package. Before editing a module, identify the
owner below and keep data flowing downstream.

```text
api -> pipeline -> vision -> tracking -> events -> report/rendering
```

The arrows show orchestration order. They do not allow downstream event code to rewrite
upstream observations.

| Package/module | Owns | Must not own |
|---|---|---|
| `api/` | Browser transport, resumable jobs, local/cloud relay | Detection thresholds |
| `cloud/` | Temporary PPIO lifecycle and worker transport | Product algorithms |
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
