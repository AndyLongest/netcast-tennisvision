# Project structure

Netcast TennisVision uses the conventional Python `src` layout. The repository root is
reserved for operator entry points, packaging metadata, documentation, and top-level
assets; production Python code belongs under one importable package.

```text
Tennis_Vision/
├── src/netcast_tennisvision/
│   ├── README.md     # close-to-code responsibility and dependency map
│   ├── api/          # HTTP transport, upload lifecycle, resumable job state
│   ├── pipeline/     # production orchestration and notebook execution
│   ├── vision/       # frozen inference, court calibration, and player identity
│   ├── tracking/     # ball lifecycle, geometry, physics and smoothing
│   ├── events/       # contact classification, touchdown and tennis ordering
│   └── paths.py      # the single repository-root resolver
├── web/              # browser product
├── notebooks/        # maintained research/orchestration notebook
├── tests/            # regression and contract tests
├── tools/            # developer diagnostics; never imported by production code
├── assets/           # versioned demo and model manifest
├── docs/             # canonical contracts and documentation index
│   ├── experiments/ # dated benchmarks and rejected alternatives
│   ├── references/  # external evidence and implementation mapping
│   └── history/     # retired behavior; never a runtime fallback
├── models/           # installed frozen weights; ignored by Git
├── data/             # active job and latest report; ignored by Git
└── outputs/          # diagnostics and benchmarks; ignored by Git
```

## Dependency direction

```text
api -> pipeline -> vision -> tracking -> events -> rendered outputs
```

The diagram describes orchestration order, not permission to mutate upstream results.
Tracking owns ball positions. Event modules consume the trajectory and may not create,
move, or delete observations. `tools/` may import the product package; production code
must never import `tools/`, `tests/`, or notebook-only helpers.

## Naming and import rules

- Use package imports such as
  `from netcast_tennisvision.tracking.world_tracker import track_ball_persistent`.
- Use relative imports only between modules inside the same subpackage.
- Do not add Python modules to the repository root.
- Put shared repository paths in `netcast_tennisvision.paths`; do not derive the root
  independently in multiple production modules.
- Keep reusable algorithms out of notebook cells. The notebook may orchestrate package
  APIs, visualize intermediate results, and preserve research provenance.

## Supported entry points

```powershell
.\run_ui.ps1
.\.venv\Scripts\python.exe -m netcast_tennisvision
.\.venv\Scripts\python.exe -m netcast_tennisvision.pipeline.runner
```

`setup.ps1` installs the package in editable mode, so module entry points and developer
edits use the same source tree.

## Runtime versus versioned files

`src/`, `web/`, `notebooks/`, `tests/`, `tools/`, `docs/` and `assets/` are versioned.
`models/`, `data/`, `outputs/`, `.venv/` and `.cache/` are local runtime state and are
ignored by Git. Do not solve an import or deployment problem by reaching into those local
directories from a different checkout.

The canonical documentation index is `docs/README.md`. Dated measurements go to
`docs/experiments/`; historical algorithm generations go to `docs/history/`. Keep durable
production rules in `docs/CURRENT_ARCHITECTURE.md` rather than duplicating them in a new file.
