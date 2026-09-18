# Documentation map

This page is the canonical index for repository documentation. New engineers and coding
agents should start at `AGENTS.md`, then use this page instead of reading every document.

## Read first

| Need | Document | Why it is authoritative |
|---|---|---|
| Understand the product and run it | [`../README.md`](../README.md) | Product scope, setup and supported entry points |
| Take over the repository | [`HANDOFF.md`](HANDOFF.md) | Frozen facts, ownership boundaries and definition of done |
| Change runtime behavior | [`CURRENT_ARCHITECTURE.md`](CURRENT_ARCHITECTURE.md) | Only supported data flow and algorithm contracts |
| Find the owning module | [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md) | Source tree, dependency direction and runtime state |
| Develop and verify safely | [`DEVELOPMENT.md`](DEVELOPMENT.md) | Change recipes, tests and regression procedure |
| Pick the next task | [`NEXT_STEPS.md`](NEXT_STEPS.md) | Prioritized backlog and explicitly rejected shortcuts |

## Contracts and operations

| Topic | Document |
|---|---|
| HTTP endpoints and `scene3d.json` | [`API_AND_SCHEMAS.md`](API_AND_SCHEMAS.md) |
| Frozen RacketVision production path | [`RACKETVISION_PRODUCTION.md`](RACKETVISION_PRODUCTION.md) |
| Frontend product rules | [`FRONTEND_PRODUCT.md`](FRONTEND_PRODUCT.md) |
| Runtime files and generated artifacts | [`OUTPUTS.md`](OUTPUTS.md) |
| On-demand PPIO deployment | [`PPIO_SERVERLESS.md`](PPIO_SERVERLESS.md) |
| Model installation and provenance | [`../assets/MODELS.md`](../assets/MODELS.md) |
| Publication/licensing blockers | [`PUBLICATION_CHECKLIST.md`](PUBLICATION_CHECKLIST.md) |
| Browser file ownership | [`../web/README.md`](../web/README.md) |
| Test and fixture selection | [`../tests/README.md`](../tests/README.md) |
| Notebook boundary | [`../notebooks/README.md`](../notebooks/README.md) |
| Developer command catalog | [`../tools/README.md`](../tools/README.md) |

## Evidence, not production alternatives

- [`experiments/BALL_TRACKING_PATH.md`](experiments/BALL_TRACKING_PATH.md) is the single
  ball-model evolution record: routes tried, trade-offs, the accepted route and its limits.
- The remaining files in [`experiments/`](experiments/) cover deployment, acceleration,
  identity and line-call measurements. They explain decisions but do not define runtime.
- [`references/`](references/) maps external papers and open-source projects to adopted or
  rejected ideas.
- [`history/`](history/) contains no executable or restorable ball-model generation.

## Documentation rule

Do not create a new top-level document for a one-off experiment. Put reproducible
measurements in `experiments/`, retired behavior in `history/`, and durable runtime rules
in the existing contract document that owns the behavior. When paths or contracts change,
update this index and run `python tools/release_check.py --mode handoff`.
