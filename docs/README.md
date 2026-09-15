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

## Evidence, not production alternatives

- [`experiments/`](experiments/) contains dated benchmarks, rejected acceleration variants
  and model evaluations. These explain decisions but do not define current behavior.
- [`references/`](references/) maps external papers and open-source projects to adopted or
  rejected ideas.
- [`history/`](history/) preserves retired algorithm generations and old regression records.
  Never restore a historical path merely because it has a higher isolated counter.

## Documentation rule

Do not create a new top-level document for a one-off experiment. Put reproducible
measurements in `experiments/`, retired behavior in `history/`, and durable runtime rules
in the existing contract document that owns the behavior. When paths or contracts change,
update this index and run `python tools/release_check.py --mode handoff`.
