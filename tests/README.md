# Test map

Tests are organized by owning production behavior, not by development chronology. Run the
narrow file while iterating, then run the complete suite and handoff gate before transfer.

| Change area | Start with |
|---|---|
| Upload, resume, history, manual court flow | `test_server.py`, `test_chunked_transfer.py` |
| On-demand GPU lifecycle | `test_ppio_lifecycle.py`, `test_cloud_worker.py` |
| Live sessions and ECS relay | `test_live_experiment.py`, `test_result_relay.py`, `test_causal_events.py` |
| Candidate inference/batching | `test_racketvision_production.py`, `test_racketvision_batching.py` |
| Single-ball association/physics | `test_temporal_world_tracker.py`, `test_trail_rendering.py` |
| Hits, contacts, touchdowns, in/out | `test_contact_hypothesis.py`, `test_landing_*`, `test_line_call.py` |
| Players and play mode | `test_player_identity_experiment.py`, `test_play_mode.py` |
| Court geometry | `test_court_calibration.py`, `test_court_registration.py` |
| Browser/live-lab contracts | `test_live_lab_frontend.py` |
| Repository handoff and frozen assets | `test_package_layout.py`, `test_asset_manifest.py` |

## Fixture ownership

- `fixtures/production_manifest.json` freezes the shipped demo artifacts and counters.
- `fixtures/cleanup_e2e_baseline.json` records the full end-to-end preservation envelope.
- `fixtures/manual_landing_annotations_v1.json` is the human review set; algorithm changes
  must inspect every listed window.
- `fixtures/manual_landing_annotations_v1_results.json` is generated evaluation evidence,
  not a replacement for the human labels.

Synthetic tests should isolate one physical rule and use calibrated geometry when testing
world motion. Coverage counters are regression guards, not claims of independent accuracy.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe tools\release_check.py --mode handoff
```
