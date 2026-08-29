# Contributing

Start with `docs/HANDOFF.md` and obey `AGENTS.md`.

1. Create a focused branch.
2. Add or update a test with every behavioral change.
3. Keep tracking, landing classification and rendering ownership separate.
4. Run `pytest -m "not assets and not integration"` for ordinary changes.
5. Run asset tests and the full native-rate demo for algorithm/model changes.
6. Run `ruff check .`, `node --check web/app.js` and
   `python tools/release_check.py --mode handoff`.
7. Explain accuracy and speed changes with the same input and hardware.

Do not add uploaded videos, caches, generated user reports, model files or third-party
paper PDFs to a pull request.
