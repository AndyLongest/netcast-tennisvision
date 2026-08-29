# Publication checklist — currently blocked

This repository is being prepared for handoff but is **not yet cleared for public
publication**. Do not create a public GitHub repository until every blocking item below is
resolved and `python tools/release_check.py --mode public` passes.

## Legal and licensing blockers

- [ ] Obtain permission/license clarification for the original
  `vahehambardzumyan/Tennis_Vision` repository. It is public but currently contains no
  license file; public visibility alone does not grant redistribution rights.
- [ ] Choose the product's license with legal/owner approval.
- [ ] Confirm Ultralytics usage under AGPL-3.0 or document an Enterprise license.
- [x] Product owner selected `assets/demo/demo.mp4` and its derived annotated video as
  bundled repository assets on 2026-08-29.
- [ ] Confirm ownership/redistribution rights for `web/assets/netcast-mark.png`.
- [ ] Review the frozen bounce classifier's training-data provenance and distribution
  terms.
- [ ] Keep third-party paper PDFs out of Git; publish citations and source links only.

## Model installation checks

- [x] Pin the RacketVision upstream revision and source checksum.
- [x] Pin the Ultralytics official checkpoint URL and checksum.
- [x] Make the bounce classifier reproducible from its pinned source CSV.
- [ ] Run model installation in an empty clone and verify every final SHA-256.
- [ ] Confirm the fixed browser demo reports 1737 frames, 28 bounces and 32 hits.

## Repository blockers

- [x] Make `Tennis_Vision/` the self-contained runtime and repository boundary; production
  code no longer reads parent-workspace assets.
- [ ] Create a new owner-controlled remote. Keep the original repository as `upstream`.
- [ ] Review and stage all intended source, docs, fixtures and web assets.
- [ ] Ensure `git status` is clean and no ignored runtime/user video is staged.
- [ ] Add the approved `LICENSE` and complete `THIRD_PARTY_NOTICES.md`.
- [ ] Pass unit CI, asset tests, the full demo regression and public release check.
- [ ] Clone into a new directory and complete setup without using files from the original
  workstation.

## Evidence already collected

- Local unit/full suite passes on Python 3.12 and RTX 3050 Ti.
- No obvious credential patterns were found in the source-text scan on 2026-08-29.
- RacketVision declares MIT licensing.
- Ultralytics declares AGPL-3.0 for its open-source package/models unless an Enterprise
  license applies.
