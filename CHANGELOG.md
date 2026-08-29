# Changelog

## 2026-08-29 — self-contained repository boundary

- Bundled the fixed source and annotated demo with the repository.
- Added reproducible, checksummed installation for all three runtime models.
- Removed production dependencies on parent-workspace files and paths.
- Quarantined legacy sibling experiments before cleanup and verified the complete demo
  before and after: byte-identical reports, identical 1220/1123/97 tracking counters,
  identical 28 bounces and 32 hits, and 118.633s versus 118.676s runtime.
- Sent the verified legacy quarantine to the Windows Recycle Bin rather than permanently
  erasing it.

## Unreleased — handoff preparation

- Frozen the RacketVision pure-inference production path.
- Separated tracking, contact, landing, tennis-order and rendering ownership.
- Added fixed-camera calibration reuse and equivalent four-frame detector batching.
- Added resumable single-job browser analysis.
- Promoted the validated demo to 28 bounces and 32 racket hits.
- Added checksummed asset management, reproducible setup scaffolding, fixtures, CI and
  engineering handoff documentation.

The first public version number will be assigned only after the publication checklist is
complete.
