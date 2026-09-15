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

- Added bounded parallel cloud upload/download, verified fixed-camera profile handoff and
  preflighted NVENC encoding with automatic x264 rollback; analysis inputs are unchanged.
- Reduced cloud review-video transfer size with a same-resolution CRF22 profile validated
  at SSIM 0.9924; local x264 defaults and structured analysis outputs remain unchanged.
- Adjusted the PPIO ephemeral root filesystem to 80 GB, within the provider's current
  84 GB ceiling.
- Added one canonical documentation index, a close-to-code package ownership map and a
  complete developer-tool catalog; dated benchmarks now live under `docs/experiments/`.
- The handoff gate now rejects broken local documentation links and missing Agent entry
  points before a repository can be transferred.
- Migrated all production Python code into the conventional
  `src/netcast_tennisvision/` package, grouped by API, pipeline, vision, tracking and
  event ownership; the repository root no longer contains Python modules.
- Centralized repository path resolution and switched supported execution to module
  entry points without changing the frozen algorithm.
- Revalidated the complete native-rate demo after the structure migration: all three
  generated artifacts are byte-identical, tracking remains 1220/1123/97, events remain
  28 bounces and 32 hits, and the warm-cache run completed in 114.118s.
- Frozen the RacketVision pure-inference production path.
- Separated tracking, contact, landing, tennis-order and rendering ownership.
- Added fixed-camera calibration reuse and equivalent four-frame detector batching.
- Added resumable single-job browser analysis.
- Promoted the current validated demo to 29 bounces and 35 racket hits after the
  physically supported rejected-fragment recall pass; the earlier 28/32 counters above
  remain historical migration evidence.
- Added checksummed asset management, reproducible setup scaffolding, fixtures, CI and
  engineering handoff documentation.

The first public version number will be assigned only after the publication checklist is
complete.
