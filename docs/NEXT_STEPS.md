# Prioritized next steps

This file is the product backlog for the next engineer. Do not reopen historical tuning
work unless a new regression demonstrates that the current frozen path is worse.

## P0 — complete before public publication

1. Obtain written permission or a license clarification for the original
   `vahehambardzumyan/Tennis_Vision` notebook, which currently has no repository license.
2. Decide whether this product will comply with Ultralytics AGPL-3.0 or use an Enterprise
   license. Do not guess.
3. Confirm ownership/redistribution rights for the Netcast brand assets.
4. Review the pinned model/data-source license terms even though automatic installation is
   technically reproducible.
5. Run a clean clone installation using only published files and URLs.

## P1 — accuracy and generalization

1. Build a small, legally shareable evaluation set spanning at least three fixed baseline
   cameras, near/far lighting differences, 30fps and 60fps. Labels should cover ball
   visibility, hits, touchdowns, in/out and rally boundaries.
2. Report precision/recall against labels. Existing coverage counters are regression
   guards, not accuracy claims.
3. Prioritize far-court touchdown recall without changing detector-backed ball anchors.
4. Add confidence calibration so only genuinely uncertain court or landing results ask for
   user correction.

## P2 — performance

1. Preserve the accepted fixed-court, four-frame batching path.
2. Profile detector, player segmentation, decode/render and disk I/O separately on the
   target GPU before optimizing.
3. Add streaming/near-real-time work only as a separate mode; do not silently replace the
   validated offline smoother.
4. Record latency, throughput and accuracy together. A faster variant is rejected when it
   loses detector anchors or reviewed touchdowns.

## P3 — maintainability

1. Move production orchestration out of notebook cell execution into a normal Python
   module while keeping the notebook as a research/demo client.
2. Version the `scene3d.json` schema and add migration tests before changing fields.
3. Replace the single global job with an explicit job store only when multi-user or remote
   deployment becomes a real product requirement.

## Explicitly rejected shortcuts

- training BallNet on each uploaded video;
- dropping frames to claim speed;
- lowering detector thresholds without temporal association evidence;
- forcing one landing between every pair of hits;
- letting audio create a landing coordinate;
- smoothing real detector coordinates into visually pleasing but false locations;
- tuning thresholds against only the fixed demo.
