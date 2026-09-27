# Contact without a captured touchdown — 2026-09-22

This is validation evidence for the shared production contact changes in
`events/contact_view.py`, `events/landing_detector.py`, the maintained notebook,
and `streaming/live_experiment.py`. No new model or ball-observation interpolation
was added. Both viewpoint classes use the same logic.

## Change

The near/general impulse passes and rally recovery no longer require ball_seen at
the candidate. Event-only geometry is fitted from same-track real samples on both
sides. Missing-centre fitting cannot fall back to a predicted pixel or to two arbitrary
intersecting branches: an upward continuous impulse must beat smooth flight. Existing
court, racket, player and sequence checks still apply. Stream proposals use the same
helper and fractional touchdown estimate. Fit availability waits for every used sample.

## Validation

Windows / RTX 3050 Ti Laptop; frozen weights, native frame rates; cached detector and
person inference retained for deterministic event regression. No remote GPU used.

- Full pytest and Ruff passed; handoff gate passed (existing parent Git warning only).
- Synthetic 0.3 and 1.0 vertical scales: contact at frame 10.4 recovered with no gap,
  one missing detection and three missing detections. Time error <0.15 frame,
  coordinate error <0.3 pixel. Input dictionaries unchanged.
- Rejected mixed tracks, predicted-only future evidence and smooth flight with a
  fabricated predicted centre. Existing hit/landing suites still passed.
- Full assets/demo/demo.mp4, 1737 native frames: 1380 positioned / 1276 detector-backed,
  27 bounces. These are the current local pre-change baseline counters, not the older
  shipped manifest's tracker counters. Every bounce field except decision_frame and
  decision_t was unchanged. All compact frame data was identical.
- The 9 manual annotation timestamp windows were reviewed through source-frame contact
  sheets and event tables. Existing event times and positions were preserved. This is
  regression preservation, not a new independent accuracy measurement.
- Full flat_demo2, 9000 frames / 300 seconds: 7076 positioned / 5304 detector-backed,
  68 bounces with the ordinary full-frame model. All compact frame data unchanged.
  Compare with the original full-frame baseline, not the separate ROI video (71 bounces).
- Additional diagnostic withholding of three observations around existing contacts
  produced supported gap fits in both recordings. A complete high-view run with frames
  125–127 withheld still produced the nearby landing through the existing two-arc path.
  These controlled omissions are diagnostic evidence, not real-world recall statistics.

Local evidence: outputs/contact_without_frame/{high,low,high_withheld}/scene3d.json,
manual_windows.json, manual_windows.jpg, withheld_contact_checks.json and run logs.

## Limits and rollout

Insufficient two-sided evidence can still leave a landing undetected. This change removes
an exact-frame prerequisite; it does not guarantee recognition through arbitrary occlusion.
Typical fitted display decisions on the high sample move 3–6 native frames later to
account honestly for future support. No fresh live latency benchmark was run.

The source covers ordinary offline analysis and live workers. The previously generated
MP4/report is unchanged. No cloud image was built or deployed in this change; cloud
instances using the pinned old image require a release before these changes take effect.
