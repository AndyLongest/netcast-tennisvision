# Rally segmentation trial, 2026-09-21

No point/game hierarchy is introduced. Production rally boundaries now include a strictly
greater-than-10-second contact gap followed by a hit, alongside existing terminal outcomes.
Scoreboard changes and two-second display expiry remain disabled.

## Fixed-input comparison

Reused `outputs/far_roi_demo3/baseline/analysis.pkl` and its scene net events to isolate
segmentation from inference changes. Terminal-only: 12 intervals. Ten-second rule: 14.
New boundaries: 122.406 and 277.705 seconds. Visual review at +/-2 seconds shows preparation
(ball handling/bouncing) at both locations, so these are not verified serve-contact boundaries.
More segments is not evidence of better accuracy. Existing false contact classifications
still affect the rule; no hand-labelled full-video boundary ground truth exists here.

## Serve prototype (not enabled)

`events/serve.py` proposes baseline/stable-player/toss/overhead/outgoing-flight sequences.
On the same clip it proposed frames 401, 3189, 6318, 6419. Three-frame visual review
revealed ordinary rally/overhead-return activity; this is insufficiently specific to serves.
Do not enable these proposals as production boundaries. No pose/racket keypoints are
available in the current detector output. A validated pose/action detector and annotated
serve/aborted-toss/overhead-return examples are still required before claiming independent
serve recognition. No cloud deployment was performed.

Artifacts: `outputs/rally_comparison.json`, `outputs/rally_gap_review.jpg`,
`outputs/serve_review.jpg`. Review images were sampled by decoder seek and are approximate;
metrics and boundaries use original cached native-rate frame indices.
