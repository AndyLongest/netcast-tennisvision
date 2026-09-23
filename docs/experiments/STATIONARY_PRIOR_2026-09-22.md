# Stationary candidate motion prior — 2026-09-22

Opt-in experiment: `track_ball_persistent(stationary_prior=True)`. Default remains
False; neither notebook product defaults nor cloud workers enable it. No new model.
The bounded causal memory lives in tracking/stationary.py; raw candidates are retained.

A fixed image anchor needs >=0.65 s dwell and >=0.5*fps observations within the last
second. Radius is max(2,2.5*spatial) pixels; anchors expire after one second without
observations, reset at non-court frames and are capped at 128. A stationary candidate
is excluded from association except near player/racket reach or during a continuously
observed moving trajectory crossing. Birth additionally requires displacement/path
length >=0.65. This is not a whole-video density mask or a permanent forbidden region.
Camera-moving videos and broad near-player exemptions remain limitations.

## Controlled flat2 experiment

User requested only the first 90 seconds: 1920x1080, 30fps, 2700 frames, original audio,
no display trail. Initial inference on a newly trimmed clip changed the model's median
background and produced zero stationary exclusions. For an apples-to-apples tracking
comparison, both final runs instead consumed the same first 2700 candidate rows cached
from the previously analysed full flat_demo2. Only association prior on/off differs.
Person/court/audio processing is shared for the 90-second clip.

RTX 3050 Ti Laptop; final outputs in outputs/stationary_flat2/{baseline,result}.
Prior excluded 529 candidate observations. Detector-backed accepted positions changed
1328 -> 1276; positioned frames 1851 -> 1844. Both outputs contain 14 bounces but their
times differ. These numbers do NOT establish improved accuracy: rejection can also
remove true balls, and human review is required before any default promotion.
The prior's final annotated render took 95 s. MP4 with audio decoded to all 2700 frames.
Review: outputs/stationary_flat2/result/review.html and render_check.jpg.

## Protection and tests

The full native 1737-frame high-view demo with prior enabled exported a scene exactly
identical to outputs/contact_without_frame/high/scene3d.json (27 bounces, same frame
states and decision times). Therefore all previously reviewed annotation windows are
preserved on this sample, not a claim for every elevated camera.

Synthetic tests cover stationary dwell/expiry, moving traversal, jitter versus launch,
raw candidate preservation, a moving track crossing a known stationary location,
brief pauses and player-held ball exemption. Full pytest, Ruff and handoff gate run.
Cloud and ordinary production defaults are unchanged pending user video review.
