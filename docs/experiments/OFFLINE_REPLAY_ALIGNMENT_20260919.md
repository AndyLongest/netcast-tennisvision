# Offline replay repair, 2026-09-19

## Evidence and approach

The current uploaded 1920x1080 clip changes vertical framing between the manually
confirmed reference and the frame at 8 seconds. ORB/RANSAC measured corner movements
of approximately -31 to -33 pixels in y. A static projection therefore put service
boxes below the actual paint. Reference registration corrects this without replacing
manual corner identities. Insufficient matches retain the previous geometry; this
is a measured small-reframing solution, not arbitrary camera-cut recognition.

Primary open-source references reviewed:
- https://github.com/yastrebksv/TennisProject — temporal coordinate-difference features
  and learned bounce classification.
- https://github.com/philippdubach/tennis-vision — separate tracking/event passes,
  bounce deduplication and rally segmentation with additional event/score evidence.

No third-party implementation or new model weights were copied. The existing frozen
classifier remains unchanged. Changes add a continuous breakpoint fit, camera-view
normalization, hit/bounce conflict checks, confirmed-contact rally boundaries and a
replay timeline. Rule consistency alone no longer confirms a landing.

## Native-rate regression

Against the actual pre-change HEAD (3dbf6a1), the complete bundled demo retains all
1737 frames at 29.9139 fps and identical detector-backed ball positions. The prior
HEAD produces 28 bounces and 35 hits (the older frozen release fixture is 27/33).
The revised run produces 27 bounces and 35 hits. The removed contact at ~19.5 seconds
was the same racket impulse counted as a near-service-box landing one frame before
a high-confidence hit, without independent ground evidence. Other landing-time
refinements are within 0.017 seconds of this HEAD's prior estimates.

Reviewed native-frame strips at the human fixture windows around 5.5, 19.5, 25.95,
30, 40.1, 41.96 and 43.13 seconds. Occlusion and closely spaced half-volleys still
limit frame-exact ground truth; the review is not a precision/recall benchmark.
The new detector does not claim to solve every missed landing.

The new complete cold registration/report run took ~115 seconds locally. The same
HEAD without registration took ~42 seconds with warm inference caches. Registration
now caches by video/reference/corners to avoid repeating this extra native-frame pass.
The older bundled artifact/checksum fixture remains unchanged.

## Remaining limits

Contact-based point segmentation can still miss a boundary when contacts themselves
are missed or false. It is not a serve-pose or scoreboard recognizer. Registration
requires shared image features and retains the previous geometry when evidence is
insufficient. A new viewpoint or strong perspective change needs separate calibration.
