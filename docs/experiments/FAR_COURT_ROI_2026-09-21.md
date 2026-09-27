# Far-court crop experiment (2026-09-21)

User-requested offline experiment on the archived `demo3_clip.mp4`.
Production detection, model weights, API and cloud image remain unchanged.

`tools/experiment_far_court_roi.py` executes the maintained notebook in isolated
`outputs/far_roi_demo3/{baseline,roi}` directories. Both variants use the same
archived court corners and native source frames. Person observations are reused from
the same isolated Pass-A cache. No production upload, status or historical report is
replaced. The experiment removes only the rendered trailing line from both videos.

The additional detector reads a losslessly encoded, fixed 16:9 crop enclosing the
far half of the court with a 16% horizontal margin. It uses the unchanged public
0.5 threshold, four-frame temporal input and background median. Coordinates are
translated back to original image pixels. The crop's primary candidate replaces
the full-frame candidate only when both are in the far court, or when the full-frame
candidate is absent and the crop candidate is in the far court. The existing
association and contact modules must still accept the observation and event.

This first experiment runs two detector passes to isolate accuracy effects. It is
not the proposed causal switching scheduler and does not establish live throughput.
Candidate coverage and event counts are not accuracy metrics. The model was not
trained on this crop distribution; review false positives as well as missing balls.

Generated timing, scene JSON, analysis metadata and annotated videos remain under
`outputs/far_roi_demo3/`. Review the baseline and experimental video together before
any production adoption. Source frame-count metadata may differ from successfully
decoded frames; crop and baseline output lengths must match exactly.


## Observed results

Both decoders produced 8,986 frames (container counts were unreliable). Crop pixels
matched source pixels exactly at frames 0, 30, 300, 3000 and 8000. Actual crop:
`x=502, y=0, width=832, height=468`, approximately 2.31x greater detector image scale.
Additional crop detection took 268.63 seconds on the local RTX 3050 Ti, including its
setup; detector-loop throughput was 34.3 FPS versus 33.7 FPS for the full frame.
These are isolated detector timings, not live pipeline throughput.

| Counter | Baseline | Crop experiment |
| --- | ---: | ---: |
| Detector-backed tracked frames | 4613 | 4735 |
| Synthetic tracked frames | 1200 | 1087 |
| All tracked frames | 5813 | 5822 |
| Near-side bounces | 41 | 40 |
| Far-side bounces | 27 | 29 |
| Total bounces | 68 | 69 |
| Racket hits | 134 | 134 |

Counts are not labelled accuracy. At about 129.6 seconds, visual inspection still
shows a serve-toss action being classified as a bounce; its court-side label changes
between variants. This establishes that enlarging the image alone does not solve
contact ambiguity. Do not promote this experiment to production on these counters.
Review the synchronized comparison, especially 38–40, 128–131, 151–154, 158–161 and
205–208 seconds. The current crop selection uses image-to-ground projection only as
a search heuristic; an airborne near-side ball can project onto the far court.
A production gate must disambiguate depth/flight and protect near-side contacts.

Validation: complete pytest suite, ruff for the harness, handoff gate and full
1,737-frame native demo regression (27 bounces, 35 racket hits in current code).
