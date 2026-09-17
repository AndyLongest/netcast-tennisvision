# Real-time landing display simulation (2026-09-16)

> Superseded for performance claims by the end-to-end mode in `web/live-lab.html`.
> The measurements below remain the causal-display baseline only.

This experiment separates two questions that must not be conflated:

1. Can a confirmed landing be delivered and displayed on a live timeline?
2. Can the current neural and temporal pipeline produce those confirmations faster than
   a 30fps camera supplies frames?

## Causal display simulation

Input was the frozen `assets/demo/demo.mp4`: 1,737 frames at 29.9139fps, 58.07 seconds,
with 29 reviewed landing events. The simulator never exposes an event at its offline
touchdown frame. The current bounce descriptor is centred on a 21-frame window, so each
event is held until at least ten future frames have arrived. A later explicit
`decision_frame` would take precedence.

| Metric | Result |
|---|---:|
| Delivered events | 29 / 29 |
| Minimum causal delay | 334.3ms |
| Median causal delay | 345.7ms |
| P95 causal delay | 366.4ms |
| Maximum causal delay | 367.1ms |

The generated demonstration is
`outputs/realtime/realtime_landing_demo.mp4`. It uses the source-video clock, clears the
minimap when the rally id changes, displays only delivered events, highlights an in-court
zone after confirmation, and uses a red cross for an out landing. This output validates
event scheduling and browser-style presentation only. Its landing coordinates come from
the frozen reviewed report, not from a new online inference pass.

## Native ball-inference throughput

A separate uncached run executed the frozen RacketVision detector over all 1,737 native
frames on the local RTX 3050 Ti, batch size four, with the accepted bounded prefetch path.

| Metric | Result |
|---|---:|
| Model-loop throughput | 33.5 source frames/s |
| Required camera throughput | 29.9139 frames/s |
| Complete one-off stage wall time | 62.953s |
| Source duration | 58.067s |

The steady model loop is 1.12× faster than the incoming camera rate. The complete run is
slower than the source because it also constructs a median background and loads the model.
Those are session-start tasks in a live service and must be completed before play begins,
not repeated during a rally.

## Decision

The experiment proves that the UI/event channel can present reviewed landings with about
0.35 seconds of algorithmic look-ahead, and that the ball network alone can narrowly keep
up on the development GPU after warm-up. It does **not** yet prove complete live analysis:
the production tracker still performs whole-clip fragment merging and RTS smoothing, and
the person branch has not been integrated into one bounded-latency stream. The next
prototype must replace those non-causal operations with a fixed-lag state machine and
measure ball, player, tracking, event confirmation and persistence together on the target
server GPU.

## End-to-end successor

The live-lab now also has a separate true-chain mode implemented by
`streaming/live_experiment.py`. It performs real-time FFmpeg publication, remote
ZLMediaKit relay, independent RTMP decode, frozen ball and person inference, bounded
eight-second tracking, fixed-lag contact decisions, browser frame delivery, and a
post-run comparison against the offline baseline. Ingest runs independently from GPU
inference and buffers compressed decoded frames, so a slow GPU creates visible queue
growth instead of silently dropping native frames.

Two complete RTX 3050 Ti runs consumed and analyzed all 1,737 input frames in 284.31 and
291.91 seconds for a 58.07-second stream: **4.90–5.03× source duration**. The second run's
queue peaked at 28.65 seconds of source time. Its fixed-lag detector produced 26 events;
23 matched the 29-event offline reference within 0.45 seconds and 2 metres, yielding
79.3% measured recall and 88.5% precision under that declared matching rule. These are
agreement metrics against the existing offline output, not independent human ground truth.

This is an honest negative result for the current local hardware and live algorithm. The
previous ~0.35-second number was only the event window after an already available
trajectory and was never complete pipeline latency. The final event comparison is written
to ignored runtime file `data/live_lab_last.json` by every completed experiment.
