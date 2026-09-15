# Quasi-realtime experiment (2026-08-26)

The production default remains the full native-rate path. Set
`TENNISVISION_QUASI_REALTIME=1` only for development experiments; its Pass A cache is
isolated from production.

## Controlled setup

- Input: fixed `demo.mp4`, 1737 frames at 29.9139fps (58.07s).
- Hardware/runtime: the same RTX 3050 Ti environment for every run.
- RacketVision candidates were cached identically; Pass A, tracking, event fusion,
  scene export and annotated rendering were timed end to end.
- Production control: person inference and nine-line court fit on every frame.

| Variant | End-to-end | Speed-up | Positioned | Bounces | Hits | Decision |
|---|---:|---:|---:|---:|---:|---|
| Production control | 231.48s | 1.000x | 1220 | 27 | 31 | baseline |
| Carry previous player, court stride 2 | 190.26s | 1.217x | 1268 | 28 | 29 | rejected |
| Interpolate players, court stride 2 | 214.67s | 1.078x | 1220 | 29 | 31 | rejected |
| Full-rate players, court stride 2 | 224.01s | 1.033x | 1220 | 28 | 30 | rejected |
| Full-rate players, court calibrated once | 128.75s | 1.798x | 1220 | 27 | 31 | accepted |

The interpolated variant retained every baseline bounce and zone, with 0.01m median
matched landing displacement, but added false touchdown candidates. The court-only
variant still changed the 7.15s racket contact into a landing. Event-count equality is a
hard acceptance gate, so neither shortcut is enabled in production.

The fixed-camera variant does not skip native video frames and does not reduce player or
ball inference frequency. It removes only redundant per-frame court optimization and is
now the production default. Its 27 touchdown times matched the control within 0.02s; all
but one retained the same zone. The exception moved 0.08m across the far service-line
boundary at 54.16s and remains flagged as a visual-review boundary case.

A 240-frame FP32/FP16 player micro-benchmark produced identical boxes and a 1.065x model
speed-up. This is too small to justify a production precision-path change without a full
event regression.

## Finding

Player and court geometry are causal evidence for ball association and hit-versus-bounce
classification, not presentation-only overlays. Blind frame skipping is therefore not a
lossless optimization. The next viable design is a validated player tracker that outputs
per-frame boxes with uncertainty, plus report/video decoupling so structured results can
be shown before annotated encoding finishes.
