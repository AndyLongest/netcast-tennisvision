# Event-preserving ball sampling experiment — 2026-09-16

## Question

Can the frozen RacketVision model skip stable in-flight timestamps while preserving the
exact hit, landing and rally events needed by the product?

The production control is the native-rate `assets/demo/demo.mp4` regression: 1,737 source
frames, 1,123 detector-backed observations, 29 confirmed bounces and 35 racket hits.

## Variants tested

| Variant | Model timestamps | Ball-model timing | Bounces | Hits | Decision |
|---|---:|---:|---:|---:|---|
| Production dense control | 1,737 (100%) | 62.6s including setup | 29 | 35 | retained |
| Raw-candidate event guards v1 | 1,664 (95.8%) | 67.7s | 28 | 32 | rejected |
| Coherent-run event guards v2 | 1,547 (89.1%) | 64.2s | 28 | 33 | rejected |
| Stride-two scout + coarse-track recovery | 1,402 (80.7%) | 37.5s scout + 25.1s recovery, excluding repeated setup | 26 | 31 | rejected |
| Four heatmaps from one non-overlapping window | 435 windows | 23.0s loop, 29.4s total | 7 | 2 | rejected |

The two-pass variant decoded the video twice and rebuilt model/background state, so its
19.3% reduction in inferred timestamps did not reduce ball-stage wall time. More
importantly, the sparse scout changed track birth and contact evidence before the recovery
planner knew which timestamps mattered.

The grouped-output variant proved that the checkpoint's four predictor channels cannot be
substituted for production causal last-channel windows. It produced temporally incompatible
peaks: the static filter removed 879 of 1,633 detections and the accepted trajectory
collapsed to 185 real anchors.

## Production decision

Production remains native-rate causal inference. None of these paths is referenced by the
notebook or enabled by default. Reproducibility helpers are isolated in
`vision/adaptive_ball.py` and opt-in candidate APIs; they are not supported fallbacks.

The next legitimate route is a separately trained or distilled event-oriented model whose
training target explicitly preserves racket contacts and ground impulses, followed by the
same 29/35 event and manual-window regression. Adding more sample-specific guard thresholds
to the frozen position model is not an acceptable acceleration strategy.

## All-route follow-up

The owner subsequently requested that every remaining route be tried rather than selected
on theory alone.

| Route | Measurement | Decision |
|---|---|---|
| Browser event overlay | local `demo` stopped after report at 134.4s; recorded cloud `deemo2` report/complete were 304.4s/497.8s | accepted for normal non-corrected display; no inference change |
| ONNX Runtime CUDA FP16 | 796.6ms per batch versus PyTorch FP16 124.4ms; max heatmap delta 0.000381 | rejected, 6.4× slower |
| TensorRT execution provider | provider DLL could not load without TensorRT runtime and fell back to CPU | not deployable in the current audited runtime |
| Sparse/two-stage event scout | 26/29 bounces, 31/35 hits and no ball-stage saving | rejected |

The event-overlay branch runs only after the structured scene has already been finalized.
It therefore cannot change detection, player identity, rally segmentation, hits or bounces.
The optional backend benchmark lives at `tools/benchmark_ball_backends.py`; ONNX packages
were removed from the application environment after the experiment.
