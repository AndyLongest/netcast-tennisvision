# Native-rate performance experiment — 2026-09-15

This experiment evaluates latency reductions from conservative to aggressive on the
fixed 1280×720 `assets/demo/demo.mp4` (1,737 frames at 29.9139fps). Every run decoded and
processed all native frames. Each end-to-end run started with an empty Pass-A and ball
cache and used the same RTX 3050 Ti Laptop GPU.

The immutable acceptance target is the bundled reviewed scene: 1,220 positioned frames,
1,123 detector anchors, 29 bounces and 35 hits. A separate current-tree control issue is
recorded below; speed changes are never allowed to redefine that target.

## Results

| Variant | Report ready | Video complete | Positioned / anchors | Bounces / hits | Scene vs current control | Decision |
|---|---:|---:|---:|---:|---|---|
| Serial control, first run | 168.7s | 216.9s | 1220 / 1123 | 28 / 32 | reference | diagnostic only |
| Serial control, warm repeat | 134.4s | 169.1s | 1220 / 1123 | 28 / 32 | byte-identical | diagnostic only |
| Full-rate person prefetch | 138.7s | 187.0s | 1220 / 1123 | 28 / 32 | byte-identical | accepted mechanism |
| Full-rate person prefetch, warm repeat | 143.5s | 192.7s | 1220 / 1123 | 28 / 32 | byte-identical | accepted mechanism |
| Prefetch + deferred low-res masks | 144.4s | 213.1s | 1220 / 1123 | 28 / 32 | byte-identical | rejected: render pays the work back |
| Person inference every second frame | 131.1s | 163.7s | 1220 / 1126 | 29 / 33 | changed | rejected: changes physical evidence |

Whole-pipeline wall time is noisy: cold model loading, GPU state and x264 changed the
same serial configuration from 216.9s to 169.1s. It must not be used alone to claim a
30-second improvement. In the adjacent warm runs, the integrated person portion of Pass A
fell from 44.75s to 41.18s (8.0%, 3.57s), but unrelated setup, ball, downstream and x264
variation made the full prefetch run slower overall. The honest production claim is a
small lossless reduction in one owned stage, not a guaranteed end-to-end percentage.

The isolated person-stage crossover benchmark ran in one process in the order serial,
prefetch, prefetch, serial. It includes full-rate YOLO segmentation, full-resolution mask
expansion, dilation and PNG encoding:

| Mode | Runs | Mean | Throughput | Output hash |
|---|---:|---:|---:|---|
| Serial | 42.976s, 41.211s | 42.094s | 41.3 FPS | `3208cc18…ac34c6` |
| One-batch prefetch | 28.250s, 28.367s | 28.309s | 61.4 FPS | `3208cc18…ac34c6` |

Prefetch therefore reduces this owned stage by 32.7% (1.49× throughput) while producing
the exact same per-frame boxes and masks. The complete `scene3d.json` from the full-rate
prefetch run is also byte-for-byte identical to its serial control. FFmpeg `framemd5`
verification produced the same SHA-256 (`f722f5b5…6ba2c0`) for all decoded annotated
frames, so the visible video is identical as well. Production enables this overlap by
default; `TENNISVISION_PERSON_PREFETCH=0` is the immediate rollback.

Deferred masks save work in Pass A but make the rendering consumer repeatedly rebuild
large masks; at 720p this increased final rendering from about 48s to 69s. The code path
was removed rather than retained as dead complexity.

The stride-two experiment is not an accuracy-preserving speed mode. It changed accepted
detector anchors (1123→1126), occlusion fills (97→94), player identity evidence, added a
14.11s landing, changed a 31.42s observed landing to modelled, and moved a racket contact
from 31.06s to 31.16s. It remains opt-in research only.

## Golden-sample discrepancy discovered

Both independent full-rate serial runs and the byte-identical prefetch run currently
produce 28 bounces and 32 hits, while the reviewed bundled report contains 29 and 35.
The tracked-ball counters still match exactly (1220 positioned, 1123 detector-backed).
This discrepancy predates the person-prefetch change: the serial control already has it.
The bundled report was generated on 2026-09-05, before the later player-identity and
training-mode commits. Those post-sample changes are the first history boundary to audit;
the golden report must not be overwritten until the event regression is explained.

Raw timing JSON and output scenes are under the ignored local directory
`outputs/benchmarks/person_pipeline_20260915/`.

## 1–1.5× source-duration target audit

The 58.07-second demo would need its report by 58.07–87.11 seconds. On this RTX 3050 Ti
Laptop, the accepted full-rate control needs about 134–138 seconds for the report. A
fresh-cache run after the experiments below completed at 137.89 seconds and the annotated
video at 186.80 seconds. Its scene matches the current serial control: 1,220 positioned
frames, 1,123 detector anchors, 28 bounces and 32 hits.

| Experiment | Native-rate result | Accuracy | Decision |
|---|---:|---|---|
| Four-worker median-background sampling | 9.44s → 4.70s | median image byte-identical | accepted, default |
| RacketVision end to end after background change | 59.19s | all 1,737 candidate rows exact | accepted |
| Concurrent ball and person branches | 100.26s → 87.23s combined stage | both output hashes exact | opt-in; GPU contention prevents target |
| Person input 640 | report 117.13s | anchors/events/landings changed | rejected |
| RacketVision batch 6/8 | about 58.3s | floating candidate tuples changed | rejected |
| cuDNN benchmark mode | 59.28s | changed tuples, no speed gain | rejected |
| ONNX Runtime CUDA, ordinary binding | 235.35s | 37 presence differences | rejected and uninstalled |
| Lossless Pass-A identity crops | 3.74s → 6.12s for identity stage | identity metrics equal | rejected and removed |

The measured lower bound matters: the frozen ball branch alone occupies roughly 59
seconds, while full-rate person processing and the downstream physical/event pipeline
still have to run. Therefore a first-run 87-second report is not attainable on this 4GB
GPU without either changing numerical evidence or changing hardware. The production-safe
local improvement is retained, but the 1–1.5× duration service-level target should be
validated on the intended server GPU with GPU-resident decode/encode. The opt-in parallel
branch can be enabled there with `TENNISVISION_PARALLEL_DETECTORS=1` and must undergo the
same complete-scene regression before becoming that server's default.
