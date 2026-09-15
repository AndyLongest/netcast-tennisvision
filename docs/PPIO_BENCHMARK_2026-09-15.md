# PPIO RTX 4090 benchmark — 2026-09-15

This is an end-to-end measurement of the unchanged native-frame production pipeline on
the bundled 1280×720 `assets/demo/demo.mp4` (1,737 frames, 29.9139 fps, 58.07 seconds).
Upload time and one-time container provisioning are excluded. Analysis, report creation,
and annotated-video rendering are included.

## Measured latency

| Run | Report ready | Video complete | Report / source | Complete / source |
|---|---:|---:|---:|---:|
| First processing on RTX 4090 | 72.589s | 104.876s | 1.25× | 1.81× |
| Exact-video repeat with persisted caches | 22.022s | 54.066s | 0.38× | 0.93× |

The first-processing result is the production capacity number. The repeat is useful for
resume/retry behavior, but must not be presented as latency for a new upload because it
reuses persisted intermediate results for the same video fingerprint.

Against the accepted fresh-cache RTX 3050 Ti result recorded in
`PERFORMANCE_EXPERIMENT_2026-09-15.md` (137.89s report, 186.80s complete), the first RTX
4090 run reduced report latency by 47.4% (1.90× throughput) and complete latency by 43.9%
(1.78× throughput).

## Regression evidence

Both RTX 4090 runs produced byte-identical `scene3d.json` and annotated-video files:

- 1,737 input frames, with no frame dropping or interpolation
- 1,220 positioned-ball frames
- 30 bounce events
- 35 hit events
- scene SHA-256: `179174ac7da2ecf6ea4e50cc897a4cf219be18a538ef1d3ef0cc7a6bfc31a621`
- video SHA-256: `7fe7f735a41960f8e137d6c7eb1832c8091c2a1ea589ea28604fa527584adacb`

This is deterministic across the two server runs, but it is not byte-identical to the
older reviewed golden scene. The golden has 1,220 positioned frames, 29 bounces, and 35
hits. The server added one observed bounce at about 40.449s. Do not claim complete
accuracy parity until that edge event is reviewed and the Python/CUDA runtime is locked
by image digest. The existing current-tree/golden discrepancy described in
`PERFORMANCE_EXPERIMENT_2026-09-15.md` also remains open; this benchmark does not redefine
the acceptance target.

## Test environment

- GPU: NVIDIA GeForce RTX 4090 24GB
- CUDA reported by PyTorch: 12.8
- PPIO base image: `image.ppinfra.com/prod-gpucloudpublic/pytorch:2.7.1-cuda12.8`
- production weights: checksummed frozen assets from `assets/manifest.json`
- production algorithm: unchanged; native 29.9139 fps processing

