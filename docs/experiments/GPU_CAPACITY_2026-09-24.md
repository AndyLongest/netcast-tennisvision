# Cloud GPU live-capacity screen — 2026-09-24

## Scope and acceptance

The experiment uses the maintained live worker with ball detection, person detection,
tracking, landing, and the enabled background speed worker. Each concurrent route
runs in its own Python process on the same GPU. A native-rate FFmpeg camera producer
pushes 1080p30 H.264 RTMP through the external ZLMediaKit server; the worker consumes
that stream and computes events. A route only counts as near-lossless if inference
queue drops remain below 0.5% and the queue does not grow over the clip. Queue drops
matter because the live worker fills their timeline positions with empty observations.

This is a **GPU-side media/inference capacity screen**, not browser glass-to-glass
latency or a complete ECS result-relay test. The one-route production browser/relay
smoke is separately recorded in `docs/HANDOFF.md`.
The demo3 court baseline separation is 57.1% of image height (high view);
flat_deemo is 25.5% (low view), according to their confirmed corner metadata.

## Actual PPIO L40S measurements

One temporary `L40S.22c125g` (NVIDIA L40S, 46,068 MiB device memory),
`production-v31` image with current Python source overlaid. All temporary instances
were released (`released: true` in each output journal).

| Source | Concurrent streams | GPU utilization mean / peak | VRAM peak | Queue drops per route | Speed results per route | Peak internal backlog | Assessment |
|---|---:|---:|---:|---|---|---|---|
| demo3, 35 s | 1 | 15.7% / 70% | 8.3 GiB | 1/1017 (0.10%) | 0 | 0.97 s | Pass for this clip; no valid speed output |
| demo3, 35 s | 3 | 49.2% / 100% | 24.8 GiB | 18/1017 (1.77%) each | 0 | 1.10 s | Misses strict drop criterion |
| demo3, 35 s | 5 | 53.0% / 100% | 41.4 GiB | 114–132/1017 (11.2–13.0%) | 0 | 1.23 s | Fail |
| demo3, 90 s | 2 | 31.8% / 100% | 16.5 GiB | 1–4/2667 (0.04–0.15%) | 0 | 0.97 s | Pass for this clip; no valid speed output |
| flat_deemo, 90 s | 1 | 2.2% / 67% | 8.3 GiB | 336/2614 (12.85%) | 3 | 2.50 s | Early burst anomaly; not representative of steady state |
| flat_deemo, 90 s | 2 | 27.5% / 100% | 16.5 GiB | 1–2/2670 (0.04–0.07%) | 4–6 | 0.97 s | Pass for this clip |
| flat_deemo, 90 s | 3 | 39.1% / 100% | 24.8 GiB | 3/2670 (0.11%) each | 4 each | 0.97 s | Pass for this clip |
| flat_deemo, 90 s | 4 | 62.0% / 100% | 33.1 GiB | 46–64/2640–2670 (1.72–2.42%) | 1–3 | About 1 s | Borderline; fail strict drop criterion |
| flat_deemo, 90 s | 5 | 60.4% / 100% | 41.4 GiB | 215–231/2670 (8.05–8.65%) | 1–7 | 1.17 s | Fail |

The initial same-process concurrency screen was misleading: 3 routes lost 26–27%
and 5 routes 43–47%. Separate processes reduced that loss substantially, indicating
that Python process scheduling/GIL contention is material. The 90-second flat source
began with an isolated 336-frame loss in the 1-route cold run; the subsequent 3-route
run on the same instance did not repeat it. Do not claim that one L40S always handles
three routes losslessly based on these short clips. Four routes appear to recover to
real-time after startup but lose too many frames for the strict event-quality target.
Five routes repeatedly lose frames and approach the VRAM limit.

The highest **near-lossless count observed across both tested viewpoints** is two.
Three passed the 90-second flat source but missed the strict 0.5% drop criterion
on the 35-second demo3 source. A longer, repeated trial and application-level
relay/browser test are required before adopting even two as a production SLA.

Evidence: `outputs/cloud_capacity_process/`,
`outputs/cloud_capacity_process_flat/`, and
`outputs/cloud_capacity_process_flat4/`,
`outputs/cloud_capacity_process_high2/`, and
`outputs/cloud_capacity_process_flat2/` contain uploaded source, full samples,
terminal snapshots, process logs, and release journals. The experimental drivers
are in ignored `outputs/benchmark_multiprocess_remote.py` and
`outputs/benchmark_capacity_process*_cloud.py`.
The repository's `python -m pytest -q` suite passed after the experiment;
the maintained runtime was not edited for this benchmark.

## Alibaba Cloud status

Read-only ECS `DescribePrice` and `DescribeAvailableResource` on 2026-09-24 found
stock and instance price quotes in Shanghai/Shenzhen. The request specified an
official Linux GPU-driver image and 100 GiB ESSD PL1; other line items (for
example, network traffic) require a separate quote and actual bill check:

| GPU | ECS type | Hourly | Monthly quote | Capacity measurement |
|---|---|---:|---:|---|
| T4 16 GiB | `ecs.gn6i-c4g1.xlarge` | ¥11.84 | ¥1,774 (promotion) | Not run |
| A10 24 GiB | `ecs.gn7i-c8g1.2xlarge` | ¥9.74 | ¥4,675.66 | Not run |
| L20 48 GiB | `ecs.gn8is.2xlarge` | ¥14.63 | ¥7,019.25 | Not run |

The account has under ¥100 usable balance/vouchers, below Alibaba's minimum for
starting pay-as-you-go ECS. A PAI-DSW/EAS trial must be claimed in the primary
account before use; eligibility/claim status is pending. No Alibaba GPU was
provisioned and no Alibaba runtime charge was incurred. The RAM credential can
query ECS quotes but cannot call BSS `QueryAccountBalance`, so no balance was
inferred from an API response. `outputs/aliyun_price_snapshot.json` records the
exact read-only quote and stock responses.

The test budget is ¥10 actual spend. Do not represent these Alibaba prices as
measured inference capacity or claim a winning Alibaba SKU until an actual GPU
run and cleanup have completed.
