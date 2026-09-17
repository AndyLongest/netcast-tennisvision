# PPIO Serverless benchmark

This adapter benchmarks the unchanged production pipeline on a PPIO RTX 4090 before
remote uploads are added to the product. It deliberately accepts only the bundled
`assets/demo/demo.mp4`, so the first measurement isolates GPU inference and rendering
from object-storage transfer time.

The internal live-lab can also run this bundled source as a real RTMP stream on a
temporary GPU instance. That experiment does not upload or download a finished video:
the worker pulls the configured ZLMediaKit stream and writes live status and landing
events to the authenticated ECS result relay. The local service reads that relay rather
than polling the GPU instance for results. When RTX 4090 inventory is unavailable, the same frozen code path may be measured
on the explicitly selected L40S product; results must name the actual GPU and may not be
reported as a 4090 benchmark.

The first completed server measurement is recorded in
[`experiments/PPIO_BENCHMARK_2026-09-15.md`](experiments/PPIO_BENCHMARK_2026-09-15.md). Use its first-processing
number for capacity planning; its faster exact-video repeat intentionally includes
persisted caches.

## Secret boundary

Never commit API keys or registry passwords. The caller uses `PPIO_API_KEY` from the
host environment. GitHub Actions uses these repository secrets:

- `PPIO_REGISTRY_USERNAME`
- `PPIO_REGISTRY_PASSWORD`
- `PPIO_IMAGE_REPOSITORY`, for example
  `image.ppinfra.com/<account-namespace>/netcast-tennisvision`

Live workers additionally receive `TENNISVISION_RESULT_RELAY_URL` and
`TENNISVISION_RESULT_RELAY_TOKEN` from the trusted local runtime configuration. The token
is shared only by the ECS relay, temporary GPU worker and local backend; it never reaches
browser JavaScript or version control.

## ECS result relay

The media ECS runs `netcast_tennisvision.streaming.result_relay` as the
`netcast-result-relay` container on host port 8001. Existing Nginx TLS routing exposes it
under `/moralspaceTennisApi/netcast-results`; persistent snapshots live outside the
container. `GET .../healthz` is public for operations, while every session read/write
requires the bearer token. The L40S writes complete coalesced snapshots with `PUT
/v1/sessions/{session_id}` and the local backend reads the same resource. The service
marks stored snapshots with `result_relay=ecs`, which the local lifecycle requires before
showing or accepting a result.

The upload username, password, and namespace are shown in PPIO Console under Security
Credentials -> Image Registry Upload Credentials.

## Build

Push an annotated `production-v*` tag to build a production release. The workflow uses
`Dockerfile.release` to layer current source and web assets over the already-audited
private `production-v1` runtime, validates that all four frozen production checkpoints
are reachable, and pushes the versioned tag only to the account's private PPIO registry.
This avoids rebuilding CUDA, Python dependencies, and model weights for every code-only
release. The full `Dockerfile` remains the reproducible path for intentionally rebuilding
the ML runtime and its checksummed assets.

## Endpoint guardrails

Use an Async endpoint with one RTX 4090, minimum workers 0, maximum workers 1, maximum
concurrency 1, and a short idle timeout. The container command is already:

```text
python -m netcast_tennisvision.cloud.worker
```

Submit only this small job during phase one:

```json
{"input":{"mode":"benchmark_demo","timeout_seconds":3600}}
```

The response contains report-ready time, complete time, GPU identity, compact scene
counts, and output hashes. It does not send the generated video through the 4 MiB Async
API response. Object storage and arbitrary uploads belong to phase two, after speed and
regression evidence pass review.

If PPIO's Async gateway cannot register a newly created endpoint, the same benchmark can
run as a Sync HTTP endpoint on port 8000. Start
`python -m netcast_tennisvision.cloud.http_worker`, use `/health` for health checks, and
POST the inner input object to `/benchmark`. This changes only transport, not analysis.
