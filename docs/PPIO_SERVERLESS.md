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

## GPU inventory fallback

The local lifecycle automatically tries `L40S.22c125g`, `L40S.28c125g`,
`4090.16c125g`, `4090.16c62g`, then `4090.16c96g.v2` when creation explicitly reports
insufficient inventory. This applies to normal uploads and live sessions. It is an
availability priority, not a claim that each GPU is slower than the previous one.
No smaller/unvalidated GPU, CPU execution or different region is selected automatically.
The existing CUDA requirement, model image and one-GPU request remain intact.

`TENNISVISION_PPIO_PRODUCT_ID` chooses the first product. A known default product starts
at its position in that list; an unknown custom product has no implicit alternatives.
`TENNISVISION_PPIO_FALLBACK_PRODUCTS` overrides the alternatives with comma-separated
product IDs; an empty value disables fallback. Duplicates are removed, with at most
eight candidates. Set these in the backend process environment before starting it.
Each new job starts again at the preferred product.

Each candidate gets its own root filesystem limit check. Explicit rootfs validation
can retry once on the same product. Inventory rejection can then advance to the next
product. Auth, balance, image, quota, timeout, network and missing-instance-ID errors
stop immediately: ambiguous creation is not safe to repeat. Cancellation is checked
between candidates. Exhaustion reports all attempted products and suggests retrying
later or explicitly selecting local GPU; there is no automatic local execution.

Progress names the attempted product. `cloud_product_id` records the chosen product
and `cloud_gpu_attempts` records attempts. The legacy `cloud-live-l40s` execution-target
string remains for client compatibility and must not be interpreted as hardware identity.
Existing success/failure/stop cleanup and orphan journals still own the one created
instance. Fallback does not create multiple GPUs in parallel. Live latency on alternative
hardware still needs measurement; successful provisioning is not a realtime guarantee.

The five default SKU IDs were verified against the provider product catalog on
2026-09-21. Catalog availability hints are not treated as a guarantee; instance creation
is authoritative. Automated tests simulate stock exhaustion and successful fallback
without consuming cloud resources.

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

After the final snapshot has been copied to local status storage and the temporary GPU is
stopped, the lifecycle deletes the ECS session resource. Normal completion, explicit stop,
and failure all use the same cleanup path. A 24-hour relay-side TTL janitor removes only
orphaned snapshots left by a client or network crash.

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
