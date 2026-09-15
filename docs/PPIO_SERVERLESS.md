# PPIO Serverless benchmark

This adapter benchmarks the unchanged production pipeline on a PPIO RTX 4090 before
remote uploads are added to the product. It deliberately accepts only the bundled
`assets/demo/demo.mp4`, so the first measurement isolates GPU inference and rendering
from object-storage transfer time.

## Secret boundary

Never commit API keys or registry passwords. The caller uses `PPIO_API_KEY` from the
host environment. GitHub Actions uses these repository secrets:

- `PPIO_REGISTRY_USERNAME`
- `PPIO_REGISTRY_PASSWORD`
- `PPIO_IMAGE_REPOSITORY`, for example
  `image.ppinfra.com/<account-namespace>/netcast-tennisvision`

The upload username, password, and namespace are shown in PPIO Console under Security
Credentials -> Image Registry Upload Credentials.

## Build

Run the `Build private PPIO benchmark image` workflow manually from the branch being
tested. The workflow builds the repository revision, installs checksummed runtime
assets from `assets/manifest.json`, and pushes only to the account's private PPIO
registry.

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

