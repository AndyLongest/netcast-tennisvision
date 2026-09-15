# Cloud acceleration experiment — 2026-09-16

## Scope

The benchmark uses `deemo2.mp4`: 133.679 seconds, 2,802 container frames,
2880×1620 H.264 and 126,659,399 bytes. Human time spent responding to a manual court
prompt is excluded. Ball, player, tracking and landing parameters were not changed.

## End-to-end result

| Runtime | Report ready | Annotated replay complete | Complete/video ratio |
|---|---:|---:|---:|
| production-v7, serial transfer, fresh court search | ~395s | ~598s | 4.48× |
| production-v8, four transfer workers, verified camera profile | 304.4s | 497.8s | 3.72× |

The accepted path saves about 100 seconds (16.8%) end to end and makes the report
available about 91 seconds earlier. The same-camera lookup matched 920 ORB/RANSAC inliers,
0.994 inlier ratio and 0.171px alignment movement; court setup fell from roughly 86
seconds to roughly 2 seconds and did not request manual confirmation.

The v8 report contains 2,769 decoded analysis frames, 14 confirmed bounces and 7 racket
hits. Its `scene3d.json` SHA-256 is
`3fc172561943e1f35e796f1900368f49391794c7d3bf9b0f896d832e8938e6cd`.

## What did and did not help

Four concurrent checked 8 MiB parts reduced one measured input transfer from about 171
seconds to about 143 seconds, but a repeat took about 159 seconds. PPIO endpoint bandwidth
variation is larger than much of the client-side gain. Parallel range download is retained
because it is byte-exact and bounded, but it did not remove the provider bottleneck.

The completed v8 replay remained 126,335,582 bytes. Its identical size and bitrate to the
x264 control strongly suggest that the FFmpeg NVENC preflight selected the safe x264
fallback in this image. Encoder selection needs explicit telemetry before claiming an
NVENC speedup.

## Same-spec bitrate experiment

A 30-second excerpt of the 2880×1620 replay was encoded at x264 veryfast CRF22 while
preserving resolution, frame rate, pixel format and H.264 compatibility:

| Profile | Size | Relative size |
|---|---:|---:|
| Existing CRF20 reference | 28,345,696 bytes | 100% |
| CRF22 candidate | 19,317,331 bytes | 68.2% |

FFmpeg SSIM against the reference was 0.992393. Cloud jobs now request CRF22 when NVENC
is unavailable, but the final cloud timing is not yet measured: the next PPIO instance
request was rejected before allocation with `balance not enough`. Do not quote a further
speedup until that exact end-to-end run completes.

## Rollback controls

- `TENNISVISION_TRANSFER_WORKERS=1` restores serial transfer.
- `TENNISVISION_CAMERA_PROFILES=0` disables camera reuse.
- `NETCAST_VIDEO_ENCODER=x264` forces software encoding.
- `NETCAST_X264_CRF=20` restores the previous cloud bitrate.
