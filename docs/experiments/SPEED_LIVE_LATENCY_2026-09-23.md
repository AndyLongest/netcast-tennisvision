# Speed v2 local pseudolive latency experiment

Local RTX3050Ti laptop, flat2 original1080p30, source66–81s, off/on/on/off.
Fresh ball/person/pose inference in all arms; ffmpeg -re input, batch16,
bounded90-frame queue. Same pose scheduling both arms. Speed hook consumes each
arrived20-frame support once and calls maintained analyze_speeds. This adapter
is NOT a deployed live worker; excludes network/browser latency. Its temporary
contact boundaries use local landing impulses, not complete offline event logic.
Scripts/data: outputs/speed_live_bench/{run.py,component.py,speed_run_*.json,component.json}.

| Mode | Source-result median s | P95 s | Dropped /450 |
|---|---:|---:|---:|
| off | 4.424 | 7.433 | 216 |
| on | 4.471 | 7.783 | 184 |
| on | 4.529 | 7.881 | 184 |
| off | 4.376 | 7.420 | 171 |

Both enabled arms emitted ZERO valid speeds. Heavy baseline dropping means this
A/B cannot establish the latency of productive speed inference. Do NOT claim
speed costs only the ~0.1s median difference or is realtime-safe from these runs.

## Productive component measurement

Full300s uses identical cached offline observations/events, actual measured CPU
calls over450 supports of20frames. Simulated source-paced single-worker queue,
NOT a wall-clock live test or detection accuracy evaluation. Does not rerun or
include upstream inference. 80 supports yielded accepted speeds. 31.85s total
compute. Accepted computation median163ms/P95242ms/max323ms. All calls P95274ms,
max1010ms (including rejected expensive fits). Simulated queue P950,max343ms.
Accepted speed age relative to window midpoint: median479ms/P95559ms/max640ms,
including acquisition of the rest of the support. Contact, calibration and
network buffering can increase actual latency beyond these figures.

Conclusion: average compute fits source time under these component conditions,
but synchronous insertion can stall detection for up to about1s in this sample.
Use an independent bounded CPU worker for an eventual live integration; skip
stale speed tasks rather than delaying ball/landing delivery. This is a
recommendation, not an implemented/deployed change. Cloud hardware untested.
