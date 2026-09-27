# L40S speed latency evaluation

User requirement: future performance/latency/capacity tests run on cloud GPU, not
local-machine substitutes. Persisted in docs/HANDOFF.md.

## Live ABBA

Actual NVIDIA L40S, PPIO L40S.22c125g, production-v30 image plus isolated current
Python source upload. flat2 first90s stream-copy excerpt, original30fps1080p.
FFmpeg-re -> external ZLMediaKit RTMP -> maintained live worker at640x360. Pose
on in all arms; speed off/on/on/off. Synchronous hook processes each arrived
20-frame support once. No inference cache. Speed contact boundaries are local
landing impulse candidates; not a complete production live speed integration.

| Speed | Batch-last delay median ms | P95 ms | Queue drops | Accepted speeds |
|---|---:|---:|---:|---:|
| False | 208.8 | 432.1 | 0 | 0 |
| True | 216.3 | 380.5 | 0 | 0 |
| True | 222.2 | 371.0 | 0 | 0 |
| False | 64.8 | 251.4 | 0 | 0 |

2674 frames ingested per run versus2700 source; zero inference queue drops is
not lossless transport. Delay measured at post-tracking/post-pose/post-speed
batch endpoint, before landing publication, relative to worker stream start;
not camera hardware timestamp or browser glass-to-glass. First5s excluded.
No valid speeds in on arms: these timings do NOT show productive speed overhead.
Underlying rejection cause requires further inspection; do not assert resolution
alone caused it. Baseline run variation also precludes subtracting one number
as the true speed overhead.

## Productive cloud fit measurement

Separate temporary L40S, same speed implementation; 97 known-accepted original
1080p cached observation windows across300s, repeated twice. 97/97 accepted each
repeat. Actual cloud CPU fit median123.5/123.9ms, P95159.6/160.4ms, max190.0/188.8ms.
Source-paced queue simulation has zero accumulated queue for these selected
windows. Adding observation lookahead yields midpoint-result age median427ms,
P95480ms, max506ms. These are selected successful windows; rejected expensive
fits and upstream model contention are excluded. Not end-to-end live latency.

Conclusion: productive fitting appears affordable on L40S CPU for this sampled
workload; full successful-speed live latency remains unverified because the live
adapter emitted none. Prefer an isolated bounded worker for eventual integration.
No production image/config changed. Both temporary instances released successfully.
Evidence/scripts/payloads/journals in outputs/speed_cloud_live and
outputs/speed_cloud_component. instance.json released:true in both directories.
