# API and report contracts

The desktop relay listens on `127.0.0.1:4173`. With no cloud configuration it runs the
pipeline locally. When `TENNISVISION_CLOUD_URL` and `TENNISVISION_CLOUD_TOKEN` are set,
the same browser contract streams `/api/*` and `/data/*` to a GPU server. The bearer
token stays in the Python relay and is never sent to browser JavaScript.

The GPU server sets `TENNISVISION_HOST=0.0.0.0`,
`TENNISVISION_EXECUTION_TARGET=cloud`, and `TENNISVISION_CLOUD_SHARED_SECRET`. Direct
requests without the matching bearer secret receive `401`. `GET /api/status` exposes
`execution_target` so operations and tests can prove where inference is running.

In on-demand PPIO mode, large relay-to-GPU transfers do not use `POST /api/analyze`
directly. The relay creates an upload session, sends independently checksummed 8 MiB
parts, completes the session, and only then enters the unchanged analysis endpoint over
container loopback. Generated MP4 files are retrieved with 8 MiB HTTP byte ranges. A
failed part is retried without retransmitting the whole video.

## HTTP endpoints

### `GET /api/status`

Returns the latest pipeline state merged with durable job identity.

| Field | Meaning |
|---|---|
| `state` | `idle`, `queued`, `running`, `needs_court_calibration`, `complete`, or `error` |
| `progress` | display progress from 0 to 100; not an ETA |
| `stage` | user-facing Chinese stage description |
| `job_id` | stable identity for the current/last job |
| `filename` | sanitized original filename |
| `video_fingerprint` | size plus first/last 256KiB SHA-256 identity |
| `fps` | measured native average frame rate |
| `workload_factor` | approximate work relative to 30fps |
| `display_correction` | display-only perspective choice, strength and normalized court corners |
| `calibration` | guided four-corner request when automatic confidence is low |

### `POST /api/analyze`

Body is the original video bytes. Required headers are `Content-Type`, `X-Filename` and
`X-Video-Fingerprint`. Optional `X-Display-Correction` (0–100) and
`X-Display-Corners` (normalized near-left, near-right, far-right, far-left JSON) select
display-only perspective correction. Only one job runs at a time. A repeated request with the same
fingerprint reattaches; a different video receives `409 analysis_in_progress` and never
overwrites the active clip.

### `POST /api/analysis/stop`

Force-stops the current ordinary analysis. Returns the latest status (202); poll
`GET /api/status` until `cancelled`. `stopping` keeps the job slot occupied while
in-flight transfers drain and the temporary PPIO GPU is stopped and deleted. A failed
release returns `stop_failed`; calling stop again retries the journaled cleanup. New
uploads remain blocked until cleanup succeeds. Local workers terminate their process
tree before cancellation completes. Fixed-worker relays forward this endpoint; they
stop the worker job but do not delete a separately administered permanent server.
Completed analysis history is preserved. Browser upload submission is settled before
sending stop, preventing a late upload from creating a job after cancellation.

### Local analysis history

- `GET /api/history` returns completed analyses newest-first. Each record includes the
  sanitized filename, timestamps, duration, landing count and local URLs for its source
  video, `scene3d.json`, and self-contained viewer.
- `DELETE /api/history/{job_id}` deletes exactly that record and its archived local
  artifacts. A deletion tombstone prevents the still-current completed job from being
  recreated by a later status poll.

History is owned by the trusted desktop relay, not by an ephemeral GPU worker. The source
video is snapshotted with a filesystem hard link when supported, while mutable report
files are copied so a later run cannot rewrite an older report. Record paths live below
the ignored `data/history/{job_id}/` directory.

### Chunked cloud transport

- `POST /api/upload/init` accepts filename, total size, video fingerprint and display
  correction metadata. Internal live-lab transfer additionally sets `purpose=live-lab`.
  It returns an opaque upload ID, part size and part count.
- `PUT /api/upload/chunk/{upload_id}/{index}` accepts one exact-size part and requires
  `X-Chunk-SHA256`.
- `POST /api/upload/complete` verifies that all parts exist with the expected total size,
  assembles them atomically and starts analysis. For `purpose=live-lab`, it stores the
  source for the causal live worker and does not start the offline notebook.

These endpoints are relay-facing transport APIs. Browser uploads continue to use the
stable `POST /api/analyze` contract against the trusted local service.

### `POST /api/camera-profiles`

The trusted relay may send the local user's bounded fixed-camera profile store to a new
isolated GPU worker. The endpoint accepts profile schema version 1, at most 12 profiles
and at most 3 MiB. A stored profile is never trusted by identity alone: the pipeline still
requires the normal ORB/RANSAC visual match before reusing its court corners. This is an
internal optimization endpoint, not a browser workflow.

### `POST /api/court-calibration`

Accepts the current request ID and four image-space corners in near-left, near-right,
far-right, far-left order. Geometry validation runs before the paused job continues.

### Internal live-lab endpoints

- `POST /api/live-lab/start` starts the bundled Demo on a temporary L40S when the trusted
  relay is in PPIO mode. Inside the isolated worker, `{source: external_rtmp}` arms a
  subscriber for the trusted stream name; it never receives the source video. The JSON
  body must include `court_corners`: four normalized `[x, y]` pairs ordered near-left,
  near-right, far-right, far-left.
- `POST /api/live-lab/upload` accepts original video bytes and `X-Filename`, validates the
  native frame rate, keeps the file on the camera-simulator host, then asynchronously
  provisions L40S and returns a local session id. It requires the same corner array
  serialized in `X-Court-Corners`. Once the worker reports
  `awaiting_stream`, local FFmpeg publishes the file at native speed to ZLMediaKit.
- `GET /api/live-lab/status?session_id=...&after_event=N` returns measured ingest/analysis
  clocks, queue backlog, newly confirmed online events, the trusted per-session HTTP-fMP4
  playback URL and the final offline comparison. In PPIO mode these snapshots have already
  travelled `L40S -> authenticated ECS result relay -> local service`; an unrelayed GPU
  response is rejected rather than used as a fallback.
- `GET /api/live-lab/frame?session_id=...` returns the latest JPEG decoded from the RTMP
  stream, not the source file. It is a diagnostics endpoint; the normal left-hand preview
  plays the ZLMediaKit HTTP-fMP4 stream directly and does not poll this endpoint.
- `POST /api/live-lab/stop` stops the named experiment session.

The local live relay mirrors remote state so the browser never receives provider
credentials. Uploaded experiments and the bundled Demo share the same automatic release
rule: success, failure and explicit stop all stop/delete the temporary GPU.

These endpoints are an internal benchmark surface. The media service host is trusted
server configuration (`TENNISVISION_ZLM_HOST` or ignored `data/live_lab_config.json`),
never a browser-supplied URL. This avoids turning the local relay into an arbitrary
network proxy.
`TENNISVISION_ZLM_WEBRTC_ORIGIN` (or `zlm_webrtc_origin` in the same ignored config)
selects the certificate-valid HTTP(S) media origin used by normal HTTP-fMP4 playback and
the optional WebRTC diagnostic handshake.

## `scene3d.json`

Top-level fields currently include:

- `fps`, `n_frames`: native timeline;
- `frames`: per-frame compact 3D/display state;
- `bounces`: confirmed touchdown events;
- `net_hits`: confirmed terminal net contacts;
- `hits`: racket-contact events including `rally_id`;
- `court_keyframes`: sorted `{frame, corners}` records in normalized image coordinates; use the latest at or before the displayed native frame;
- `rallies`: `{rally_id, start_frame, end_frame, end_reason, display_end_frame}`; the display interval is start-inclusive/end-exclusive;
- `play_mode`: temporal near/far head-count decision, confidence and vote distribution;
- `player_identities`: stable A/B display colours;
- `player_identity_metrics`: auditable OSNet-AIN sampling and side-change diagnostics;
- court/player/camera fields used by the self-contained viewer.

Important bounce fields:

| Field | Meaning |
|---|---|
| `touchdown_frame_f` | sub-frame touchdown estimate |
| `decision_frame` | first frame allowed to display the landing |
| `x`, `y` | court metres under the production court coordinate system |
| `zone` | service box, backcourt, doubles alley, or `Out` |
| `rally_id` | rally used to clear the minimap between points |
| `source` | observed, interpolated/modelled, or recovered-candidate evidence |
| `landing_confidence` | evidence score, not a calibrated probability |
| `landing_uncertainty_px` | image-space uncertainty when available |
| `line_call` | `in`, `out`, or `review`; only a fully supported `out` is rendered as a red cross |
| `line_call_confidence` | margin beyond the uncertainty/review band; zero for review and not a calibrated probability |
| `line_signed_margin_m` | positive centre distance inside the outside edge of the nearest legal line, negative outside |
| `line_uncertainty_m` | local world-space uncertainty propagated through the inverse homography |
| `line_nearest_boundary` | sideline/baseline (or corner pair) governing the call |
| `player_id` | identity of the player whose preceding strike produced this landing |
| `player_color` | stabilized dominant torso colour of that hitter, as a CSS hex colour |
| `identity_confidence` | OSNet pair-assignment margin; not a calibrated probability |
| `identity_rally_consensus` | share of confidence-weighted frames supporting the rally mapping |
| `identity_source` | rally-tracklet consensus plus landing-half corroboration, or an explicit unresolved fallback |

Hit records also expose `player_id` and `identity_confidence`. Identity is attached after
tracking and touchdown decisions, so it is not an input to ball or landing accuracy.

Net-hit records expose the last trustworthy lateral `x`, fixed court `net_y`, tracker
`frame`, later `decision_frame`, `rally_id` and preceding hitter identity. The UI must
wait for `decision_frame`, draw a red cross on the net, and treat the event as a hard
trajectory boundary rather than a missing observation.

Frame field `b` is the compact ball coordinate `[court_x, court_y, height]` or `null`.
Frame field `c` is the compact confidence/source class consumed by the viewer. Any schema
change must update `web/app.js`, the 3D exporter, frozen demo, manifest and tests together.

## Coordinate and rendering rules

- Court dimensions use ITF metres: 10.97m by 23.77m for doubles geometry.
- Homography coordinates are trustworthy for a ball only at a known ground contact.
- `decision_frame`, never the candidate frame, controls yellow highlighting.
- Out events render a red cross and never a yellow zone.
- Confirmed net hits render a red cross on the net and terminate the current flight.
- The minimap filters by the active `rally_id` and does not draw the trajectory.
- In-court landing dots use the hitter's A/B colour; out events remain red crosses.
- Perspective correction is applied after inference and annotation rendering but before
  the minimap is composited. It cannot change tracking, contacts, landing coordinates or zones.

Offline reports preserve decoder presentation timestamps in `frame_times` and contact `decision_t`. Browser frame selection uses this timeline, not only frame index divided by nominal fps. This preserves initial offsets and internal gaps without adding or interpolating ball observations.

`scoreboard_boundaries` lists native frames where persistent score-panel changes provide additional point boundaries. Unsupported panel layouts produce an empty list.

### Per-upload local GPU selection

`POST /api/analyze` accepts optional `X-Execution-Target: local-gpu`; omitted or `auto` preserves configured routing. Local selection validates CUDA in the server Python runtime (422 `local_gpu_unavailable` on failure), bypasses cloud provisioning, and runs the maintained pipeline with event overlays. Status/accepted responses identify `execution_target: local-gpu`. Calibration and force-stop follow that job target, including when PPIO is configured. Local means the computer hosting this API server, not a remote browser device. Requires NVIDIA drivers, CUDA PyTorch, installed project dependencies and model weights. No CPU/cloud fallback on failed CUDA validation.

### Viewpoint metadata (descriptive only)

`POST /api/viewpoint` accepts `{corners: [[x,y],...], width, height}` with four
original-source normalized corners in NL, NR, FR, FL order. The local relay returns
`version`, `category` (`low|high|unavailable`), `height_ratio`, `depth_width_ratio`,
`reason`, and `analysis_path: existing`. No job or GPU is started. Missing/invalid
geometry returns unavailable. Version 2 uses only <=30% low and >30% high. Bodies are limited to 4096 bytes.
New `scene3d.json` exports contain the same `viewpoint` and `source_size: {width,height}`
for the initial calibration. They are optional for older reports and have no effect on
ball/event data or analysis routing.


### Flight speed estimates (2026-09-23)

Optional `scene.speed_analysis` uses `court-camera-gravity-v1`. `status` is
`estimated` or `unavailable`; `estimates` contains bounded flight windows with
`start_frame`, `end_frame`, `start_time_s`, `end_time_s`, `time_s`, `speed_kmh`,
`observations`, `reprojection_median_px`, `camera_height_m` and
`camera_elevation_to_court_centre_deg` (depression toward the court centre,
not optical-axis pitch). Speed is the 3D velocity magnitude at window midpoint,
not racket exit speed. `pixel_sensitivity_kmh` is a linearized two-sigma pixel
noise sensitivity conditional on calibration/model; it is NOT a total accuracy
bound or radar validation. Old reports omit this field and display unavailable.

Both viewpoints use the same read-only method. Assumptions: centred principal
point, square pixels, negligible lens distortion, short gravity-only flight.
Court corners alone do not uniquely determine arbitrary camera intrinsics or
instantaneous airborne depth. Lens calibration, drag/spin and court marking
error remain systematic error sources. Reject short/gapped, ill-conditioned,
nonphysical or poorly fitting trajectories. No new neural model or GPU pass.
This report extension is not yet shipped in pinned cloud images or live workers.


Speed v2 defaults to `court-camera-drag-v2`. Windows additionally export
`drag_per_m` and `drag_sensitivity_kmh` (sensitivity to ±25% drag prior).
Set `TENNISVISION_SPEED_METHOD=gravity` to restore v1 or `off` to emit
`status=disabled`, `method=off`, `estimates=[]`. Default `drag` independently
adapts the physical-reprojection approach documented in
[open-source comparison](experiments/SPEED_OPEN_SOURCE_2026-09-23.md).
It models tennis quadratic drag but not spin or wind; accuracy remains unverified
against real speed ground truth. No additional Python package or model is needed.


Speed calibration normalizes the projective matrix by its (2,2) entry before
camera recovery. A homography and any nonzero scaled equivalent, including a
negative multiple produced by matrix inversion, must yield identical speeds.
Optional internal `diagnostics` counters record rejection reasons without changing
returned scene schema or filtering thresholds. Regression includes signed scales.

### Live speed status

Live snapshots optionally contain `speed`: `enabled`, `method`, `latest`, `recent` (last 30 accepted windows), `count`, `mean_kmh`, `max_kmh`, `metric`. Each accepted estimate additionally has `id`, `emitted_at`, `decision_t`, `delay_ms`. Times are source seconds except `emitted_at` (Unix seconds). Speed is km/h. This field is independent of landing events and their cursor. The frontend gates individual records by playback `decision_t`, expires the current value after three source seconds, and supports hiding the panel without disabling computation.
