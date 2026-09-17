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
  subscriber for the trusted stream name; it never receives the source video.
- `POST /api/live-lab/upload` accepts original video bytes and `X-Filename`, validates the
  native frame rate, keeps the file on the camera-simulator host, then asynchronously
  provisions L40S and returns a local session id. Once the worker reports
  `awaiting_stream`, local FFmpeg publishes the file at native speed to ZLMediaKit.
- `GET /api/live-lab/status?session_id=...&after_event=N` returns measured ingest/analysis
  clocks, queue backlog, newly confirmed online events, the trusted per-session HTTP-fMP4
  playback URL and the final offline comparison.
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
- `hits`: racket-contact events;
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
