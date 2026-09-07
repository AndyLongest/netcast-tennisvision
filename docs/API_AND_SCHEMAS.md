# Local API and report contracts

The server listens on `127.0.0.1:4173`. It is a local desktop service, not an
internet-facing multi-user API.

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

### `POST /api/court-calibration`

Accepts the current request ID and four image-space corners in near-left, near-right,
far-right, far-left order. Geometry validation runs before the paused job continues.

## `scene3d.json`

Top-level fields currently include:

- `fps`, `n_frames`: native timeline;
- `frames`: per-frame compact 3D/display state;
- `bounces`: confirmed touchdown events;
- `hits`: racket-contact events;
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
| `player_id` | identity of the player whose preceding strike produced this landing |
| `identity_confidence` | OSNet pair-assignment margin; not a calibrated probability |
| `identity_source` | `preceding_hit` or the clipped-rally fallback `opposite_landing_half` |

Hit records also expose `player_id` and `identity_confidence`. Identity is attached after
tracking and touchdown decisions, so it is not an input to ball or landing accuracy.

Frame field `b` is the compact ball coordinate `[court_x, court_y, height]` or `null`.
Frame field `c` is the compact confidence/source class consumed by the viewer. Any schema
change must update `web/app.js`, the 3D exporter, frozen demo, manifest and tests together.

## Coordinate and rendering rules

- Court dimensions use ITF metres: 10.97m by 23.77m for doubles geometry.
- Homography coordinates are trustworthy for a ball only at a known ground contact.
- `decision_frame`, never the candidate frame, controls yellow highlighting.
- Out events render a red cross and never a yellow zone.
- The minimap filters by the active `rally_id` and does not draw the trajectory.
- In-court landing dots use the hitter's A/B colour; out events remain red crosses.
- Perspective correction is applied after inference and annotation rendering but before
  the minimap is composited. It cannot change tracking, contacts, landing coordinates or zones.
