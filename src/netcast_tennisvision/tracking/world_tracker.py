"""Persistent single-ball state for fixed-view tennis video at its native frame rate.

The detector supplies *candidates*.  This module decides whether those candidates can
belong to the one physical ball that already exists in the rally.  A detection cannot
create a ball by itself, and a missed detection does not make the ball disappear.
"""
from __future__ import annotations

from math import floor, isfinite
from typing import Any

import numpy as np

from .ballistics import (
    predict_ballistic_pixel as _ballistic_pixel_prediction,  # noqa: F401 - compatibility API
)
from .ballistics import (
    repair_occluded_flight as _repair_occluded_flight_with_ballistics,
)
from .geometry import (
    candidate_xy as _candidate_xy,  # noqa: F401 - compatibility API
)
from .geometry import (
    inside_court_projection as _inside_court_projection,
)
from .geometry import (
    inside_player_body as _inside_player_body,
)
from .geometry import (
    launches_toward_opponent as _launches_toward_opponent,
)
from .geometry import (
    near_player as _near_player,
)
from .geometry import (
    pixel_to_world_local as _pixel_to_world_local,
)
from .smoothing import (
    build_segment as _build_segment,
)
from .smoothing import (
    repair_isolated_midflight_backtracks as _repair_isolated_midflight_backtracks,
)
from .types import TrackerDiagnostics


def track_ball_persistent(
    frames_meta: list[dict[str, Any]],
    *,
    fps: float,
    spatial: float,
    speed_scale: float,
    frame_size: tuple[int, int],
    chi2_gate: float = 9.21,
    hard_cap: float | None = None,
    confidence_bonus: float | None = None,
    min_track_span: float | None = None,
    search_radii: tuple[float, ...] | None = None,
    # Candidate generators already apply their detector-specific confidence
    # floors.  Re-applying a generic 0.20 floor here discards borderline but
    # temporally coherent observations (notably tiny far-court balls).
    search_confidences: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0),
    search_score_slack: float = 0.0,
    speed_radius_gain: float = 0.65,
) -> tuple[list[dict[str, Any]], TrackerDiagnostics]:
    """Track exactly one persistent physical ball through a native-rate clip.

    Birth requires three mutually reachable detections.  During short occlusions the state
    is predicted with widening uncertainty.  Re-acquisition must fall inside both the
    uncertainty gate and a hard physical displacement cap.  Compatible fragments are then
    joined across occlusions before the backward smoother sees them.
    """
    if not isfinite(fps) or fps <= 0:
        raise ValueError(f"persistent tracker requires a positive native fps, got {fps!r}")

    width, height = frame_size
    transition = np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float
    )
    observation = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
    process_noise = np.diag([0.7, 0.7, 18.0, 18.0]) * (speed_scale ** 2)
    measurement_noise = np.eye(2) * (3.5 * spatial) ** 2
    initial_covariance = np.diag([4.0, 4.0, 144.0, 144.0]) * (spatial ** 2)
    hard_cap = float(hard_cap if hard_cap is not None else 220.0 * speed_scale)
    confidence_bonus = float(confidence_bonus if confidence_bonus is not None else 18.0 * speed_scale)
    min_track_span = float(min_track_span if min_track_span is not None else 5.0 * spatial)
    search_radii = search_radii or tuple(
        min(hard_cap, radius * spatial) for radius in (18.0, 36.0, 72.0, 144.0)
    ) + (4.0 * float(np.hypot(width, height)),)
    search_radii = tuple(sorted(set(float(radius) for radius in search_radii)))
    if len(search_confidences) < len(search_radii):
        search_confidences = search_confidences + (search_confidences[-1],) * (
            len(search_radii) - len(search_confidences)
        )
    # Keep the online hypothesis conservative.  Longer gaps are handled below by an
    # offline, future-confirmed fragment join; making the live gate stay wide for too long
    # lets an unrelated bright object hijack the only-ball state.
    max_occlusion = max(8, int(round(0.80 * fps)))
    max_bridge_occlusion = max(max_occlusion, int(round(1.20 * fps)))
    birth_window = max(4, int(round(0.17 * fps)))
    birth_hits = 3
    net_grace = max(4, int(round(0.20 * fps)))
    net_band = max(10.0 * spatial, 0.025 * height)
    exit_margin = max(12.0 * spatial, 0.025 * min(width, height))

    raw_segments: list[dict[str, Any]] = []
    active: dict[str, Any] | None = None
    hypotheses: list[dict[str, Any]] = []
    next_track_id = 0
    confirmed_births = rejected_singletons = occluded_frames = rejected_teleports = 0
    adaptive_search_recoveries = net_terminations = frame_exit_terminations = 0
    bounce_velocity_resets = player_hit_velocity_resets = curved_flight_rejections = 0
    midflight_backtrack_repairs = 0
    ballistic_predictions = 0
    search_radius_used: dict[int, float] = {}
    search_speed_used: dict[int, float] = {}
    search_centre_used: dict[int, tuple[float, float]] = {}
    search_axes_used: dict[int, tuple[float, float]] = {}
    motion_mode_used: dict[int, str] = {}

    def finish_active(reason: str) -> None:
        nonlocal active
        if active is None:
            return
        # Keep predicted frames inside an occlusion, but never let an unobserved tail turn
        # into a ghost ball after the rally has actually left the picture.
        end = active["last_seen"]
        if end >= active["start"]:
            raw_segments.append(_build_segment(
                active["start"], end, active["observations"], transition, observation,
                process_noise, measurement_noise, initial_covariance,
                active["track_id"], reason,
            ))
        active = None

    for frame, meta in enumerate(frames_meta):
        candidates = [c for c in meta.get("candidates", ())
                      if -0.08 * width <= c[0] <= 1.08 * width
                      and -0.08 * height <= c[1] <= 1.08 * height]
        if not meta.get("is_court", False):
            finish_active("camera_cut")
            rejected_singletons += len(hypotheses)
            hypotheses = []
            continue

        if active is not None:
            previous_state = active["state"].copy()
            previous_position = previous_state[:2].copy()
            predicted = transition @ active["state"]
            predicted_cov = transition @ active["covariance"] @ transition.T + process_noise
            innovation_cov = observation @ predicted_cov @ observation.T + measurement_noise
            inv_innovation = np.linalg.pinv(innovation_cov)

            # Keep a short shadow hypothesis around a player even while the old track is
            # still alive.  A slice or serve can otherwise leave the correct candidates
            # stranded because the incoming Kalman state keeps winning.  Only three
            # consecutive, geometrically coherent points may switch the live track.
            if _near_player(meta, previous_position, spatial):
                active["launch_armed_until"] = frame + 7
            shadows: list[dict[str, Any]] = []
            for candidate in candidates:
                point = np.asarray(candidate[:2], dtype=float)
                conf = float(candidate[2])
                if active.get("launch_armed_until", -1) >= frame and _near_player(meta, point, spatial):
                    shadows.append({"obs": [(frame, point, conf)], "score": conf})
                elif active.get("launch_armed_until", -1) >= frame:
                    # The first clean detection after a hard near-court stroke can already
                    # be well clear of the body.  Seed the outgoing hypothesis from an
                    # earlier racket-adjacent observation instead of requiring that clean
                    # point itself to remain inside the player envelope.
                    for anchor_frame, anchor_value in sorted(
                            active["observations"].items(), reverse=True):
                        gap_from_hit = frame - anchor_frame
                        if gap_from_hit > 12:
                            break
                        if gap_from_hit < 2:
                            continue
                        anchor = np.asarray(anchor_value[:2], dtype=float)
                        if not _near_player(frames_meta[anchor_frame], anchor, spatial):
                            continue
                        launch_speed = float(np.linalg.norm(point - anchor) / gap_from_hit)
                        if launch_speed > 30.0 * spatial:
                            continue
                        seed = [
                            (anchor_frame, anchor, float(anchor_value[2])),
                            (frame, point, conf),
                        ]
                        # With only two samples use the same net-side test directly; the
                        # three-sample certification below still decides whether to switch.
                        net_y = frames_meta[anchor_frame].get("net_y_px")
                        if net_y is None:
                            continue
                        toward = -1.0 if anchor[1] > float(net_y) else 1.0
                        if (point[1] - anchor[1]) * toward <= 2.0 * spatial:
                            continue
                        # A geometrically valid away-from-player seed is rarer and more
                        # informative than another high-scoring blob on the player's body;
                        # keep it in the bounded hypothesis beam even at modest detector
                        # confidence.
                        shadows.append({"obs": seed,
                                        "score": float(anchor_value[2]) + conf + 1.0})
                        break
                for hypothesis in active.get("launch_hypotheses", []):
                    gap = frame - hypothesis["obs"][-1][0]
                    if not 1 <= gap <= 2:
                        continue
                    obs_h = hypothesis["obs"]
                    velocity_h = ((obs_h[-1][1] - obs_h[-2][1])
                                  / max(1, obs_h[-1][0] - obs_h[-2][0])
                                  if len(obs_h) >= 2 else np.zeros(2))
                    expected = obs_h[-1][1] + velocity_h * gap
                    error = float(np.linalg.norm(point - expected))
                    if error <= 12.0 * spatial:
                        shadows.append({"obs": obs_h + [(frame, point, conf)],
                                        "score": hypothesis["score"] + conf
                                                 - error / max(hard_cap, 1.0)})
            shadows.sort(key=lambda h: (len(h["obs"]), h["score"]), reverse=True)
            active["launch_hypotheses"] = shadows[:10]
            launch = next((h for h in shadows if len(h["obs"]) >= 3
                           and _launches_toward_opponent(frames_meta, h["obs"], spatial)), None)
            if launch is not None:
                launch_obs = launch["obs"][-3:]
                for f_launch, p_launch, c_launch in launch_obs:
                    active["observations"][f_launch] = (
                        float(p_launch[0]), float(p_launch[1]), float(c_launch), "player_hit"
                    )
                    motion_mode_used[f_launch] = "player_hit"
                launch_frames = np.asarray([o[0] for o in launch_obs], dtype=float)
                launch_points = np.vstack([o[1] for o in launch_obs])
                design = np.column_stack([launch_frames, np.ones_like(launch_frames)])
                velocity_fit, intercept_fit = np.linalg.lstsq(
                    design, launch_points, rcond=None
                )[0]
                speed_fit = float(np.linalg.norm(velocity_fit))
                max_event_speed = 22.5 * spatial  # 45 px/frame on the 720p reference
                if speed_fit > max_event_speed:
                    velocity_fit *= max_event_speed / speed_fit
                f1, p1, _ = launch_obs[-1]
                active["state"] = np.array([
                    p1[0], p1[1], *velocity_fit
                ], dtype=float)
                active["covariance"] = initial_covariance.copy()
                active["last_seen"] = frame
                active["coast"] = 0
                active["last_motion_event"] = frame
                active["launch_hypotheses"] = []
                active["launch_armed_until"] = -1
                player_hit_velocity_resets += 1
                continue

            # Estimate image velocity from the last two *real* observations.  The Kalman
            # velocity remains the fallback during/just after an occlusion, but a pair of
            # detections is more directly tied to what the camera actually saw.
            recent = sorted(active["observations"].items())[-2:]
            if len(recent) == 2:
                (f0, p0), (f1, p1) = recent
                observed_velocity = (
                    np.asarray(p1[:2], dtype=float) - np.asarray(p0[:2], dtype=float)
                ) / max(1, f1 - f0)
                estimated_velocity = 0.75 * observed_velocity + 0.25 * previous_state[2:]
            else:
                estimated_velocity = previous_state[2:].copy()
            estimated_speed = float(np.linalg.norm(estimated_velocity))
            flight_centre = previous_position + estimated_velocity

            # Convert image velocity to a local ground-plane scale.  A pixel near the far
            # baseline represents much more court than a pixel near the camera, so the
            # admissible region is a perspective ellipse rather than an image-space circle.
            local = _pixel_to_world_local(meta, flight_centre)
            if local is not None:
                _, metres_x, metres_y = local
                world_speed = float(np.hypot(
                    estimated_velocity[0] * metres_x,
                    estimated_velocity[1] * metres_y,
                ))
                physical_base = max(0.42, 0.16 + speed_radius_gain * min(world_speed, 3.0))
                base_axes = np.array([
                    np.clip(physical_base / metres_x, 18.0 * spatial, hard_cap),
                    np.clip(physical_base / metres_y, 12.0 * spatial, hard_cap),
                ])
            else:
                world_speed = None
                adaptive_base = min(
                    hard_cap,
                    max(search_radii[0], 8.0 * spatial + speed_radius_gain * estimated_speed),
                )
                base_axes = np.array([adaptive_base, adaptive_base])

            player_contact = _near_player(meta, previous_position, spatial) or _near_player(
                meta, flight_centre, spatial
            )
            player_contact = player_contact and frame - active.get("last_motion_event", -999) > 6
            bounce_contact = (
                not player_contact
                and estimated_velocity[1] > 2.0 * spatial
                and _inside_court_projection(meta, previous_position)
            )
            if player_contact:
                # Keep both hypotheses until a candidate supplies evidence of reversal.
                # Merely entering a person's expanded box is not itself a racket hit.
                search_centres = [
                    (flight_centre, "flight"),
                    (previous_position.copy(), "player_hit"),
                ]
            elif bounce_contact:
                rebound_velocity = np.array([
                    estimated_velocity[0], -0.62 * estimated_velocity[1]
                ])
                search_centres = [
                    (previous_position + rebound_velocity, "bounce"),
                    (flight_centre, "flight"),
                ]
            else:
                search_centres = [(flight_centre, "flight")]

            dynamic_axes = tuple(
                np.minimum(hard_cap, base_axes * (2.0 ** tier))
                for tier, _radius in enumerate(search_radii[:-1])
            ) + (np.array([search_radii[-1], search_radii[-1]]),)
            search_speed_used[frame] = estimated_speed
            search_centre_used[frame] = tuple(map(float, search_centres[0][0]))

            reachable: list[tuple[float, list, str, tuple[float, ...], float, bool]] = []
            for candidate in candidates:
                point = np.asarray(candidate[:2], dtype=float)
                delta = point - observation @ predicted
                distance = float(np.linalg.norm(delta))
                centre_distances = [
                    (np.abs(point - centre), mode) for centre, mode in search_centres
                ]
                mahal = float(delta @ inv_innovation @ delta)
                baseline_compatible = mahal <= chi2_gate and distance <= hard_cap
                event_compatible = False
                if player_contact or bounce_contact:
                    event_axes = dynamic_axes[min(2, len(dynamic_axes) - 2)]
                    event_compatible = any(
                        mode != "flight" and float(np.sum((delta_centre / event_axes) ** 2)) <= 1.0
                        for delta_centre, mode in centre_distances
                    )
                if not baseline_compatible and not event_compatible:
                    rejected_teleports += 1
                    continue
                # A ball may reverse at a racket or bounce, but its image speed cannot
                # explode from one frame to the next.  Keep generous impact headroom and
                # reject only physically extreme changes.
                observation_gap = max(1, frame - active["last_seen"])
                candidate_delta = (
                    point - np.asarray(active["observations"][active["last_seen"]][:2])
                ) / observation_gap
                candidate_speed = float(np.linalg.norm(candidate_delta))
                if not baseline_compatible and event_compatible and candidate_speed > 22.5 * spatial:
                    rejected_teleports += 1
                    continue
                if local is not None and world_speed is not None:
                    candidate_metric_speed = float(np.hypot(
                        candidate_delta[0] * metres_x, candidate_delta[1] * metres_y
                    ))
                    speed_change_limit = max(4.0, 4.0 * world_speed + 1.0)
                else:
                    candidate_metric_speed = candidate_speed
                    world_speed = estimated_speed
                    speed_change_limit = max(
                        36.0 * spatial, 1.75 * estimated_speed + 12.0 * spatial
                    )
                if (not baseline_compatible and len(recent) == 2
                        and abs(candidate_metric_speed - world_speed) > speed_change_limit):
                    rejected_teleports += 1
                    continue
                direction_cosine = 1.0
                if len(recent) == 2:
                    previous_norm = max(1e-6, estimated_speed * candidate_speed)
                    direction_cosine = float(estimated_velocity @ candidate_delta / previous_norm)
                    # A large Kalman covariance after several missed frames is only a
                    # statement of uncertainty; it is not permission for an ordinary
                    # flight candidate to turn sharply.  Reclassify such a point as an
                    # impact hypothesis (when geometrically possible), otherwise reject
                    # it.  This prevents one late false candidate from pulling the RTS
                    # path far ahead of the visible ball.
                    if (baseline_compatible
                            and observation_gap >= 6
                            and candidate_speed > max(12.0 * spatial, 1.50 * estimated_speed)
                            and _near_player(meta, point, spatial)):
                        baseline_compatible = False
                        if not event_compatible:
                            curved_flight_rejections += 1
                            rejected_teleports += 1
                            continue
                    if (baseline_compatible and observation_gap >= 6
                            and _inside_player_body(meta, point, spatial)):
                        baseline_compatible = False
                        if not event_compatible:
                            rejected_teleports += 1
                            continue
                    if (not baseline_compatible and not player_contact and not bounce_contact
                            and direction_cosine < -0.80
                            and distance > 2.5 * float(max(base_axes))):
                        curved_flight_rejections += 1
                        rejected_teleports += 1
                        continue
                score = (mahal + 0.015 * distance
                         - confidence_bonus * float(candidate[2]) / max(hard_cap, 1.0))
                reachable.append((score, centre_distances, "flight", tuple(candidate),
                                  direction_cosine, baseline_compatible))

            # Search locally first.  Only if nothing credible exists do we expand the
            # radius; a far-away candidate must also clear a higher visual-confidence bar.
            best: tuple[float, tuple[float, ...], int, float, str, np.ndarray] | None = None
            baseline_exists = any(item[5] for item in reachable)
            eligible = [item for item in reachable if item[5] == baseline_exists]
            global_best_score = min((item[0] for item in eligible), default=float("inf"))
            for tier, (axes, min_confidence) in enumerate(
                zip(dynamic_axes, search_confidences, strict=False)
            ):
                in_ring = []
                for item in eligible:
                    if float(item[3][2]) < min_confidence or item[0] > global_best_score + search_score_slack:
                        continue
                    matches = [
                        (mode, delta) for delta, mode in item[1]
                        if float(np.sum((delta / axes) ** 2)) <= 1.0
                        and ((baseline_exists and mode == "flight")
                             or (not baseline_exists and mode == "player_hit"
                                 and item[4] < -0.45 and float(item[3][2]) >= 0.30)
                             or (not baseline_exists and mode == "bounce" and item[4] < 0.30))
                    ]
                    if matches:
                        mode, _ = min(
                            matches,
                            key=lambda pair: float(np.sum((pair[1] / axes) ** 2)),
                        )
                        in_ring.append((item[0], item[3], mode))
                if not in_ring:
                    continue
                score, candidate, mode = min(in_ring, key=lambda item: item[0])
                best = (score, candidate, tier, float(max(axes)), mode, axes)
                break
            if best is not None:
                chosen, tier, radius, motion_mode, axes = best[1], best[2], best[3], best[4], best[5]
                z = np.asarray(chosen[:2], dtype=float)
                gain = predicted_cov @ observation.T @ inv_innovation
                active["state"] = predicted + gain @ (z - observation @ predicted)
                active["covariance"] = (np.eye(4) - gain @ observation) @ predicted_cov
                if motion_mode in {"bounce", "player_hit"}:
                    last_point = np.asarray(active["observations"][active["last_seen"]][:2])
                    gap = max(1, frame - active["last_seen"])
                    active["state"][:2] = z
                    reset_velocity = (z - last_point) / gap
                    reset_speed = float(np.linalg.norm(reset_velocity))
                    max_reset_speed = 22.5 * spatial
                    if reset_speed > max_reset_speed:
                        reset_velocity *= max_reset_speed / reset_speed
                    active["state"][2:] = reset_velocity
                    active["covariance"][2:, 2:] = initial_covariance[2:, 2:]
                    active["last_motion_event"] = frame
                    if motion_mode == "bounce":
                        bounce_velocity_resets += 1
                    else:
                        player_hit_velocity_resets += 1
                active["observations"][frame] = (
                    float(z[0]), float(z[1]), float(chosen[2]), motion_mode
                )
                if len(chosen) >= 5 and float(chosen[3]) > 0 and float(chosen[4]) > 0:
                    active["last_area"] = float(chosen[3]) * float(chosen[4])
                active["last_seen"] = frame
                active["coast"] = 0
                active["net_pending"] = 0
                active["pending_terminal"] = None
                search_radius_used[frame] = radius
                search_axes_used[frame] = (float(axes[0]), float(axes[1]))
                motion_mode_used[frame] = motion_mode
                if tier > 0:
                    adaptive_search_recoveries += 1
            else:
                active["state"], active["covariance"] = predicted, predicted_cov
                active["coast"] += 1
                occluded_frames += 1

                # A disappearance immediately after a real observation at the image edge
                # is a physically meaningful exit, not an ordinary detector miss.
                vx, vy = float(previous_state[2]), float(previous_state[3])
                px_, py_ = map(float, previous_position)
                exited = active["last_seen"] == frame - 1 and (
                    (px_ <= exit_margin and vx < 0)
                    or (px_ >= width - exit_margin and vx > 0)
                    or (py_ <= exit_margin and vy < 0)
                    or (py_ >= height - exit_margin and vy > 0)
                )
                if exited:
                    active["pending_terminal"] = "out_of_frame"

                # The projected net line varies slightly with camera motion.  Enter a
                # short pending state when the flight reaches that band; only terminate if
                # no visual observation reappears during the confirmation window.
                net_y = meta.get("net_y_px")
                if net_y is not None and active.get("pending_terminal") != "out_of_frame":
                    crossed = (previous_position[1] - net_y) * (predicted[1] - net_y) <= 0
                    near_net = abs(previous_position[1] - net_y) <= net_band
                    if active.get("net_pending", 0) or crossed or near_net:
                        active["net_pending"] = active.get("net_pending", 0) + 1
                    if active.get("net_pending", 0) >= net_grace:
                        active["pending_terminal"] = "net_hit"
                if active["coast"] > max_occlusion:
                    reason = active.get("pending_terminal") or "uncertainty_exhausted"
                    if reason == "net_hit":
                        net_terminations += 1
                    elif reason == "out_of_frame":
                        frame_exit_terminations += 1
                    finish_active(reason)
                    hypotheses = []
            continue

        # No accepted world state exists yet.  Candidate detections are merely hypotheses;
        # none becomes a ball until a short, physically reachable sequence confirms it.
        extended: list[dict[str, Any]] = []
        for candidate in candidates:
            point = np.asarray(candidate[:2], dtype=float)
            conf = float(candidate[2])
            extended.append({"obs": [(frame, point, conf)], "score": conf})
            for hypothesis in hypotheses:
                gap = frame - hypothesis["obs"][-1][0]
                if not 1 <= gap <= 2:
                    continue
                observations_h = hypothesis["obs"]
                if len(observations_h) >= 2:
                    f0, p0, _ = observations_h[-2]
                    f1, p1, _ = observations_h[-1]
                    predicted = p1 + (p1 - p0) * gap / max(1, f1 - f0)
                else:
                    predicted = observations_h[-1][1]
                distance = float(np.linalg.norm(point - predicted))
                if distance <= hard_cap * gap:
                    extended.append({"obs": observations_h + [(frame, point, conf)],
                                     "score": hypothesis["score"] + conf
                                              - distance / max(hard_cap, 1.0)})
                else:
                    rejected_teleports += 1
        hypotheses = [h for h in extended if frame - h["obs"][0][0] < birth_window]
        hypotheses.sort(key=lambda h: (len(h["obs"]), h["score"]), reverse=True)
        hypotheses = hypotheses[:12]
        confirmed = next((h for h in hypotheses if len(h["obs"]) >= birth_hits and
                          np.linalg.norm(h["obs"][-1][1] - h["obs"][0][1]) >= min_track_span), None)
        if confirmed is None:
            continue
        obs = {f: (float(p[0]), float(p[1]), float(conf)) for f, p, conf in confirmed["obs"]}
        start = min(obs)
        seed = _build_segment(start, frame, obs, transition, observation, process_noise,
                              measurement_noise, initial_covariance, next_track_id, "active")
        active = {
            "start": start, "last_seen": frame, "observations": obs,
            "state": seed["x_post"][-1].copy(), "covariance": seed["P_post"][-1].copy(),
            "coast": 0, "track_id": next_track_id,
            "last_area": 0.0,
            "net_pending": 0,
            "pending_terminal": None,
            "last_motion_event": -999,
            "launch_hypotheses": [],
            "launch_armed_until": -1,
        }
        next_track_id += 1
        confirmed_births += 1
        rejected_singletons += max(0, len(hypotheses) - 1)
        hypotheses = []

    finish_active("end_of_clip")
    rejected_singletons += len(hypotheses)

    # Join fragments when the old state can physically reach the new observations through
    # the blind interval.  This is an offline clip, so future evidence is allowed to prove
    # that the ball continued to exist behind a player or the net.
    merged: list[dict[str, Any]] = []
    merged_occlusions = 0
    for segment in raw_segments:
        if not merged:
            merged.append(segment)
            continue
        previous = merged[-1]
        gap = segment["frames"][0] - previous["frames"][-1] - 1
        state = previous["x_post"][-1].copy()
        covariance = previous["P_post"][-1].copy()
        for _ in range(max(0, gap + 1)):
            state = transition @ state
            covariance = transition @ covariance @ transition.T + process_noise
        first_measurement = next((m for m in segment["meas"] if m is not None), None)
        compatible = False
        if 0 <= gap <= max_bridge_occlusion and first_measurement is not None:
            delta = np.asarray(first_measurement) - observation @ state
            innovation_cov = observation @ covariance @ observation.T + measurement_noise
            compatible = (float(delta @ np.linalg.pinv(innovation_cov) @ delta) <= chi2_gate * 1.5
                          and float(np.linalg.norm(delta)) <= hard_cap * max(1, gap + 1))
        if not compatible:
            merged.append(segment)
            continue
        observations = {
            f: (*m, float(c)) for f, m, c in zip(
                previous["frames"], previous["meas"], previous["confidence"], strict=False
            ) if m is not None
        }
        observations.update({
            f: (*m, float(c)) for f, m, c in zip(
                segment["frames"], segment["meas"], segment["confidence"], strict=False
            ) if m is not None
        })
        merged[-1] = _build_segment(
            previous["frames"][0], segment["frames"][-1], observations,
            transition, observation, process_noise, measurement_noise, initial_covariance,
            previous["track_id"], segment["termination"],
        )
        merged_occlusions += 1

    for segment in merged:
        ballistic_predictions += _repair_occluded_flight_with_ballistics(
            segment, frames_meta, fps, spatial, frame_size
        )
        midflight_backtrack_repairs += _repair_isolated_midflight_backtracks(
            segment, frames_meta, spatial
        )

    for meta in frames_meta:
        meta["ball_px"] = None
        meta["ball_px_raw"] = None
        meta["ball_seen"] = False
        meta["ball_state"] = "unacquired"
        meta["ball_track_id"] = None
        meta["ball_confidence"] = 0.0
        meta["ball_uncertainty_px"] = None
        meta["ball_search_radius_px"] = None
        meta["ball_estimated_speed_px_frame"] = None
        meta["ball_search_centre_px"] = None
        meta["ball_search_axes_px"] = None
        meta["ball_motion_mode"] = None
        meta["ball_terminal_reason"] = None
    for segment in merged:
        for i, (frame, state) in enumerate(zip(segment["frames"], segment["smooth"], strict=False)):
            meta = frames_meta[frame]
            meta["ball_px"] = (float(state[0]), float(state[1]))
            raw = segment["meas"][i]
            meta["ball_px_raw"] = (tuple(map(float, raw)) if raw is not None else None)
            meta["ball_seen"] = bool(segment["seen"][i])
            meta["ball_state"] = "observed" if segment["seen"][i] else "occluded_predicted"
            meta["ball_track_id"] = int(segment["track_id"])
            meta["ball_confidence"] = float(segment["confidence"][i])
            meta["ball_uncertainty_px"] = float(np.sqrt(max(0.0, np.trace(segment["P_post"][i][:2, :2]))))
            meta["ball_search_radius_px"] = search_radius_used.get(frame)
            reported_speed = search_speed_used.get(frame)
            if reported_speed is not None and not segment["seen"][i]:
                reported_speed = min(float(reported_speed), 18.0 * spatial)
            meta["ball_estimated_speed_px_frame"] = reported_speed
            meta["ball_search_centre_px"] = search_centre_used.get(frame)
            meta["ball_search_axes_px"] = search_axes_used.get(frame)
            meta["ball_motion_mode"] = motion_mode_used.get(frame)
            if i == len(segment["frames"]) - 1 and segment["termination"] in {
                "net_hit", "out_of_frame"
            }:
                meta["ball_terminal_reason"] = segment["termination"]

    return merged, TrackerDiagnostics(
        confirmed_births=confirmed_births,
        rejected_singletons=rejected_singletons,
        occluded_frames=occluded_frames,
        rejected_teleports=rejected_teleports,
        merged_occlusions=merged_occlusions,
        adaptive_search_recoveries=adaptive_search_recoveries,
        net_terminations=net_terminations,
        frame_exit_terminations=frame_exit_terminations,
        bounce_velocity_resets=bounce_velocity_resets,
        player_hit_velocity_resets=player_hit_velocity_resets,
        curved_flight_rejections=curved_flight_rejections,
        midflight_backtrack_repairs=midflight_backtrack_repairs,
        ballistic_predictions=ballistic_predictions,
    )


def refine_touchdown_subframe(
    frame: int,
    frames_meta: list[dict[str, Any]],
    *,
    radius: int = 4,
) -> dict[str, Any]:
    """Estimate a touchdown between 30 fps samples from the two local flight branches."""
    before: list[tuple[float, np.ndarray]] = []
    after: list[tuple[float, np.ndarray]] = []
    lo, hi = max(0, frame - radius), min(len(frames_meta), frame + radius + 1)
    for index in range(lo, hi):
        meta = frames_meta[index]
        point = meta.get("ball_px")
        if point is None or not meta.get("ball_seen", False):
            continue
        target = before if index <= frame else after
        target.append((float(index), np.asarray(point, dtype=float)))
    fallback = frames_meta[frame].get("ball_px")
    if len(before) < 2 or len(after) < 2 or fallback is None:
        return {"frame_f": float(frame), "px": fallback, "confidence": "frame", "support": len(before) + len(after)}

    def line(samples: list[tuple[float, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, float]:
        t = np.asarray([sample[0] for sample in samples], dtype=float)
        p = np.vstack([sample[1] for sample in samples])
        design = np.column_stack([t, np.ones_like(t)])
        coeff, _, _, _ = np.linalg.lstsq(design, p, rcond=None)
        residual = float(np.sqrt(np.mean((design @ coeff - p) ** 2)))
        return coeff[0], coeff[1], residual

    velocity_before, intercept_before, residual_before = line(before)
    velocity_after, intercept_after, residual_after = line(after)
    dv = velocity_before - velocity_after
    di = intercept_before - intercept_after
    denom = float(dv @ dv)
    if denom < 1e-6:
        return {"frame_f": float(frame), "px": fallback, "confidence": "frame", "support": len(before) + len(after)}
    frame_f = float(np.clip(-float(dv @ di) / denom, frame - 0.95, frame + 0.95))
    p_before = velocity_before * frame_f + intercept_before
    p_after = velocity_after * frame_f + intercept_after
    point = (p_before + p_after) / 2.0
    residual = residual_before + residual_after + float(np.linalg.norm(p_before - p_after))
    confidence = "subframe" if residual <= 8.0 else "frame"
    if confidence == "frame":
        frame_f, point = float(frame), np.asarray(fallback, dtype=float)
    return {
        "frame_f": frame_f,
        "px": (float(point[0]), float(point[1])),
        "confidence": confidence,
        "support": len(before) + len(after),
        "residual_px": residual,
        "decision_frame": floor(frame_f) + 1,
    }
