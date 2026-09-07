"""Appearance-backed identity persistence for the two active tennis players.

The ball tracker and landing detector deliberately do not import this module.  Player
identity is presentation metadata: it consumes the person boxes already produced by the
main vision pass and labels the near/far player as identity ``A`` or ``B``.  A change of
side is accepted only after three strong, consecutive OSNet-AIN observations.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

COURT_WIDTH = 10.97
COURT_LENGTH = 23.77
NET_Y = COURT_LENGTH / 2
MODEL_NAME = "osnet_ain_x1_0"
WEIGHT_NAME = (
    "osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_"
    "b64_fb10_softmax_labsmth_flip_jitter.pth"
)
PLAYER_COLORS_RGB = {"A": (181, 66, 246), "B": (45, 212, 191)}
PLAYER_COLORS_BGR = {key: value[::-1] for key, value in PLAYER_COLORS_RGB.items()}
MAX_PLAYER_SPEED_MPS = 12.0
POSITION_REACQUIRE_SECONDS = 1.25
POSITION_SLACK_METRES = 1.5


@dataclass(frozen=True)
class IdentityResult:
    """Dense side-to-identity decisions plus auditable run metrics."""

    frames: list[dict[str, Any]]
    metrics: dict[str, Any]


def load_osnet(weights: Path, device: torch.device) -> torch.nn.Module:
    """Build the exact OSNet-AIN model validated by the repository experiment."""
    if not weights.is_file():
        raise FileNotFoundError(f"缺少球员身份模型：{weights}")
    try:
        import torchreid
    except ImportError as exc:
        raise RuntimeError("缺少 torchreid；请重新运行项目安装命令") from exc
    model = torchreid.models.build_model(
        name=MODEL_NAME, num_classes=1041, loss="softmax", pretrained=False,
    )
    torchreid.utils.load_pretrained_weights(model, str(weights))
    return model.to(device).eval()


def _box_iou(first: np.ndarray, second: np.ndarray) -> float:
    left = max(float(first[0]), float(second[0]))
    top = max(float(first[1]), float(second[1]))
    right = min(float(first[2]), float(second[2]))
    bottom = min(float(first[3]), float(second[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, float(first[2] - first[0])) * max(0.0, float(first[3] - first[1]))
    second_area = max(0.0, float(second[2] - second[0])) * max(0.0, float(second[3] - second[1]))
    return intersection / max(first_area + second_area - intersection, 1e-9)


def select_side_players(
    meta: dict[str, Any],
    previous: dict[str, dict[str, Any]] | None = None,
    elapsed_seconds: float | None = None,
) -> dict[str, dict[str, Any]]:
    """Select one active player per half using court geometry and short-term continuity.

    A weak person box may continue an established track, but a distant bystander cannot
    replace a player in one sample.  After a genuine gap the gate is released so a player
    returning from a changeover can be reacquired.
    """
    corners = np.asarray(meta.get("court_corners"), dtype=np.float32)
    boxes = np.asarray(meta.get("person_boxes", ()), dtype=np.float32).reshape(-1, 4)
    if corners.shape != (4, 2) or not len(boxes):
        return {}
    target = np.float32([
        [0, 0], [COURT_WIDTH, 0], [COURT_WIDTH, COURT_LENGTH], [0, COURT_LENGTH],
    ])
    matrix = cv2.getPerspectiveTransform(corners, target)
    feet = np.column_stack(((boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3])).astype(np.float32)
    world = cv2.perspectiveTransform(feet[None], matrix)[0]
    selected: dict[str, tuple[float, dict[str, Any]]] = {}
    for box, (court_x, court_y) in zip(boxes, world, strict=True):
        height = float(box[3] - box[1])
        if height < 30 or not (-2.5 <= court_x <= COURT_WIDTH + 2.5):
            continue
        if not (-8 <= court_y <= COURT_LENGTH + 8):
            continue
        side = "near" if court_y < NET_Y else "far"
        outside = max(-float(court_y), float(court_y) - COURT_LENGTH, 0.0)
        score = height * 0.002 - abs(float(court_x) - COURT_WIDTH / 2) * 0.05 - outside * 0.08
        previous_player = (previous or {}).get(side)
        if previous_player is not None and elapsed_seconds is not None:
            previous_world = np.asarray(previous_player["world"], dtype=np.float32)
            distance = float(np.linalg.norm(np.asarray([court_x, court_y]) - previous_world))
            if elapsed_seconds <= POSITION_REACQUIRE_SECONDS:
                reachable = MAX_PLAYER_SPEED_MPS * elapsed_seconds + POSITION_SLACK_METRES
                if distance > reachable:
                    continue
                score += 0.75 * (1.0 - distance / max(reachable, 1e-6))
                score += 0.25 * _box_iou(box, np.asarray(previous_player["box"]))
        record = {
            "box": box.copy(),
            "world": np.asarray([court_x, court_y], dtype=np.float32),
        }
        if side not in selected or score > selected[side][0]:
            selected[side] = (score, record)
    return {side: value[1] for side, value in selected.items()}


def select_side_boxes(meta: dict[str, Any]) -> dict[str, np.ndarray]:
    """Compatibility wrapper returning only the selected boxes."""
    return {side: player["box"] for side, player in select_side_players(meta).items()}


def crop_player(frame: np.ndarray, box: np.ndarray) -> np.ndarray | None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = map(float, box)
    pad_x, pad_y = (x2 - x1) * 0.08, (y2 - y1) * 0.04
    left, top = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
    right, bottom = min(width, int(x2 + pad_x)), min(height, int(y2 + pad_y))
    if right - left < 10 or bottom - top < 25:
        return None
    return frame[top:bottom, left:right].copy()


def preprocess(crops: list[np.ndarray]) -> torch.Tensor:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    tensors = []
    for crop in crops:
        rgb = cv2.cvtColor(cv2.resize(crop, (128, 256)), cv2.COLOR_BGR2RGB)
        normalized = (rgb.astype(np.float32) / 255.0 - mean) / std
        tensors.append(torch.from_numpy(normalized.transpose(2, 0, 1)))
    return torch.stack(tensors)


@torch.inference_mode()
def embed_crops(
    model: torch.nn.Module, crops: list[np.ndarray], device: torch.device,
) -> np.ndarray:
    features = model(preprocess(crops).to(device))
    return torch.nn.functional.normalize(features, dim=1).cpu().numpy()


def collect_observations(
    video: Path,
    frames_meta: list[dict[str, Any]],
    model: torch.nn.Module,
    device: torch.device,
    stride: int = 5,
    batch_size: int = 48,
) -> tuple[dict[int, dict[str, dict[str, Any]]], float]:
    """Decode only sparse sample frames and embed both active players in batches."""
    capture = cv2.VideoCapture(str(video))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    observations: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    pending_crops: list[np.ndarray] = []
    pending_keys: list[tuple[int, str]] = []
    previous_players: dict[str, dict[str, Any]] = {}
    previous_frames: dict[str, int] = {}

    def flush() -> None:
        if not pending_crops:
            return
        features = embed_crops(model, pending_crops, device)
        for (frame_index, side), feature in zip(pending_keys, features, strict=True):
            observations[frame_index][side]["feature"] = feature
        pending_crops.clear()
        pending_keys.clear()

    for frame_index, meta in enumerate(frames_meta):
        ok = capture.grab()
        if not ok:
            break
        if frame_index % max(1, stride):
            continue
        ok, frame = capture.retrieve()
        if not ok:
            continue
        elapsed = {
            side: (frame_index - previous_frames[side]) / max(fps, 1e-6)
            for side in previous_frames
        }
        selected = select_side_players(meta)
        # Apply a separate elapsed gate for each half.  This keeps the public helper simple
        # while allowing one side to survive an occlusion independently of the other.
        for side in tuple(selected):
            if side not in previous_players:
                continue
            selected_for_side = select_side_players(
                meta, {side: previous_players[side]}, elapsed.get(side),
            ).get(side)
            if selected_for_side is None:
                selected.pop(side, None)
            else:
                selected[side] = selected_for_side
        for side, player in selected.items():
            box = player["box"]
            crop = crop_player(frame, box)
            if crop is None:
                continue
            observations[frame_index][side] = {
                "box": box.copy(), "world": player["world"].copy(), "crop": crop,
            }
            pending_crops.append(crop)
            pending_keys.append((frame_index, side))
            previous_players[side] = player
            previous_frames[side] = frame_index
        if len(pending_crops) >= batch_size:
            flush()
    flush()
    capture.release()
    return dict(observations), fps


def _normalized_mean(features: list[np.ndarray]) -> np.ndarray:
    centre = np.mean(np.stack(features), axis=0)
    return centre / max(float(np.linalg.norm(centre)), 1e-9)


def enrol_identities(
    observations: dict[int, dict[str, dict[str, Any]]], fps: float, seconds: float = 20.0,
) -> dict[str, np.ndarray]:
    """Create robust A/B appearance prototypes from the clearest opening crops."""
    enrol_end = int(fps * seconds)
    candidates: dict[str, list[tuple[float, np.ndarray]]] = defaultdict(list)
    for frame_index, by_side in observations.items():
        if frame_index > enrol_end:
            continue
        for side, observation in by_side.items():
            box = observation["box"]
            quality = float((box[2] - box[0]) * (box[3] - box[1]))
            candidates[side].append((quality, observation["feature"]))
    if any(len(candidates[side]) < 6 for side in ("near", "far")):
        raise RuntimeError("开场阶段没有收集到足够的近端、远端球员清晰画面")
    prototypes = {}
    for side, identity in (("near", "A"), ("far", "B")):
        best = sorted(candidates[side], key=lambda item: item[0], reverse=True)[:24]
        prototypes[identity] = _normalized_mean([item[1] for item in best])
    return prototypes


def classify_observations(
    observations: dict[int, dict[str, dict[str, Any]]],
    prototypes: dict[str, np.ndarray],
    enrol_end: int,
    confirmations: int = 3,
    min_pair_advantage: float = 0.08,
    fps: float = 30.0,
    max_player_speed: float = MAX_PLAYER_SPEED_MPS,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    """Assign A/B jointly and debounce a physically possible changeover.

    Appearance proposes a mapping; time and court distance decide when that proposal can
    become true.  This prevents a brief bad crop from teleporting A and B across the net.
    """
    decisions: dict[int, dict[str, Any]] = {}
    current = {"near": "A", "far": "B"}
    proposed: dict[str, str] | None = None
    proposed_since: int | None = None
    proposal_anchor: dict[str, np.ndarray] = {}
    current_positions: dict[str, np.ndarray] = {}
    current_confidence = 0.0
    proposed_count = switches = pair_frames = ambiguous = 0
    motion_rejections = 0
    margins: list[float] = []
    for frame_index in sorted(observations):
        by_side = observations[frame_index]
        scores = {
            side: {
                identity: float(observation["feature"] @ prototype)
                for identity, prototype in prototypes.items()
            }
            for side, observation in by_side.items()
        }
        raw = current.copy()
        pair_advantage = 0.0
        strong = False
        if "near" in scores and "far" in scores:
            direct = scores["near"]["A"] + scores["far"]["B"]
            swapped = scores["near"]["B"] + scores["far"]["A"]
            pair_advantage = abs(direct - swapped) / 2
            raw = ({"near": "A", "far": "B"} if direct >= swapped
                   else {"near": "B", "far": "A"})
            strong = pair_advantage >= min_pair_advantage
            if frame_index > enrol_end:
                pair_frames += 1
                margins.append(pair_advantage)
                ambiguous += int(not strong)

        if strong and raw != current:
            if raw == proposed:
                proposed_count += 1
            else:
                proposed, proposed_count = raw, 1
                proposed_since = frame_index
                proposal_anchor = {
                    identity: position.copy() for identity, position in current_positions.items()
                }

            required_seconds = 0.0
            motion_available = all(
                side in by_side and "world" in by_side[side]
                and identity in proposal_anchor
                for side, identity in raw.items()
            )
            if motion_available:
                required_seconds = max(
                    float(np.linalg.norm(
                        np.asarray(by_side[side]["world"], dtype=np.float32)
                        - proposal_anchor[identity]
                    )) / max(max_player_speed, 1e-6)
                    for side, identity in raw.items()
                )
            proposal_seconds = (
                (frame_index - proposed_since) / max(fps, 1e-6)
                if proposed_since is not None else 0.0
            )
            motion_ready = not motion_available or proposal_seconds + 1e-9 >= required_seconds
            if proposed_count >= max(1, confirmations) and motion_ready:
                current = raw.copy()
                current_confidence = pair_advantage
                proposed, proposed_count, proposed_since = None, 0, None
                proposal_anchor = {}
                switches += 1
            elif proposed_count >= max(1, confirmations) and not motion_ready:
                motion_rejections += 1
        elif strong:
            current_confidence = max(current_confidence * 0.9, pair_advantage)
            proposed, proposed_count, proposed_since = None, 0, None
            proposal_anchor = {}

        for side, identity in current.items():
            observation = by_side.get(side)
            if observation is not None and "world" in observation:
                current_positions[identity] = np.asarray(observation["world"], dtype=np.float32)

        decisions[frame_index] = {
            "mapping": current.copy(), "scores": scores,
            "pair_advantage": pair_advantage,
            "mapping_confidence": current_confidence,
            "strong": strong,
        }
    metrics = {
        "pair_samples_after_enrolment": pair_frames,
        "identity_side_switches": switches,
        "ambiguous_pair_samples": ambiguous,
        "motion_rejected_switch_samples": motion_rejections,
        "mean_pair_similarity_advantage": round(float(np.mean(margins)) if margins else 0.0, 4),
        "median_pair_similarity_advantage": round(float(np.median(margins)) if margins else 0.0, 4),
    }
    return decisions, metrics


def make_dense_decisions(
    decisions: dict[int, dict[str, Any]], frame_count: int,
) -> list[dict[str, Any]]:
    """Hold the latest reliable sparse decision across every video frame."""
    latest = {
        "mapping": {"near": "A", "far": "B"}, "pair_advantage": 0.0,
        "strong": False,
    }
    dense = []
    for frame_index in range(frame_count):
        if frame_index in decisions:
            latest = decisions[frame_index]
        dense.append({
            "mapping": latest["mapping"].copy(),
            "confidence": float(latest.get(
                "mapping_confidence", latest.get("pair_advantage", 0.0),
            )),
            "observed": frame_index in decisions,
        })
    return dense


def identify_players(
    video: Path,
    frames_meta: list[dict[str, Any]],
    weights: Path,
    *,
    sample_stride: int = 5,
    enrol_seconds: float = 20.0,
    device: torch.device | None = None,
) -> IdentityResult:
    """Run the approved sparse OSNet-AIN identity logic over a completed vision pass."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_osnet(weights, device)
    observations, fps = collect_observations(
        video, frames_meta, model, device, stride=sample_stride,
    )
    try:
        prototypes = enrol_identities(observations, fps, enrol_seconds)
    except RuntimeError as exc:
        fallback = make_dense_decisions({}, len(frames_meta))
        return IdentityResult(fallback, {
            "model": MODEL_NAME, "mode": "position_fallback",
            "reason": str(exc), "sample_stride": sample_stride,
            "sample_frames": len(observations), "fps": round(fps, 4),
        })
    sparse, metrics = classify_observations(
        observations, prototypes, int(fps * enrol_seconds), fps=fps,
    )
    metrics.update({
        "model": MODEL_NAME, "mode": "appearance_reid", "sample_stride": sample_stride,
        "sample_frames": len(observations), "fps": round(fps, 4),
    })
    return IdentityResult(make_dense_decisions(sparse, len(frames_meta)), metrics)


def attach_identity_to_frames(
    frames_meta: list[dict[str, Any]], result: IdentityResult,
) -> None:
    """Attach IDs to existing player records without modifying their geometry."""
    for meta, decision in zip(frames_meta, result.frames, strict=True):
        mapping = decision["mapping"]
        meta["player_identity_by_side"] = mapping.copy()
        meta["player_identity_confidence"] = decision["confidence"]
        for player in meta.get("players_world", ()):
            player["player_id"] = mapping.get(player.get("side"))


def player_identity_at(
    frames_meta: list[dict[str, Any]], frame_index: int, side: str | None,
) -> tuple[str | None, float | None]:
    if side is None or not (0 <= frame_index < len(frames_meta)):
        return None, None
    meta = frames_meta[frame_index]
    identity = (meta.get("player_identity_by_side") or {}).get(side)
    confidence = meta.get("player_identity_confidence")
    return identity, (float(confidence) if confidence is not None else None)


def attribute_landings_to_hitters(
    events: list[dict[str, Any]],
    bounces: list[dict[str, Any]],
    frames_meta: list[dict[str, Any]],
) -> None:
    """Make each landing inherit the identity of the preceding racket strike."""
    hits_by_rally: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if event.get("kind") != "hit":
            continue
        identity, confidence = player_identity_at(
            frames_meta, int(event["frame"]), event.get("contact_side"),
        )
        event["player_id"] = identity
        event["identity_confidence"] = confidence
        if event.get("rally_id") is not None:
            hits_by_rally[int(event["rally_id"])].append(event)
    for bounce in bounces:
        preceding = [
            hit for hit in hits_by_rally.get(int(bounce.get("rally_id", -1)), ())
            if int(hit["frame"]) < int(bounce["frame"])
        ]
        owner = max(preceding, key=lambda hit: int(hit["frame"])) if preceding else None
        if owner:
            bounce["player_id"] = owner.get("player_id")
            bounce["identity_confidence"] = owner.get("identity_confidence")
            bounce["identity_source"] = "preceding_hit"
            continue
        # A clipped point may begin after the strike. Tennis still makes the hitter's
        # side knowable: a valid first landing on one half came from the opposite half.
        world = bounce.get("world")
        if world is None:
            bounce["player_id"] = None
            bounce["identity_confidence"] = None
            bounce["identity_source"] = "unresolved"
            continue
        landing_side = "near" if float(world[1]) < NET_Y else "far"
        hitter_side = "far" if landing_side == "near" else "near"
        identity, confidence = player_identity_at(
            frames_meta, int(bounce["frame"]), hitter_side,
        )
        bounce["player_id"] = identity
        bounce["identity_confidence"] = confidence
        bounce["identity_source"] = "opposite_landing_half"
