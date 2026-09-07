"""Court-aware match-mode inference from temporally aggregated person detections."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

COURT_WIDTH = 10.97
COURT_LENGTH = 23.77
NET_Y = COURT_LENGTH / 2

MODE_LABELS = {
    "singles": "单打模式",
    "doubles": "双打模式",
    "training": "训练模式",
}


@dataclass(frozen=True)
class PlayModeResult:
    mode: str
    near_players: int
    far_players: int
    confidence: float
    sampled_frames: int
    valid_frames: int
    pair_distribution: dict[str, int]

    @property
    def label(self) -> str:
        return MODE_LABELS[self.mode]

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "label": self.label,
            "near_players": self.near_players,
            "far_players": self.far_players,
            "confidence": round(self.confidence, 4),
            "sampled_frames": self.sampled_frames,
            "valid_frames": self.valid_frames,
            "pair_distribution": self.pair_distribution,
        }


def _frame_side_counts(meta: dict[str, Any]) -> tuple[int, int] | None:
    """Count plausible on-court people by the side containing their feet."""
    matrix = meta.get("M_inv")
    boxes = np.asarray(meta.get("person_boxes", ()), dtype=np.float32).reshape(-1, 4)
    if matrix is None or not len(boxes):
        return None
    heights = boxes[:, 3] - boxes[:, 1]
    usable = boxes[heights >= 20.0]
    if not len(usable):
        return None
    feet = np.column_stack(((usable[:, 0] + usable[:, 2]) / 2, usable[:, 3])).astype(
        np.float32
    )
    world = cv2.perspectiveTransform(feet[None], np.asarray(matrix, np.float32))[0]
    near = far = 0
    for court_x, court_y in world:
        # A modest apron admits a baseline player whose feet land just outside the paint,
        # while excluding spectators, officials and people behind adjacent courts.
        if not (-1.5 <= court_x <= COURT_WIDTH + 1.5):
            continue
        if not (-3.0 <= court_y <= COURT_LENGTH + 3.0):
            continue
        if court_y < NET_Y:
            near += 1
        else:
            far += 1
    return min(near, 3), min(far, 3)


def detect_play_mode(
    frames_meta: list[dict[str, Any]],
    *,
    sample_stride: int = 5,
    presence_quantile: float = 0.65,
) -> PlayModeResult:
    """Infer singles, doubles or training from a stable whole-video head count.

    The upper-middle quantile tolerates intermittent occlusion of a doubles partner without
    letting a rare false detection change a singles match.  Both sides must be visible in
    a frame for it to vote on the final pair.
    """
    samples = [
        counts
        for index in range(0, len(frames_meta), max(1, sample_stride))
        if (counts := _frame_side_counts(frames_meta[index])) is not None
    ]
    paired = [counts for counts in samples if counts[0] > 0 and counts[1] > 0]
    if not paired:
        return PlayModeResult("training", 0, 0, 0.0, len(samples), 0, {})

    values = np.asarray(paired, dtype=float)
    near_players, far_players = (
        int(np.quantile(values[:, side], presence_quantile, method="higher"))
        for side in (0, 1)
    )
    mode = (
        "singles" if (near_players, far_players) == (1, 1)
        else "doubles" if (near_players, far_players) == (2, 2)
        else "training"
    )
    distribution = Counter(paired)
    exact_support = distribution[(near_players, far_players)] / len(paired)
    # Confidence also records how much of the sampled clip had both sides observable.
    observability = len(paired) / max(len(samples), 1)
    confidence = float(np.sqrt(exact_support * observability))
    return PlayModeResult(
        mode, near_players, far_players, confidence,
        len(samples), len(paired),
        {f"{near}+{far}": count for (near, far), count in distribution.most_common()},
    )
