"""Optional RacketVision RacketPose inference and framework-neutral outputs.

The OpenMMLab imports are deliberately lazy.  The production environment can consume
saved racket timelines without installing the legacy RacketPose dependency stack.
"""

from __future__ import annotations

import hashlib
import math
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from netcast_tennisvision.paths import repository_path

KEYPOINT_NAMES = ("top", "bottom", "handle", "left", "right")
TENNIS_CATEGORY_ID = 2

DETECTOR_PATH = repository_path("models", "racketpose_rtmdet_m_epoch300.pth")
POSE_PATH = repository_path("models", "racketpose_rtmpose_m_epoch90.pth")
DETECTOR_SHA256 = "e6ad74371259d844b11529a64b09052edaec4277ce9ebeeca64d77b9131a19cd"
POSE_SHA256 = "faab22d4c8753759a6c6d4b48241e97b560f5a1b4374ba5cd4074932801a51e2"

_TORCH_LOAD_LOCK = threading.Lock()


class RacketPoseUnavailable(RuntimeError):
    """Raised when the optional inference backend or its assets are unavailable."""


@dataclass(frozen=True)
class RacketKeypoint:
    x: float
    y: float
    confidence: float

    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "confidence": self.confidence}


@dataclass(frozen=True)
class RacketPosePrediction:
    """One racket observation in original-frame pixel coordinates."""

    bbox_xyxy: tuple[float, float, float, float]
    bbox_confidence: float
    keypoints: dict[str, RacketKeypoint]
    source_index: int = 0

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def mean_keypoint_confidence(self) -> float:
        values = [point.confidence for point in self.keypoints.values()]
        return sum(values) / len(values) if values else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox_xyxy": list(self.bbox_xyxy),
            "bbox_confidence": self.bbox_confidence,
            "keypoints": {
                name: self.keypoints[name].to_dict()
                for name in KEYPOINT_NAMES
                if name in self.keypoints
            },
            "source_index": self.source_index,
        }

    @classmethod
    def from_upstream(cls, value: dict[str, Any], source_index: int = 0) -> RacketPosePrediction:
        """Normalize the JSON shape emitted by upstream RacketVision."""
        raw_bbox = value.get("bbox")
        if isinstance(raw_bbox, Sequence) and len(raw_bbox) == 1:
            raw_bbox = raw_bbox[0]
        if not isinstance(raw_bbox, Sequence) or len(raw_bbox) != 4:
            raise ValueError("RacketPose bbox must contain four xyxy values")

        raw_points = value.get("keypoints")
        raw_scores = value.get("keypoint_scores")
        if not isinstance(raw_points, Sequence) or len(raw_points) != len(KEYPOINT_NAMES):
            raise ValueError("RacketPose must contain the five canonical keypoints")
        if not isinstance(raw_scores, Sequence) or len(raw_scores) != len(KEYPOINT_NAMES):
            raise ValueError("RacketPose must contain five keypoint scores")

        keypoints: dict[str, RacketKeypoint] = {}
        for name, coordinates, confidence in zip(
            KEYPOINT_NAMES, raw_points, raw_scores, strict=True
        ):
            if not isinstance(coordinates, Sequence) or len(coordinates) < 2:
                raise ValueError(f"invalid RacketPose keypoint: {name}")
            keypoints[name] = RacketKeypoint(
                x=_finite_float(coordinates[0], f"{name}.x"),
                y=_finite_float(coordinates[1], f"{name}.y"),
                confidence=_probability(confidence, f"{name}.confidence"),
            )

        bbox = tuple(_finite_float(item, "bbox") for item in raw_bbox)
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            raise ValueError("RacketPose bbox must have positive area")
        return cls(
            bbox_xyxy=bbox,
            bbox_confidence=_probability(value.get("bbox_score", 0.0), "bbox_score"),
            keypoints=keypoints,
            source_index=source_index,
        )


@dataclass(frozen=True)
class RacketFrame:
    """Racket observations attached to one source-video timestamp."""

    frame_index: int
    timestamp_s: float
    rackets: tuple[RacketPosePrediction, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_s": self.timestamp_s,
            "rackets": [racket.to_dict() for racket in self.rackets],
        }


def choose_player_racket(
    rackets: Iterable[RacketPosePrediction],
    wrist_points: Sequence[tuple[float, float]] = (),
    previous_center: tuple[float, float] | None = None,
) -> RacketPosePrediction | None:
    """Associate a racket with a player using handle-to-wrist and temporal proximity.

    Distances are normalized by the candidate bbox diagonal, so this remains usable for
    both near- and far-court players.  It is an association rule, not a scoring rule.
    """
    candidates = list(rackets)
    if not candidates:
        return None

    def cost(racket: RacketPosePrediction) -> float:
        x1, y1, x2, y2 = racket.bbox_xyxy
        diagonal = max(math.hypot(x2 - x1, y2 - y1), 1.0)
        result = -(0.65 * racket.bbox_confidence + 0.35 * racket.mean_keypoint_confidence)
        handle = racket.keypoints.get("handle")
        if wrist_points and handle is not None:
            result += min(math.dist((handle.x, handle.y), wrist) for wrist in wrist_points) / diagonal
        if previous_center is not None:
            result += 0.35 * math.dist(racket.center, previous_center) / diagonal
        return result

    return min(candidates, key=cost)


class OpenMMLabRacketPose:
    """Exact RTMDet-M + RTMPose-M backend published with RacketVision.

    Install the isolated optional environment described in ``docs/RACKET_POSE.md``.
    Construction does not import OpenMMLab; :meth:`load` and :meth:`infer` do.
    """

    def __init__(
        self,
        detector_path: Path = DETECTOR_PATH,
        pose_path: Path = POSE_PATH,
        *,
        device: str = "cuda:0",
        bbox_threshold: float = 0.3,
        max_bbox_area_ratio: float = 0.5,
        max_instances: int = 4,
    ) -> None:
        self.detector_path = Path(detector_path)
        self.pose_path = Path(pose_path)
        self.device = device
        self.bbox_threshold = bbox_threshold
        self.max_bbox_area_ratio = max_bbox_area_ratio
        self.max_instances = max_instances
        self._detector: Any = None
        self._pose_model: Any = None
        self._apis: tuple[Any, Any, Any] | None = None

    @property
    def loaded(self) -> bool:
        return self._detector is not None and self._pose_model is not None

    def load(self) -> None:
        if self.loaded:
            return
        _verify_asset(self.detector_path, DETECTOR_SHA256)
        _verify_asset(self.pose_path, POSE_SHA256)
        try:
            import torch
            from mmdet.apis import inference_detector, init_detector
            from mmengine.registry import DefaultScope
            from mmpose.apis import inference_topdown
            from mmpose.apis import init_model as init_pose_model
        except ImportError as error:
            raise RacketPoseUnavailable(
                "RacketPose needs the isolated OpenMMLab environment; see docs/RACKET_POSE.md"
            ) from error

        config_dir = Path(__file__).with_name("racket_pose_configs")
        detector_config = config_dir / "rtmdet_m_racket_infer.py"
        pose_config = config_dir / "rtmpose_m_racket_infer.py"

        # PyTorch 2.6 changed torch.load's default to weights_only=True.  These exact,
        # SHA-256-verified MMEngine checkpoints contain metadata needed by init_model.
        with _TORCH_LOAD_LOCK:
            original_torch_load = torch.load

            def trusted_torch_load(*args: Any, **kwargs: Any) -> Any:
                kwargs.setdefault("weights_only", False)
                return original_torch_load(*args, **kwargs)

            torch.load = trusted_torch_load
            try:
                detector = init_detector(
                    str(detector_config), str(self.detector_path), device=self.device
                )
                pose_model = init_pose_model(
                    str(pose_config), str(self.pose_path), device=self.device
                )
            finally:
                torch.load = original_torch_load

        detector.eval()
        pose_model.eval()
        self._detector = detector
        self._pose_model = pose_model
        self._apis = (DefaultScope, inference_detector, inference_topdown)

    def infer(self, image_bgr: Any) -> tuple[RacketPosePrediction, ...]:
        """Infer all tennis rackets in one BGR frame."""
        if not self.loaded:
            self.load()
        assert self._apis is not None
        DefaultScope, inference_detector, inference_topdown = self._apis
        height, width = image_bgr.shape[:2]
        with DefaultScope.overwrite_default_scope("mmdet"):
            result = inference_detector(self._detector, image_bgr)

        instances = result.pred_instances
        bboxes = instances.bboxes.detach().cpu().numpy()
        scores = instances.scores.detach().cpu().numpy()
        labels = instances.labels.detach().cpu().numpy()
        selected: list[tuple[Any, float]] = []
        for bbox, score, label in zip(bboxes, scores, labels, strict=True):
            if int(label) != TENNIS_CATEGORY_ID or float(score) < self.bbox_threshold:
                continue
            area = max(float(bbox[2] - bbox[0]), 0.0) * max(float(bbox[3] - bbox[1]), 0.0)
            if area / max(width * height, 1) >= self.max_bbox_area_ratio:
                continue
            selected.append((bbox, float(score)))
            if len(selected) >= self.max_instances:
                break
        if not selected:
            return ()

        import numpy as np

        selected_bboxes = np.asarray([bbox for bbox, _ in selected])
        with DefaultScope.overwrite_default_scope("mmpose"):
            pose_results = inference_topdown(self._pose_model, image_bgr, selected_bboxes)

        predictions = []
        for index, (pose_result, (bbox, bbox_score)) in enumerate(
            zip(pose_results, selected, strict=True)
        ):
            predictions.append(
                RacketPosePrediction.from_upstream(
                    {
                        "bbox": [bbox.tolist()],
                        "bbox_score": bbox_score,
                        "keypoints": pose_result.pred_instances.keypoints[0].tolist(),
                        "keypoint_scores": pose_result.pred_instances.keypoint_scores[0].tolist(),
                    },
                    source_index=index,
                )
            )
        return tuple(predictions)


def _finite_float(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _probability(value: Any, label: str) -> float:
    result = _finite_float(value, label)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be between zero and one")
    return result


def _verify_asset(path: Path, expected_sha256: str) -> None:
    if not path.is_file():
        raise RacketPoseUnavailable(
            f"missing RacketPose model: {path}; run tools/install_assets.py --group racket_pose"
        )
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected_sha256:
        raise RacketPoseUnavailable(f"RacketPose model checksum mismatch: {path}")
