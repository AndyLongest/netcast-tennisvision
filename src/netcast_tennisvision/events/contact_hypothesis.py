"""Evidence fusion for distinguishing racket contact from ground contact.

This module consumes an already accepted ball trajectory.  It never creates, deletes,
or moves a ball observation.  Predictions and tennis-order rules may support a decision,
but only measured motion plus scene geometry can change an existing contact label.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp
from typing import Any, Literal

ContactKind = Literal["hit", "bounce", "flight", "ambiguous"]


def _unit(value: float | None) -> float:
    return max(0.0, min(1.0, float(value or 0.0)))


def _near_score(gap_in_player_heights: float | None, *, close: float, far: float) -> float:
    if gap_in_player_heights is None:
        return 0.0
    gap = float(gap_in_player_heights)
    if gap <= close:
        return 1.0
    if gap >= far:
        return 0.0
    return (far - gap) / (far - close)


@dataclass(frozen=True)
class ContactDecision:
    kind: ContactKind
    hit_score: float
    bounce_score: float
    margin: float
    confidence: float
    evidence: tuple[str, ...]
    changed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify_contact_hypotheses(
    *,
    existing_kind: str | None,
    racket_gap: float | None,
    player_gap: float | None,
    horizontal_reversal: bool,
    ground_turn: bool,
    ground_arc: float = 0.0,
    ground_impulse: float = 0.0,
    sequence_probability: float | None = None,
    subframe_support: float = 0.0,
    audio_support: bool = False,
    departing_opposite_half: bool = False,
) -> ContactDecision:
    """Compare racket-hit and touchdown explanations for one trajectory impulse.

    Distances are normalized by the relevant player's image height, so one gate works
    at both ends of a perspective court.  A close racket is strong hit evidence; merely
    overlapping a player's body is deliberately weak.  Conversely, a ground-like turn
    needs an arc/impulse/temporal companion before it can decisively beat racket evidence.
    """

    racket = _near_score(racket_gap, close=0.42, far=1.35)
    player = _near_score(player_gap, close=0.28, far=0.95)
    arc = _unit(ground_arc)
    impulse = _unit(ground_impulse)
    temporal = _unit((sequence_probability or 0.0) / 0.40)
    subframe = _unit(subframe_support)

    hit_score = (
        0.46 * racket
        + 0.13 * player
        + 0.23 * float(horizontal_reversal)
        + 0.08 * float(audio_support)
        + 0.10 * float(departing_opposite_half)
    )
    bounce_score = (
        0.20 * float(ground_turn)
        + 0.28 * arc
        + 0.24 * impulse
        + 0.16 * temporal
        + 0.12 * subframe
    )

    # Evidence competes.  This is the key difference from stacking independent rules:
    # a strong racket explanation actively weakens a ground claim and vice versa.
    hit_score *= 1.0 - 0.42 * max(arc, impulse)
    bounce_score *= 1.0 - 0.48 * racket
    hit_score, bounce_score = _unit(hit_score), _unit(bounce_score)
    margin = hit_score - bounce_score

    evidence: list[str] = []
    if racket >= 0.65:
        evidence.append("racket-proximity")
    if player >= 0.65:
        evidence.append("player-proximity")
    if horizontal_reversal:
        evidence.append("horizontal-reversal")
    if ground_turn:
        evidence.append("ground-direction-change")
    if arc >= 0.55:
        evidence.append("two-sided-ground-arc")
    if impulse >= 0.45:
        evidence.append("measured-trajectory-impulse")
    if temporal >= 0.45:
        evidence.append("temporal-bounce-shape")
    if audio_support:
        evidence.append("audio-timing")

    # Require a decisive, scene-supported margin to relabel an existing event.  Ambiguous
    # evidence preserves the upstream label and remains visible in diagnostics.
    # A ball within one player-height of the reconstructed racket point is the production
    # baseline's proven contact gate. Preserve it unless a two-sided ground arc provides
    # direct contradictory evidence; the probabilistic margin handles the softer cases.
    racket_geometry_hit = (
        racket_gap is not None and float(racket_gap) < 1.0 and arc < 0.45
    )
    decisive_hit = racket_geometry_hit or (
        margin >= 0.18 and (racket >= 0.55 or departing_opposite_half)
    )
    decisive_bounce = margin <= -0.18 and (arc >= 0.45 or impulse >= 0.50)
    if decisive_hit:
        kind: ContactKind = "hit"
    elif decisive_bounce:
        kind = "bounce"
    elif existing_kind in {"hit", "bounce", "flight"}:
        kind = existing_kind  # type: ignore[assignment]
    else:
        kind = "ambiguous"

    confidence = 1.0 / (1.0 + exp(-5.0 * abs(margin)))
    return ContactDecision(
        kind=kind,
        hit_score=hit_score,
        bounce_score=bounce_score,
        margin=margin,
        confidence=float(confidence),
        evidence=tuple(evidence),
        changed=bool(existing_kind in {"hit", "bounce", "flight"} and kind != existing_kind),
    )
