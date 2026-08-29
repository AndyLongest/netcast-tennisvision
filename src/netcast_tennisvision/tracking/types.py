"""Shared result types for the persistent ball tracker."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TrackerDiagnostics:
    """Counters exported by one complete association pass."""

    confirmed_births: int
    rejected_singletons: int
    occluded_frames: int
    rejected_teleports: int
    merged_occlusions: int
    adaptive_search_recoveries: int
    net_terminations: int
    frame_exit_terminations: int
    bounce_velocity_resets: int
    player_hit_velocity_resets: int
    curved_flight_rejections: int
    midflight_backtrack_repairs: int
    ballistic_predictions: int
