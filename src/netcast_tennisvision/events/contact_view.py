"""Express trajectory evidence in one camera view without changing observations."""
import numpy as np


def contact_track_id(frame, frames, radius):
    """Resolve identity across a short hole, never join unrelated tracks.

    A missing centre needs real observations on both sides belonging to one track.
    Predicted coordinates are not evidence. No frame dictionary is modified.
    """
    centre = frames[frame]
    if centre.get("ball_seen") and centre.get("ball_px") is not None:
        return centre.get("ball_track_id")
    left = next((frames[i] for i in range(frame-1, max(-1, frame-radius-1), -1)
                 if frames[i].get("ball_seen") and frames[i].get("ball_px") is not None), None)
    right = next((frames[i] for i in range(frame+1, min(len(frames), frame+radius+1))
                  if frames[i].get("ball_seen") and frames[i].get("ball_px") is not None), None)
    if left is None or right is None:
        return None
    track = left.get("ball_track_id")
    return track if track is not None and track == right.get("ball_track_id") else None


def point_in_contact_view(meta, centre):
    point = np.asarray(meta["ball_px"], dtype=float)
    # Court homographies carry the camera registration. This composes the two
    # image planes; it does not assert that the airborne ball is on the ground.
    if centre.get("M") is None or meta.get("M_inv") is None:
        return point
    transform = np.asarray(centre["M"]) @ np.asarray(meta["M_inv"])
    projected = transform @ np.append(point, 1.0)
    if not np.isfinite(projected).all() or abs(projected[2]) < 1e-9:
        return point
    return projected[:2] / projected[2]
