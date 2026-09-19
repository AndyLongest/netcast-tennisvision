"""Express trajectory evidence in one camera view without changing observations."""
import numpy as np


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
