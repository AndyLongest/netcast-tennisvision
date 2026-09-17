# Uncertainty-aware in/out line call

## Problem

The previous production notebook classified a touchdown by its point centre and then
silently pulled any point up to 0.12m outside the legal zones back onto the court. That
made the yellow feedback look complete, but it conflated display tolerance with a tennis
rule and could turn a genuinely out ball into an in ball.

The replacement does not move a landing. It compares the confirmed touchdown with the
outer edge of the legal singles or doubles rectangle, includes the physical ball radius,
and propagates both touchdown and court-fit pixel error through the local inverse
homography. Because perspective changes metres-per-pixel with depth, this uncertainty is
computed at each landing rather than using one near/far tolerance.

The result has three states:

- `in`: the complete uncertainty interval still touches or lies inside the line;
- `out`: the complete uncertainty interval is outside the ball-touching-line boundary;
- `review`: the image evidence straddles the decision boundary.

Only `in` produces a yellow zone flash. Only `out` produces a red cross. `review` is
exported and shown as an amber dashed ring so low-quality footage is not presented as a
centimetre-accurate automatic call.

## References and adopted ideas

- The 2026 ITF Rules measure court dimensions to the outside edge of lines; the rule that
  a ball touching a line is in defines the physical decision boundary:
  https://www.itftennis.com/media/7221/2026-rules-of-tennis-english.pdf
- `TennisCourtDetector` refines court keypoints with painted-line evidence and uses a
  reference-court homography to reconstruct inconsistent points. Production already uses
  the corresponding multi-line fixed-camera principle:
  https://github.com/yastrebksv/TennisCourtDetector
- *Monocular Visual Analysis for Electronic Line Calling of Tennis Games* compares the
  touchdown with the court sideline in calibrated image geometry:
  https://arxiv.org/abs/2107.09255
- *Where Is The Ball* evaluates landing position after calibrated ray/ground intersection
  and reports that monocular landing error remains material. This supports retaining a
  review band instead of pretending a single-camera 30fps call is Hawk-Eye-equivalent:
  https://openaccess.thecvf.com/content/CVPR2025W/CVSPORTS/papers/Ponglertnapakorn_Where_Is_The_Ball_3D_Ball_Trajectory_Estimation_From_2D_CVPRW_2025_paper.pdf

## Contract

Implementation lives in `src/netcast_tennisvision/events/line_call.py`. It consumes the
frozen touchdown and court calibration after landing detection. It may classify or flag
the result, but it may not change touchdown time, image position or world position.
