# Open-source speed reconstruction selection

## Sources inspected

- abdullahtarek/tennis_analysis, commit d557527793820f1e6b06872256824255facd47fd,
  https://github.com/abdullahtarek/tennis_analysis/blob/main/main.py
  computes mini-court endpoint distance divided by shot interval, hardcoding 24 fps.
  Not selected for airborne 3D speed or arbitrary native frame rates.
- https://github.com/seymaerdogan0/tennis-ball-3d-tracking-and-speed-estimation
  uses calibrated multi-camera DLT reconstruction. Suitable with synchronized
  cameras, not a drop-in for our single uploaded video.
- TT3D, commit a2ef524ea0400262d6808db6cacf4a0b90bd0ad7:
  https://github.com/cogsys-tuebingen/tt3d/blob/main/tt3d/rally/casadi_dae.py
  and casadi_reconstruction.py implement physical dynamics and optimize image
  reprojection. Paper:
  https://openaccess.thecvf.com/content/CVPR2025W/CVSPORTS/papers/Gossard_TT3D_Table_Tennis_3D_Reconstruction_CVPRW_2025_paper.pdf
  Selected as the methodological reference, not an imported software dependency.
  Repository has no explicit LICENSE at inspection. No source, weights or
  dependencies copied. Independent implementation of standard equations.

## Adaptation

Keep existing calibrated rays and real detections. Replace constant horizontal
velocity with acceleration g - k |v| v. Numerical ODE integration and robust
reprojection optimization use existing SciPy. Tennis k is rho Cd pi r²/(2m),
with rho=1.21 kg/m³, Cd=.55, r=.033 m, m=.057 kg. These are priors, not measured
properties of the uploaded ball. Tennis physics reference:
https://www.physics.usyd.edu.au/~cross/TRAJECTORIES/42.%20Ball%20Trajectories.pdf

The TT3D table-tennis ball constants, bounce friction/restitution and spin solver
are NOT reused. Current sparse tennis video cannot reliably identify spin.
The present adaptation is gravity+drag only, with unmodelled spin/wind clearly
identified as limitations. It is not the complete TT3D algorithm and does not
inherit its reported accuracy. Calibration assumptions remain unchanged.

Refit with drag ±25%; withhold speeds changing by >20%. Export
`drag_per_m` and `drag_sensitivity_kmh`; neither parameter sensitivity nor pixel
sensitivity is a total confidence interval. Output remains midpoint 3D speed.
`TENNISVISION_SPEED_METHOD=drag` is default; `gravity` restores v1; `off` disables
speed reporting. No detector, event or court coordinate mutation. No new GPU model.

## Validation

Independent tight-tolerance forward ODE generates known drag trajectories for
1.43 m and 6.2 m camera heights. Both reconstructed midpoint speeds are within
0.3 km/h on noiseless synthetic inputs. This is implementation verification only.
Full native 1737-frame high demo: all non-speed scene fields exactly equal v1,
including every manual landing review window. 24 accepted speed windows.
Full flat2 reuses identical cached detections/events and refreshes report speeds.
Measurements are saved to outputs/speed_v2/timing.json. No real radar benchmark;
no claim that more accepted windows means more accurate speeds. Cloud images
and live workers have not been deployed or benchmarked by this change.

Measured full-flat2 CPU postprocessing: 42.59 seconds, 97 accepted windows, versus roughly 2.65 seconds for v1. This increases CPU cost substantially; no live rollout.
