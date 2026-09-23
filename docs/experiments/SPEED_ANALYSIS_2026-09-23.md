# Monocular flight speed experiment

The report now fits 3D flight from real ball observations and court-derived camera
rays. It never differentiates the projected ground position of an airborne ball.
Contact, track and calibration changes split windows; 10+ observations spanning
0.3–0.85 seconds are required. Robust pixel reprojection fit, physical height,
Jacobian conditioning and pixel-noise sensitivity gates suppress weak estimates.
Both low and high viewpoints use the same method. Existing tracking camera-height
default remains unchanged; speed permits physically positive low camera heights.

Synthetic cameras at 1.43 m and 6.2 m recover known noiseless 3D speeds within
0.2 km/h. This verifies geometry, NOT real-world measurement accuracy. Full native
1737-frame high demo run leaves all pre-existing scene fields exactly unchanged.
Full flat2 9000-frame pose-contact experiment yields 99 accepted speed windows;
high cached analysis yields 23. CPU postprocessing measured about 2.65 s and
0.44 s respectively on this machine, not an end-to-end live benchmark. Window
support requires future observations, so a live integration would add buffering.

No radar or calibrated ground-truth speed is available. Principal point, focal
length assumptions, court corner error, lens distortion, drag/spin, missed contacts
and false ball associations can bias speed despite small pixel residuals. Do not
advertise this as accurate radar-equivalent measurement. Frontend labels estimates
and does not claim the pixel sensitivity is total uncertainty. Cloud not deployed.

Artifacts: outputs/contact_pose_full/result/review.html and scene3d.json;
outputs/speed_validation/high/scene3d.json. Full flat2 keeps original audio and
no ball trail. Pose-assisted contact changes remain experimental as documented.
