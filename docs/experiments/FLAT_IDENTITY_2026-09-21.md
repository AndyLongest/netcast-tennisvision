# Flat demo identity and palette correction

Input: `flat_deemo.mp4`, archived job `6c731f96161c448f9227f07c1643003e`,
6982 frames at 30 fps. Existing person boxes were reused; 1313 sample frames were
decoded and embedded using the unchanged OSNet model. No new model was introduced.

The previous result recorded zero identity side switches and blue-grey/dark-red
display colours despite black and white kits. The far player was often missing, while
classification required a pair. Border-colour subtraction sometimes removed the shirt
itself and selected court pixels. A low torso region also included shorts during bends.

Changes: higher torso sampling without border subtraction; black-kit desaturation;
enrollment-frozen, pair-separated display palette; conservative clear-single-player
identity evidence; colour corroboration when enrollment shirts differ substantially;
sustained identity segments for attribution even if rally segmentation merges cuts.

Revised sampled identity checks: black A at 5/25/65/85 seconds, white B at
45/105/125/165 seconds. Display colours A `#696969`, B `#ebe8e1`. Six confirmed
mapping changes replace zero; these are not a claim that all transitions are frame-exact.
Partial bodies and cut boundaries can still have confirmation delay. At 145 seconds
the old mapping persists briefly until 146.67 seconds; do not call this perfect recall.

25 of 50 landing ownership labels changed. The 47.7483-second landing is now B.
All frame trajectories, landing coordinates/times/counts and rally boundaries were
asserted unchanged before updating the existing report. The self-contained 3D report
was updated too. Originals are retained under `outputs/flat_identity/`.

Evidence: `audit.log`, `observations.pkl`, `revised.pkl`, `identity_review.jpg`,
`original_scene3d.json`, `scene3d.json` in that output directory. These runtime
artifacts and source images are not committed. The observations are regression data,
not an independently labeled accuracy benchmark.

Full native bundled-demo pipeline completed in 45.2 seconds, 1737 frames and 27
bounces. Identity unit tests cover missing far player, sustained versus false swaps,
white/black shirts and similar palette separation; full pytest suite passed.
The local source and repaired report include the changes. Published cloud image
production-v30 is not changed by editing local Python; a release is still needed for
new cloud analyses. No cloud deployment is claimed by this experiment.
