# Player identity: validation and production integration

The player-identity layer reuses `person_boxes` from Pass A, extracts OSNet-AIN appearance
features every fifth native frame, assigns the two players jointly, and requires three
strong consecutive samples plus physically sufficient travel time before accepting a side
change. Short occlusions keep the existing track alive; a person detection outside the
reachable court-space gate cannot replace a player in one sample.

## Production implementation

The approved logic lives in `src/netcast_tennisvision/vision/player_identity.py`. The
notebook calls it after the ball/person vision pass and before contact attribution. It
writes identity metadata only; it cannot alter ball candidates, trajectory coordinates,
contact classification, or landing coordinates.

Confirmed landings inherit the preceding hitter. If a clip begins after that strike,
tennis-side logic assigns the hitter from the opposite half of the first landing. The
fixed demo exercises two accepted side changes. All 35 hits and all 29 landings receive
an A/B owner; the landing markers split into 14 for A and 15 for B. The main trajectory
remains 1220 positioned frames, including 1123 detector-backed observations.

## Reproduce the original diagnostic

Install the normal project environment and checksummed assets:

```powershell
.\setup.ps1
```

Then run:

```powershell
.\.venv\Scripts\python.exe tools\experiment_player_identity.py
```

The diagnostic video and JSON report are written under `outputs/`. Production analyses
write identity metadata directly into `scene3d.json`.

## 2026-09-05 validation result

Input: the 133.68-second local validation clip, 2769 cached frames at 20.9669 fps.

| Diagnostic | Result |
|---|---:|
| OSNet sample frames | 554 |
| Two-player samples after enrolment | 409 |
| Agreement with unchanged initial sides | 100.0% |
| Accepted identity-side switches | 0 |
| Mean pairwise assignment advantage | 0.3008 |

That clip contains no actual changeover. It demonstrates stable identity separation under
scale and pose changes, not changeover recall. The fixed product demo's two detected side
changes are an integration check, not a labelled changeover-accuracy claim. Similarity
advantage is an auditable margin, not a calibrated probability.

## Design references

- ByteTrack, ECCV 2022: retain established tracks through weak detections instead of
  immediately fragmenting identity: https://arxiv.org/abs/2110.06864
- Deep SORT, ICIP 2017: combine motion state with cosine appearance association:
  https://arxiv.org/abs/1703.07402
- BoT-SORT, 2022: jointly use motion, appearance and camera compensation:
  https://arxiv.org/abs/2206.14651
- Global ID Fusion, WACV 2026: separate short tracklets from global player identity and
  fuse them contextually: https://openaccess.thecvf.com/content/WACV2026/html/Wojtulewicz_Advancing_Player_Identification_and_Tracking_with_Global_ID_Fusion_GIF_WACV_2026_paper.html

The production implementation adopts the inexpensive parts that fit this fixed-camera,
two-player setting: court-coordinate motion gating, track persistence, paired assignment
and delayed changeover confirmation. It does not add a second general-purpose MOT model.
