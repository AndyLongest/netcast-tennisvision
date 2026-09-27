# RacketPose integration

RacketPose is an optional, read-only vision source. It does not change BallTrack, court
registration, bounce detection or any existing production result. Its output is the common
input for later **shot preparation** and **whip/action** scoring work.

## Output contract

For every sampled source frame the extractor records:

- original frame index and timestamp;
- zero or more racket bounding boxes and detector confidence;
- five original-pixel keypoints: `top`, `bottom`, `handle`, `left`, `right`;
- confidence for every keypoint.

`choose_player_racket(...)` associates a candidate with human-pose wrists and, when
available, the previous racket center. This is deliberately an association rule only. It
does not decide whether a player is preparing, swinging or hitting.

The geometry lets downstream code derive the handle-to-head vector, racket angle, angular
velocity, distance to the hand and distance to the ball on the same video timeline. Those
are the measurements needed by both scoring systems; scoring thresholds remain a separate
layer.

## Why the backend is isolated

The upstream RacketVision implementation was tested with `mmcv 2.1.0`, `mmdet 3.3.0`,
`mmpose 1.3.2`, NumPy 1.x and Python 3.10/3.11-era wheels. The main TennisVision runtime is
Python 3.12 with NumPy 2, so installing this compiled stack into `.venv` would make the
stable pipeline fragile.

The contract module therefore has no OpenMMLab import at module load time. Existing code
can read and use saved timelines in the normal environment. Actual model inference runs
from a separate Python 3.10 or 3.11 environment and fails closed when that environment or
the optional weights are absent.

## Install and extract

From the repository root, download the two immutable upstream checkpoints (about 495 MiB):

```powershell
.\.venv\Scripts\python.exe tools\install_assets.py --group racket_pose
```

For the reproducible Windows CUDA validation environment, run:

```powershell
.\setup_racket_pose.ps1
```

The script creates its environment outside the repository at
`..\.tools\tennis-racketpose-env` and installs matching CUDA 12.1 builds of PyTorch and
MMCV. CPU is retained only as an explicit compatibility fallback:

```powershell
.\setup_racket_pose.ps1 -Accelerator cpu
```

The equivalent manual setup is:

```powershell
conda create -n tennis-racketpose python=3.10 -y
conda run -n tennis-racketpose python -m pip install `
  torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
conda run -n tennis-racketpose python -m pip install "setuptools<81" numpy==1.26.4
conda run -n tennis-racketpose python -m pip install chumpy==0.70 --no-build-isolation
conda run -n tennis-racketpose python -m pip install mmcv==2.1.0 `
  --find-links https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html --no-deps
conda run -n tennis-racketpose python -m pip install -r requirements-racketpose.txt
conda run -n tennis-racketpose python -m pip install -e .
```

`mmcv` contains compiled operators. If the plain pip command has no wheel for the chosen
PyTorch/CUDA pair, install the matching OpenMMLab wheel with `mim`; do not compile it inside
the production `.venv`. The optional requirements also retain `setuptools<81`, because
MMPose 1.3 still imports the removed `pkg_resources` compatibility module.

Extract a complete frame-by-frame timeline:

```powershell
conda run -n tennis-racketpose python tools\extract_racket_pose.py `
  assets\demo\demo.mp4 outputs\racket_pose\demo.json --device cuda:0
```

The loader checks both model SHA-256 values before allowing PyTorch to deserialize their
MMEngine metadata. The model revision, sizes and hashes are frozen in
`assets/manifest.json`.

## Validation status

The framework-neutral schema, validation, association rule and fail-closed optional loading
are covered by the normal test suite. On 2026-09-27, the pinned Windows CUDA environment
loaded both checkpoints and ran the complete RTMDet-to-RTMPose chain on the bundled demo.
On an RTX 3050 Ti, sampling every fifth source frame processed 348 samples from the
57.9-second video: model loading took 12.20 seconds, inference took 46.15 seconds, and full
extraction including decode and JSON output took 64.29 seconds.

This is a functional and performance smoke test, not an accuracy acceptance result. A
video-level review is still required before the upload-analysis rating pipeline relies on
these observations. In particular, far-court recall, motion blur, occlusion and left/right
point flips must be measured rather than assumed.
