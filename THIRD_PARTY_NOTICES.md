# Third-party notices and provenance

This file records technical provenance. It is not a substitute for the unresolved main
project license decision in `docs/PUBLICATION_CHECKLIST.md`.

## Original Tennis_Vision notebook

- Source: <https://github.com/vahehambardzumyan/Tennis_Vision>
- Use: original notebook and project foundation.
- Observed status on 2026-08-29: public repository with no license file.
- Publication status: **blocked pending permission or license clarification**.

## RacketVision

- Source: <https://github.com/OrcustD/RacketVision>
- Model repository: <https://huggingface.co/linfeng302/RacketVision-Models>
- Use: MS-TrackNetV3 ball candidate network and upstream `balltrack_best.pth` tensors.
- License stated by upstream: MIT.
- Local production file is a state-dict-only extraction; tensor shapes/counts and its
  SHA-256 are frozen in `assets/manifest.json`.

## Ultralytics

- Source: <https://github.com/ultralytics/ultralytics>
- Use: YOLO11 person/court segmentation runtime and `yolo11n-seg.pt`.
- Open-source license stated by upstream: AGPL-3.0; an Enterprise license is the
  alternative for incompatible proprietary use.
- Publication status: owner/legal decision required before public or commercial release.

## Torchreid / OSNet-AIN

- Source: <https://github.com/KaiyangZhou/deep-person-reid>
- Model repository: <https://huggingface.co/kaiyangzhou/osnet>
- Use: sparse appearance embeddings that keep player A/B identity stable across court-side changes.
- Torchreid source license stated by upstream: MIT.
- The exact OSNet-AIN checkpoint revision and SHA-256 are frozen in `assets/manifest.json`.
- Publication status: checkpoint training-data terms require review before public or commercial release.

## Scientific references

Citation metadata and source links live in `docs/references/references.bib` and
`docs/references/README.md`. Local PDF copies are ignored and must not be uploaded unless
their redistribution terms are separately verified.
