import hashlib
import json
from pathlib import Path

import pytest

from pipeline_runner import (
    FROZEN_BOUNCE_CLASSIFIER,
    FROZEN_BOUNCE_CLASSIFIER_SHA256,
    RACKETVISION_BALLTRACK,
    RACKETVISION_BALLTRACK_SHA256,
)

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.assets


def test_racketvision_production_weights_are_frozen():
    manifest = json.loads(
        (ROOT / "tests/fixtures/production_manifest.json").read_text(encoding="utf-8")
    )
    actual = hashlib.sha256(RACKETVISION_BALLTRACK.read_bytes()).hexdigest()
    assert actual == RACKETVISION_BALLTRACK_SHA256
    assert actual == manifest["weights"]["racketvision"]["sha256"]
    assert manifest["training_during_upload"] is False


def test_temporal_bounce_classifier_is_frozen():
    actual = hashlib.sha256(FROZEN_BOUNCE_CLASSIFIER.read_bytes()).hexdigest()
    assert actual == FROZEN_BOUNCE_CLASSIFIER_SHA256
