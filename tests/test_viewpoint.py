import copy
import io
import json
from unittest.mock import Mock

import numpy as np
import pytest

from netcast_tennisvision.vision.viewpoint import classify_viewpoint

FLAT = [[-0.02787, .74483], [1.08033, .75929], [.61967, .49603], [.39836, .49893]]
HIGH = [[.0328, .72697], [.96597, .70496], [.59468, .14391], [.35705, .14611]]


def test_existing_examples_and_no_mutation():
    before = copy.deepcopy(FLAT)
    flat = classify_viewpoint(FLAT, 960, 544)
    assert flat['category'] == 'low'
    assert flat['height_ratio'] == pytest.approx(.25458)
    assert classify_viewpoint(HIGH, 1920, 1080)['category'] == 'high'
    assert FLAT == before  # Outside-image corners are retained, not clamped.
    assert flat['analysis_path'] == 'existing'


@pytest.mark.parametrize('depth,category', [(.29999999, 'low'), (.30, 'low'), (.30000001, 'high'), (.35, 'high'), (.40, 'high')])
def test_exact_thirty_percent_split(depth, category):
    corners = [[0, .8], [1, .8], [1, .8-depth], [0, .8-depth]]
    assert classify_viewpoint(corners, 1000, 1000)['category'] == category


def test_resizing_is_invariant_but_source_padding_changes_occupancy():
    initial = classify_viewpoint(HIGH, 1920, 1080)
    assert classify_viewpoint(HIGH, 960, 540) == initial
    # Extra vertical padding reduces screen occupancy, but shape stays high.
    padded = [[x, y / 3 + 1 / 3] for x, y in HIGH]
    result = classify_viewpoint(padded, 1920, 3240)
    assert result['height_ratio'] < .30
    assert result['depth_width_ratio'] == initial['depth_width_ratio']
    assert result['category'] == 'low'


def test_rotation_does_not_add_an_extra_classification_gate():
    theta = np.deg2rad(30)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    corners = (np.asarray(HIGH) - .5) @ rotation.T + .5
    result = classify_viewpoint(corners, 1000, 1000)
    assert result['category'] == ('low' if result['height_ratio'] <= .30 else 'high')


@pytest.mark.parametrize('corners', [None, [], [[0, 0]]*4, [[0, 0], [1, 1], [0, 1], [1, 0]], [[float('nan'), 0]]*4])
def test_missing_or_invalid_geometry_is_nonblocking(corners):
    assert classify_viewpoint(corners, 1920, 1080)['category'] == 'unavailable'


def test_preview_endpoint_runs_locally_even_with_cloud_configured(monkeypatch):
    from netcast_tennisvision.api import server
    monkeypatch.setattr(server, 'CLOUD_API_URL', 'https://cloud.invalid')
    body = json.dumps({'corners': FLAT, 'width': 960, 'height': 544}).encode()
    handler = object.__new__(server.Handler)
    handler.path = '/api/viewpoint'
    handler.headers = {'Content-Length': str(len(body))}
    handler.rfile = io.BytesIO(body)
    handler.cloud_request_authorized = lambda: True
    handler.send_json = Mock()
    handler.proxy_cloud_request = Mock()
    handler.do_POST()
    assert handler.send_json.call_args.args[0]['category'] == 'low'
    handler.proxy_cloud_request.assert_not_called()
