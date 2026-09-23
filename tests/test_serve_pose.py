import builtins

from netcast_tennisvision.vision.serve_pose import raised_arm_sequence, verify_serve_pose


def test_off_never_imports_model_or_reads_video(monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert name not in {"torch", "ultralytics", "cv2"}
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    serves, report = verify_serve_pose("missing.mp4", None, fps=30, mode="off")
    assert serves == []
    assert report["status"] == "disabled"
    assert report["samples"] == 0


def pose(left, right, confidence=1):
    p = [[0, 0, confidence] for _ in range(17)]
    p[5][1] = p[6][1] = 100
    p[11][1] = p[12][1] = 200
    p[9][1], p[10][1] = left, right
    return p


def test_pose_requires_ordered_opposite_arms_and_confidence():
    assert raised_arm_sequence([(-.4, pose(40, 160)), (0, pose(160, 40))])
    assert not raised_arm_sequence([(-.4, pose(40, 160)), (0, pose(40, 160))])
    assert not raised_arm_sequence([(-.4, pose(40, 160, .2)), (0, pose(160, 40))])
    assert not raised_arm_sequence([(0, pose(160, 40))])


def test_shadow_with_no_candidates_keeps_baseline():
    serves, report = verify_serve_pose("missing.mp4", [], fps=30, mode="shadow")
    assert serves == []
    assert report["status"] == "no_candidates"


def test_budget_exhaustion_returns_no_partial_boundaries(monkeypatch):
    from netcast_tennisvision.events import serve_sequence as serve
    from netcast_tennisvision.vision import serve_pose
    monkeypatch.setattr(serve, "pose_schedule", lambda *a, **k: [{"frame": 10}])
    times = iter([0, 2, 3])
    monkeypatch.setattr(serve_pose.time, "perf_counter", lambda: next(times))
    serves, report = verify_serve_pose("missing.mp4", [], fps=30, mode="on", budget_seconds=1)
    assert serves == []
    assert report["status"] == "budget_exceeded"


def test_candidate_backend_failure_falls_back(monkeypatch):
    from netcast_tennisvision.events import serve_sequence as serve
    def fail(*args, **kwargs):
        raise LookupError("test backend failure")
    monkeypatch.setattr(serve, "pose_schedule", fail)
    serves, report = verify_serve_pose("missing.mp4", [], fps=30, mode="on")
    assert serves == []
    assert report["status"] == "unavailable"
    assert "LookupError" in report["error"]
