import json
from pathlib import Path

import pytest

from netcast_tennisvision.pipeline import runner as pipeline_runner
from netcast_tennisvision.pipeline import video_encoding
from netcast_tennisvision.pipeline.video_encoding import raw_h264_output_args

ROOT = Path(__file__).resolve().parents[1]


def test_notebook_has_one_frozen_ball_path_and_stable_runner_cells():
    notebook = json.loads(
        (ROOT / "notebooks" / "tennis_detection.ipynb").read_text(encoding="utf-8")
    )
    ids = {cell.get("id") for cell in notebook["cells"]}
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert {"a782a51e", "3aa72724"} <= ids
    assert not {"23c8fa62", "9fe84949", "e6a661f9"} & ids
    assert "class BallNet" not in source
    assert "ball_heatmap_" not in source
    assert "def run_production_tracker" in source


def test_notebook_progress_is_bound_to_stable_cell_ids():
    assert pipeline_runner.progress_for("1551a712") == (
        48,
        "正在识别网球与球员，并复用固定球场标定",
    )
    assert pipeline_runner.progress_for("2b62675d")[0] == 70
    assert pipeline_runner.progress_for("9338f9cf")[0] == 82
    assert pipeline_runner.progress_for("1d6d5185")[0] == 96


def test_status_write_retries_a_transient_windows_lock(tmp_path, monkeypatch):
    status = tmp_path / "job_status.json"
    real_replace = pipeline_runner.os.replace
    attempts = 0

    def flaky_replace(source, destination):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError(5, "locked")
        real_replace(source, destination)

    monkeypatch.setattr(pipeline_runner, "STATUS", status)
    monkeypatch.setattr(pipeline_runner.os, "replace", flaky_replace)
    monkeypatch.setattr(pipeline_runner.time, "sleep", lambda _seconds: None)
    pipeline_runner.write_status("running", 48, "测试")

    assert attempts == 3
    assert json.loads(status.read_text(encoding="utf-8")) == {
        "state": "running", "progress": 48, "stage": "测试"
    }


def test_status_lock_never_aborts_analysis(tmp_path, monkeypatch):
    status = tmp_path / "job_status.json"
    monkeypatch.setattr(pipeline_runner, "STATUS", status)
    monkeypatch.setattr(pipeline_runner, "_STATUS_REPLACE_ATTEMPTS", 2)
    monkeypatch.setattr(
        pipeline_runner.os, "replace",
        lambda _source, _destination: (_ for _ in ()).throw(PermissionError(5, "locked")),
    )
    monkeypatch.setattr(pipeline_runner.time, "sleep", lambda _seconds: None)

    pipeline_runner.write_status("running", 48, "测试")
    assert not list(tmp_path.glob("*.tmp"))


def test_event_overlay_output_is_explicit_and_reversible(monkeypatch):
    monkeypatch.delenv("TENNISVISION_OUTPUT_MODE", raising=False)
    assert not pipeline_runner.event_overlay_output_enabled()
    monkeypatch.setenv("TENNISVISION_OUTPUT_MODE", "event-overlay")
    assert pipeline_runner.event_overlay_output_enabled()
    monkeypatch.setenv("TENNISVISION_OUTPUT_MODE", "annotated-video")
    assert not pipeline_runner.event_overlay_output_enabled()


def test_video_encoder_uses_fast_reversible_default(monkeypatch):
    monkeypatch.setattr(video_encoding, "_nvenc_usable", lambda: False)
    monkeypatch.delenv("NETCAST_X264_PRESET", raising=False)
    monkeypatch.delenv("NETCAST_X264_CRF", raising=False)
    args = raw_h264_output_args()
    assert args[args.index("-preset") + 1] == "veryfast"
    assert args[args.index("-crf") + 1] == "20"
    assert "+faststart" in args


def test_video_encoder_can_restore_previous_preset(monkeypatch):
    monkeypatch.setenv("NETCAST_VIDEO_ENCODER", "x264")
    monkeypatch.setenv("NETCAST_X264_PRESET", "medium")
    assert raw_h264_output_args()[3] == "medium"


def test_video_encoder_uses_nvenc_only_after_a_successful_preflight(monkeypatch):
    monkeypatch.setenv("NETCAST_VIDEO_ENCODER", "auto")
    monkeypatch.setattr(video_encoding, "_nvenc_usable", lambda: True)

    args = raw_h264_output_args()

    assert args[args.index("-c:v") + 1] == "h264_nvenc"
    assert args[args.index("-cq") + 1] == "20"


def test_video_encoder_rejects_unknown_mode(monkeypatch):
    monkeypatch.setenv("NETCAST_VIDEO_ENCODER", "mystery")
    with pytest.raises(ValueError, match="unsupported NETCAST_VIDEO_ENCODER"):
        raw_h264_output_args()
