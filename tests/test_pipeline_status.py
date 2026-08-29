import json

from netcast_tennisvision.pipeline import runner as pipeline_runner


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
