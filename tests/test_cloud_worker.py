from __future__ import annotations

from netcast_tennisvision.cloud import worker


def test_handle_job_rejects_missing_input() -> None:
    assert worker.handle_job({}) == {"ok": False, "error": "input 必须是 JSON 对象"}


def test_handle_job_rejects_unknown_mode() -> None:
    result = worker.handle_job({"input": {"mode": "analyze_url"}})
    assert result["ok"] is False
    assert "benchmark_demo" in result["error"]


def test_handle_job_passes_valid_timeout(monkeypatch) -> None:
    monkeypatch.setattr(
        worker,
        "benchmark_demo",
        lambda *, timeout_seconds: {"ok": True, "timeout": timeout_seconds},
    )
    assert worker.handle_job(
        {"input": {"mode": "benchmark_demo", "timeout_seconds": 900}}
    ) == {"ok": True, "timeout": 900.0}


def test_handle_job_rejects_unsafe_timeout() -> None:
    result = worker.handle_job(
        {"input": {"mode": "benchmark_demo", "timeout_seconds": 10}}
    )
    assert result["ok"] is False
    assert "60 到 7200" in result["error"]
