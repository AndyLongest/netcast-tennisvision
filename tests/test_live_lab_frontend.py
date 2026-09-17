from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_live_lab_assets_and_entrypoint_are_wired() -> None:
    index = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    page = (ROOT / "web" / "live-lab.html").read_text(encoding="utf-8")

    assert 'href="live-lab.html"' in index
    assert 'src="live-lab.js?v=' in page
    assert 'href="live-lab.css?v=' in page
    assert (ROOT / "web" / "live-lab.js").is_file()
    assert (ROOT / "web" / "live-lab.css").is_file()


def test_live_lab_exposes_real_chain_and_separate_offline_baseline() -> None:
    page = (ROOT / "web" / "live-lab.html").read_text(encoding="utf-8")
    script = (ROOT / "web" / "live-lab.js").read_text(encoding="utf-8")

    assert "真实链路实验" in page
    assert "开始真实端到端实验" in page
    assert "查看离线基准回放" in page
    assert "上传其他视频" in page
    assert "RTMP / ZLMediaKit" in page
    assert "/api/live-lab/start" in script
    assert "/api/live-lab/status" in script
    assert "/api/live-lab/upload" in script
    assert "/api/live-lab/frame" not in script
    assert "ZLMRTCClient.js" not in page
    assert "payload.fmp4_playback_url" in script
    assert "连续直播 · HTTPS" in script
    assert "Math.max(reportedDecision, touchdownFrame + 10)" in script
    assert '<div class="court-orientation"' in page
    assert "<span>远端</span><span>近端</span>" in page
    assert "球员 A" in page
    assert "球员 B" in page
    assert "待复核" in page
    assert "line_call" in script
    assert "压线待复核" in script
    assert "courtWidth = 10.97, courtLength = 23.77" in script
    assert "courtW = courtH * (courtWidth / courtLength)" in script
    assert "marginY + (courtLength - metres) / courtLength * courtH" in script
