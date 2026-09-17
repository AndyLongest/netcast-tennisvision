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
    assert "payload.ingested_frames" in script
    assert "payload.processed_frames" in script
    assert "pulseRouteLink('upload')" in script
    assert "pulseRouteLink('result', 150)" in script
    assert "displayedLiveSourceTime" in script
    assert "queueLiveEvents(payload.events, payload)" in script
    assert "算法已确认 · 等待左侧画面到达确认时刻" in script
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


def test_expired_live_session_returns_to_clean_idle_page() -> None:
    script = (ROOT / "web" / "live-lab.js").read_text(encoding="utf-8")

    assert "function restoreIdleLab" in script
    assert "sessionStorage.removeItem('netcastLiveSession')" in script
    assert "history.replaceState(null, '', window.location.pathname)" in script
    assert "restoreIdleLab('上一次实验已经结束，请开始新的测试')" in script
    assert "window.scrollTo({top: 0, left: 0, behavior: 'instant'})" in script


def test_offline_minimap_persists_beyond_the_yellow_flash() -> None:
    script = (ROOT / "web" / "app.js").read_text(encoding="utf-8")

    assert "const confirmedEvents" in script
    assert "const recentDecision = now - latest.t <= 1.15 ? latest : null" in script
