from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_live_calibration_clicks_follow_contained_image() -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required for browser coordinate regression")
    script = (ROOT / "web" / "live-lab.js").read_text(encoding="utf-8")
    start = script.index("function liveCalibrationPoint(")
    end = script.index("\n}", start) + 2
    checks = r"""
const assert = require('node:assert/strict');
function check(actual, expected) {
  assert.ok(actual);
  actual.forEach((value, i) => assert.ok(Math.abs(value - expected[i]) < 1e-10));
}
// Wide but height-limited dialog: image occupies x=250..850, not x=50..1050.
const wide = {left: 50, top: 100, width: 1000, height: 337.5};
check(liveCalibrationPoint(310, 370, wide, 1280, 720), [0.1, 0.8]);
check(liveCalibrationPoint(250, 100, wide, 1280, 720), [0, 0]);
assert.equal(liveCalibrationPoint(100, 200, wide, 1280, 720), null);
// Tall box: ignore top/bottom bars rather than clamping them to court corners.
const tall = {left: 20, top: 40, width: 400, height: 500};
check(liveCalibrationPoint(120, 357.5, tall, 1920, 1080), [0.25, 0.8]);
assert.equal(liveCalibrationPoint(120, 50, tall, 1920, 1080), null);
// Resizing keeps the same normalized point without introducing any snapping.
check(liveCalibrationPoint(60, 220, {left:20, top:40, width:400, height:225}, 1280,720), [0.1,0.8]);
assert.equal(liveCalibrationPoint(0,0,{left:0,top:0,width:0,height:0},1280,720), null);
"""
    subprocess.run([node, "-e", script[start:end] + checks], check=True, capture_output=True, text=True)


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
    assert "待复核" not in page
    assert "line_call" in script
    assert "review-ball" not in page
    assert "ctx.setLineDash([3, 2])" not in script
    assert "courtWidth = 10.97, courtLength = 23.77" in script
    assert "courtW = courtH * (courtWidth / courtLength)" in script
    assert "marginY + (courtLength - metres) / courtLength * courtH" in script
    assert 'id="liveCalibrationDialog"' in page
    assert 'id="liveCalibrationCanvas"' in page
    assert "openLiveCalibration(file)" in script
    assert "JSON.stringify({court_corners: corners})" in script
    assert "'X-Court-Corners': JSON.stringify(corners)" in script
    assert 'id="playerAColor"' in page
    assert 'id="playerBColor"' in page
    assert "event.player_color" in script
    assert "state.latestLivePayload?.player_colors" in script
    assert "球员 A 主色" in page
    assert "球员 B 主色" in page


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
    assert "const recentDecision = latest && now - eventDecisionTime(latest) <= 1.15" in script
    assert "const mapWidth = width * (50 / 640)" in script
    assert "const left = width - mapWidth - mapMargin" in script
    assert "if (!latest) return" not in script
    assert "drawVideoBallAndTrail(ctx, width, height, now)" in script
    assert "courtProjector(courtCornersAt(state.scene, now)" in script
    assert "zoneBounds[recentDecision.zone]" in script


def test_completed_report_reuse_is_algorithm_versioned() -> None:
    script = (ROOT / "web" / "app.js").read_text(encoding="utf-8")

    assert "const ANALYSIS_VERSION = 'production-v29'" in script
    assert "current.algorithm_version === ANALYSIS_VERSION" in script
