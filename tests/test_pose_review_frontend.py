from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_pose_review_entry_and_page_are_wired() -> None:
    index = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    page = (ROOT / "web" / "pose-review.html").read_text(encoding="utf-8")

    assert 'href="pose-review.html">姿态与球拍样例</a>' in index
    assert 'href="pose-review.css?v=' in page
    assert "full_ball_human_racket_review_h264.mp4" in page
    assert 'id="poseFullscreen"' in page
    assert "video.requestFullscreen" in page
    assert (ROOT / "web" / "pose-review.css").is_file()


def test_pose_review_uses_baked_video_for_fullscreen_reliability() -> None:
    page = (ROOT / "web" / "pose-review.html").read_text(encoding="utf-8")

    assert "已经写入视频帧" in page
    assert "全屏回放以成片为准" in page
    assert "canvas" not in page
