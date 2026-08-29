"""Execute the repository's original notebook as a callable, progress-reporting job."""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import random
import threading
import time
import traceback
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "notebooks" / "tennis_detection.ipynb"
STATUS = ROOT / "data" / "job_status.json"
CLIP = ROOT / "data" / "clip.mp4"
RACKETVISION_BALLTRACK = ROOT / "models" / "racketvision_balltrack_state_v1.pt"
RACKETVISION_BALLTRACK_SHA256 = "64c871b5079d7f3440b377ef5bb66c9ed070ed91060e075c8672c5b7ab4c691f"
FROZEN_BOUNCE_CLASSIFIER = ROOT / "models" / "bounce_classifier_production_v1.pkl"
FROZEN_BOUNCE_CLASSIFIER_SHA256 = "b53b1d3330af2b457e3b329c94ded9994d3098c6c7b3850fa80c727539ec7cfb"
CALIBRATION_REQUEST = ROOT / "data" / "court_calibration_request.json"
CALIBRATION_RESPONSE = ROOT / "data" / "court_calibration_response.json"
CALIBRATION_PREVIEW = ROOT / "data" / "court_calibration_preview.jpg"
_STATUS_LOCK = threading.Lock()
_STATUS_REPLACE_ATTEMPTS = 30


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_racketvision_balltrack() -> str:
    """Verify the immutable public RacketVision checkpoint used for production inference."""
    if not RACKETVISION_BALLTRACK.exists():
        raise RuntimeError(f"缺少公开球追踪模型：{RACKETVISION_BALLTRACK.name}")
    actual_hash = sha256(RACKETVISION_BALLTRACK)
    if actual_hash != RACKETVISION_BALLTRACK_SHA256:
        raise RuntimeError("RacketVision公开权重校验失败，拒绝开始分析")
    return actual_hash


def install_frozen_bounce_classifier() -> None:
    """Replace runtime fitting with a verified, pre-fitted temporal classifier."""
    if not FROZEN_BOUNCE_CLASSIFIER.exists():
        raise RuntimeError(f"缺少冻结落点时序模型：{FROZEN_BOUNCE_CLASSIFIER.name}")
    if sha256(FROZEN_BOUNCE_CLASSIFIER) != FROZEN_BOUNCE_CLASSIFIER_SHA256:
        raise RuntimeError("冻结落点时序模型校验失败，拒绝开始分析")
    with FROZEN_BOUNCE_CLASSIFIER.open("rb") as stream:
        frozen_model = pickle.load(stream)
    if getattr(frozen_model, "reference_rows_", None) != 3674:
        raise RuntimeError("冻结落点时序模型元数据异常")
    import bounce_sequence
    bounce_sequence.train_open_classifier = lambda _reference_csv: frozen_model


def write_status(
    state: str, progress: int, stage: str, error: str | None = None, **extra: object,
) -> None:
    payload = {"state": state, "progress": progress, "stage": stage}
    if error:
        payload["error"] = error
    payload.update(extra)
    body = json.dumps(payload, ensure_ascii=False)
    # Windows may briefly deny replacement while the HTTP status endpoint is reading the
    # destination. Progress reporting is auxiliary and must never abort video inference.
    with _STATUS_LOCK:
        temporary = STATUS.with_name(
            f".{STATUS.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            temporary.write_text(body, encoding="utf-8")
            for attempt in range(_STATUS_REPLACE_ATTEMPTS):
                try:
                    os.replace(temporary, STATUS)
                    return
                except PermissionError:
                    if attempt + 1 < _STATUS_REPLACE_ATTEMPTS:
                        time.sleep(0.01 * (1 + attempt // 5))
            print("warning: progress file stayed locked; analysis continues", flush=True)
        finally:
            temporary.unlink(missing_ok=True)


def progress_for(cell_index: int) -> tuple[int, str]:
    if cell_index <= 8:
        return 12, "正在识别球场并进行九线校准"
    if cell_index <= 19:
        return 48, "正在识别网球与球员，并复用固定球场标定"
    if cell_index <= 24:
        return 70, "正在建立连续球轨迹与物理状态"
    if cell_index <= 25:
        return 82, "正在融合音频并判断击球与落地"
    if cell_index <= 30:
        return 90, "正在重建三维坐标与球的飞行高度"
    return 96, "正在渲染真实标注视频与交互式三维报告"


def inference_progress(done: int, total: int) -> None:
    fraction = done / max(1, total)
    progress = 50 + int(15 * min(1.0, fraction))
    write_status("running", progress, f"正在逐帧识别球员（{done}/{total}帧）")


def racketvision_progress(done: int, total: int) -> None:
    fraction = done / max(1, total)
    progress = 34 + int(15 * min(1.0, fraction))
    write_status("running", progress, f"正在用时序模型识别网球（{done}/{total}帧）")


def request_manual_court_calibration(
    preview_frame: object,
    *,
    reason: str,
    automatic_confidence: float,
    timeout_seconds: float = 30 * 60,
) -> list[list[float]]:
    """Pause only a low-confidence job until the UI supplies four court corners."""
    import cv2
    import numpy as np

    frame = np.asarray(preview_frame)
    if frame.ndim != 3 or frame.shape[0] < 2 or frame.shape[1] < 2:
        raise ValueError("无法生成球场校准画面")
    request_id = f"{sha256(CLIP)[:16]}-{uuid.uuid4().hex[:8]}"
    CALIBRATION_RESPONSE.unlink(missing_ok=True)
    if not cv2.imwrite(str(CALIBRATION_PREVIEW), frame):
        raise OSError("无法保存球场校准画面")
    request = {
        "request_id": request_id,
        "video_sha256": sha256(CLIP),
        "width": int(frame.shape[1]),
        "height": int(frame.shape[0]),
        "reason": reason,
        "automatic_confidence": round(float(automatic_confidence), 4),
        "point_order": ["近端左角", "近端右角", "远端右角", "远端左角"],
    }
    CALIBRATION_REQUEST.write_text(
        json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
    write_status(
        "needs_court_calibration", 15, "需要确认球场边界",
        calibration={
            **request,
            "preview": "/data/court_calibration_preview.jpg",
        },
    )

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if CALIBRATION_RESPONSE.exists():
            try:
                response = json.loads(CALIBRATION_RESPONSE.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                time.sleep(0.25)
                continue
            if response.get("request_id") == request_id:
                corners = response.get("corners")
                if not isinstance(corners, list) or len(corners) != 4:
                    raise ValueError("人工球场校准没有包含四个角点")
                write_status("running", 16, "球场边界已确认，继续自动分析")
                return corners
        time.sleep(0.35)
    raise TimeoutError("等待球场边界确认超时，请重新发起分析")


def main() -> None:
    write_status("running", 2, "正在加载 RacketVision 与冻结落点模型（纯推理）")
    verify_racketvision_balltrack()
    install_frozen_bounce_classifier()
    os.environ["TENNISVISION_MODEL_MODE"] = "racketvision_open_weights"
    # Keep the remaining numerical pipeline reproducible as well.
    random.seed(20260824)
    try:
        import numpy as np
        np.random.seed(20260824)
        import torch
        torch.manual_seed(20260824)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(20260824)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    except ImportError:
        pass
    # Notebook plotting must never open a blocking desktop window in a background job.
    os.environ.setdefault("MPLBACKEND", "Agg")
    if os.name == "nt":
        packages = Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages"
        full_ffmpeg = sorted(packages.glob("Gyan.FFmpeg.Shared_*/ffmpeg-*/bin/ffmpeg.exe"),
                             reverse=True)
        if full_ffmpeg:
            os.environ["PATH"] = str(full_ffmpeg[0].parent) + os.pathsep + os.environ.get("PATH", "")
    # Notebook-only rich displays can dump megabytes of embedded HTML to a background
    # process and break its output pipe.  The runner writes those artifacts to disk, so
    # suppressing their inline copies changes no analysis result.
    try:
        import IPython.display as ipython_display
        ipython_display.display = lambda *_args, **_kwargs: None
    except ImportError:
        pass
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    namespace: dict[str, object] = {
        "__name__": "__main__",
        "_pipeline_inference_progress": inference_progress,
        "_racketvision_inference_progress": racketvision_progress,
        "_pipeline_manual_court_calibration": request_manual_court_calibration,
    }
    os.chdir(ROOT)
    try:
        for index, cell in enumerate(notebook["cells"]):
            # Cells 23-24 are the retired clip-trained BallNet. RacketVision now supplies
            # the candidates; running both would let two detectors fight over one ball.
            if cell.get("cell_type") != "code" or index in (23, 24, 36):
                continue
            progress, stage = progress_for(index)
            write_status("running", progress, stage)
            source = "".join(cell.get("source", []))
            exec(compile(source, f"{NOTEBOOK.name}:cell-{index}", "exec"), namespace)
        outputs = ROOT / "data" / "outputs"
        required = [outputs / "annotated_clip.mp4", outputs / "scene3d.json", outputs / "rally3d.html"]
        missing = [path.name for path in required if not path.exists()]
        if missing:
            raise RuntimeError("分析结束但缺少输出：" + ", ".join(missing))
        write_status("complete", 100, "真实分析完成（RacketVision公开权重·纯推理）")
    except Exception as exc:
        traceback.print_exc()
        write_status("error", 100, "分析失败", f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
