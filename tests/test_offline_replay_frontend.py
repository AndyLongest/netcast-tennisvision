"""Execute replay geometry/timeline helpers, including seeking backwards."""
import shutil
import subprocess
from pathlib import Path

import pytest


def test_replay_uses_current_camera_and_resets_at_hit():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required for replay regression")
    source = (Path(__file__).resolve().parents[1] / "web/app.js").read_text(encoding="utf-8")
    helpers = []
    for name in ("sceneFrameAt", "activeRallyAt", "courtCornersAt"):
        start = source.index(f"function {name}(")
        helpers.append(source[start:source.index("\n}", start) + 2])
    checks = """
const assert = require('node:assert/strict');
const scene = {fps:30, court_image_corners:'old',
  court_keyframes:[{frame:0,corners:'first'},{frame:60,corners:'shifted'}],
  rallies:[{rally_id:0,start_frame:0,display_end_frame:50},
           {rally_id:1,start_frame:50,display_end_frame:100}]};
assert.equal(sceneFrameAt({fps:30,frame_times:[.1,.1333,.1667]},.14),1);
assert.equal(courtCornersAt(scene, 2), 'shifted');
assert.equal(courtCornersAt(scene, 1), 'first');
assert.equal(activeRallyAt(scene, 50/30), 1);
assert.equal(activeRallyAt(scene, 1), 0);
assert.equal(activeRallyAt(scene, 4), null);
assert.equal(activeRallyAt({fps:30,hits:[{frame:30,rally_id:1}],
  bounces:[{frame:20,rally_id:0}]},1.1),1);
"""
    subprocess.run([node, "-e", "\n".join(helpers) + checks], check=True, capture_output=True)
