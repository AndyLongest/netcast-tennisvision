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
    for name in ("sceneFrameAt", "activeRallyAt", "courtCornersAt", "courtProjector"):
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
// Reproduce the ~32px reframing from the user's 1:50 screenshot. The
// first-frame fallback must not be used after the camera changes or on seek.
const before = [[.00772,.67649],[.95421,.71157],[.60389,.21166],[.36727,.20911]];
const after = [[.00946,.70569],[.95271,.73947],[.6034,.24171],[.36759,.23947]];
const moving = {fps:30, frame_times:[0,105,106,110,113],
  court_image_corners:before,
  court_keyframes:[{frame:0,corners:before},{frame:2,corners:after}]};
const project = t => courtProjector(courtCornersAt(moving,t),1920,1080)(1.37,5.48);
assert.ok(project(110)[1]-project(105)[1] > 30);
assert.deepEqual(project(113),project(110));
assert.deepEqual(project(104),project(105));
"""
    subprocess.run([node, "-e", "\n".join(helpers) + checks], check=True, capture_output=True)
