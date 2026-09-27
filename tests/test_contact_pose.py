import copy

import numpy as np

from netcast_tennisvision.events.contact_pose import (
    pose_contact_evidence,
    propose_pose_contact_corrections,
)


def sample(swing=False, visible=True, impulse=True):
    frames=[dict(ball_px=(100+2*t,200+4*t-10*max(t-15.4,0) if impulse else 200+4*t),ball_seen=True,ball_track_id=1,ball_confidence=.9) for t in range(32)]
    poses=[]
    for f in range(9,23,2):
        kp=np.zeros((17,3))
        kp[:,2]=.95 if visible else .1
        for shoulder,elbow,wrist in ((5,7,9),(6,8,10)):
            kp[shoulder,:2]=[130,220]
            kp[elbow,:2]=[130,235]
            kp[wrist,:2]=[130+(f-15)*6 if swing else 130,260]
        poses.append(dict(frame=f,side='near',height=100,points=kp.tolist()))
    return frames,poses


def test_quiet_arms_plus_real_impulse_can_correct_a_hit_without_mutation():
    frames,poses=sample()
    events=[dict(frame=15,kind='hit')]
    before=copy.deepcopy(frames)
    out=propose_pose_contact_corrections(events,frames,poses,fps=30,enabled=True)
    assert len(out)==1 and out[0]['kind']=='bounce'
    assert out[0]['decision_frame']>=out[0]['candidate_frame']+7
    assert frames==before and events==[dict(frame=15,kind='hit')]


def test_swing_and_missing_joints_do_not_become_bounces():
    for swing,visible in [(True,True),(False,False)]:
        frames,poses=sample(swing,visible)
        assert pose_contact_evidence(15,frames,poses,fps=30)['status']==('swing' if visible else 'unknown')
        assert propose_pose_contact_corrections([dict(frame=15,kind='hit')],frames,poses,fps=30,enabled=True)==[]


def test_off_and_quiet_arms_without_ground_evidence_do_nothing():
    frames,poses=sample()
    assert propose_pose_contact_corrections([dict(frame=15,kind='hit')],frames,poses,fps=30)==[]
    frames,poses=sample(impulse=False)
    assert propose_pose_contact_corrections([dict(frame=15,kind='hit')],frames,poses,fps=30,enabled=True)==[]
