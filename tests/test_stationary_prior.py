import numpy as np

from netcast_tennisvision.tracking.stationary import StationaryPrior, coherent_birth
from netcast_tennisvision.tracking.world_tracker import track_ball_persistent


def test_dwell_expires_and_does_not_ban_revisited_positions():
    prior=StationaryPrior(30,1)
    for i in range(30):
        result=prior.update(i,[(100+i%2,100)])
    assert result==[True]
    assert prior.update(65,[(100,100)])==[False]
    prior.reset()
    assert prior.update(66,[(100,100)])==[False]


def test_moving_ball_is_not_static_and_jitter_is_not_a_launch():
    prior=StationaryPrior(30,1)
    for i in range(60):
        assert prior.update(i,[(100+4*i,100)])==[False]
    obs=[(i,np.array(p),.9) for i,p in enumerate([(0,0),(8,0),(1,0),(7,0)])]
    assert not coherent_birth(obs,5)
    assert coherent_birth([(i,np.array([i*4,i]),.9) for i in range(3)],5)


def test_prior_retains_raw_candidates_and_marks_static_rejection():
    frames=[dict(is_court=True,candidates=[(100+i%2,200,.9,0,0)]) for i in range(45)]
    original=[list(m['candidates']) for m in frames]
    track_ball_persistent(frames,fps=30,spatial=1,speed_scale=1,frame_size=(640,360),stationary_prior=True)
    assert [m['candidates'] for m in frames]==original
    assert sum(m.get('stationary_prior_rejected',0) for m in frames)>10
    assert not any(m.get('ball_seen') for m in frames)


def test_moving_track_can_cross_a_known_stationary_location():
    frames=[]
    for i in range(80):
        candidates=[(200.,180.,.7,0,0)]
        if 35<=i<75:
            candidates.append((100.+5*(i-35),180.,.95,0,0))
        frames.append(dict(is_court=True,candidates=candidates))
    track_ball_persistent(frames,fps=30,spatial=1,speed_scale=1,frame_size=(640,360),stationary_prior=True)
    assert frames[55]['ball_seen']
    assert abs(frames[55]['ball_px'][0]-200)<3


def test_brief_pause_and_player_held_ball_are_not_blanket_rejected():
    prior=StationaryPrior(30,1)
    assert not any(prior.update(i,[(100,100)])[0] for i in range(8))
    frames=[dict(is_court=True,candidates=[(100.,100.,.9,0,0)],
                 person_boxes=np.array([[80.,60.,120.,150.]])) for _ in range(40)]
    track_ball_persistent(frames,fps=30,spatial=1,speed_scale=1,frame_size=(640,360),stationary_prior=True)
    assert all(m['stationary_prior_rejected']==0 for m in frames)
