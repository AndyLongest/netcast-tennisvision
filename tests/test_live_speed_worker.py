import threading

from netcast_tennisvision.streaming.speed_worker import LiveSpeedWorker


def test_worker_bounded_and_publishes(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    import netcast_tennisvision.streaming.speed_worker as module
    monkeypatch.setattr(module, "detect_landing_impulses", lambda *a, **k: [])
    calls = []
    def analyze(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return {"estimates": [{"time_s": 1., "speed_kmh": 60.}]}
    monkeypatch.setattr(module, "analyze_speeds", analyze)
    worker = LiveSpeedWorker(lambda **value: calls.append(value))
    frames = [{} for _ in range(20)]
    worker.submit(frames, 0, 30, (640,360), 0)
    assert entered.wait(5)
    worker.submit(frames, 0, 30, (640,360), 0)
    release.set()
    worker.close(drain=True)
    assert len(calls) == 1
    assert calls[0]['speed']['count'] == 1
    assert calls[0]['speed']['latest']['speed_kmh'] == 60
    assert frames == [{} for _ in range(20)]


def test_disabled_worker_never_submits():
    worker = LiveSpeedWorker(lambda **k: None, enabled=False)
    worker.submit([{}]*20,0,30,(640,360),0)
    assert worker.executor is None
    worker.close()
