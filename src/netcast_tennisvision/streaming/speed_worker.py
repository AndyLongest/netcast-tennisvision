"""Bounded, optional live speed worker. Never blocks the inference submission path."""
from __future__ import annotations

import copy
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from netcast_tennisvision.events.landing_event_detector import detect_landing_impulses
from netcast_tennisvision.tracking.speed import analyze_speeds


class LiveSpeedWorker:
    def __init__(self, publish, *, enabled=True):
        self.publish = publish
        self.enabled = enabled
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="live-speed") if enabled else None
        self.lock = threading.Lock()
        self.busy = False
        self.closed = False
        self.last_time = -10.0
        self.count = 0
        self.total = 0.0
        self.maximum = 0.0
        self.recent = []

    def submit(self, frames, base, fps, frame_size, origin):
        end = len(frames) - 3
        if not self.enabled or end < 10:
            return
        with self.lock:
            if self.closed or self.busy:
                return
            self.busy = True
        start = max(0, end - 16)
        segment = copy.deepcopy(frames[start:end])
        def solve():
            try:
                contacts = detect_landing_impulses(segment, radius=7, min_score=.52)
                result = analyze_speeds(segment, contacts,
                    np.arange(base+start, base+end)/fps, frame_size, method="linear")
                for event in result["estimates"]:
                    with self.lock:
                        if self.closed or event["time_s"] - self.last_time < .3:
                            continue
                        self.last_time = event["time_s"]
                        self.count += 1
                        self.total += event["speed_kmh"]
                        self.maximum = max(self.maximum, event["speed_kmh"])
                        now = time.time()
                        item = dict(event, id=self.count, emitted_at=now,
                                    decision_t=(base+len(frames)-1)/fps,
                                    delay_ms=max(0, (now-origin-event["time_s"])*1000))
                        self.recent = (self.recent + [item])[-30:]
                        payload = dict(enabled=True, method="linear", latest=item,
                            recent=list(self.recent), count=self.count,
                            mean_kmh=self.total/self.count, max_kmh=self.maximum,
                            metric="flight-window midpoint estimate")
                    self.publish(speed=payload)
            except Exception as error:
                with self.lock:
                    closed = self.closed
                if not closed:
                    self.publish(speed_error=type(error).__name__)
            finally:
                with self.lock:
                    self.busy = False
        try:
            self.executor.submit(solve)
        except RuntimeError:
            with self.lock:
                self.busy = False

    def close(self, *, drain=False):
        if not drain:
            with self.lock:
                self.closed = True
        if self.executor:
            self.executor.shutdown(wait=True, cancel_futures=True)
        with self.lock:
            self.closed = True
