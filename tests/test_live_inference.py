from __future__ import annotations

import threading

import torch

from netcast_tennisvision.streaming.live_inference import LiveInferencePool


def test_shared_live_inference_keeps_single_input_and_coalesces_parallel_streams() -> None:
    calls: list[int] = []
    people_calls: list[int] = []
    loads = 0

    class Ball:
        def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
            calls.append(len(inputs))
            return inputs * 2

    class People:
        def predict(self, frames, **_options):
            people_calls.append(len(frames))
            return frames

    def prepare():
        nonlocal loads
        loads += 1
        return Ball(), People()

    pool = LiveInferencePool(gather_ms=50)
    pool.acquire(prepare, torch.device("cpu"))
    single = torch.tensor([[3.0], [4.0]])
    assert torch.equal(pool.infer_ball(single), single * 2)
    assert calls == [2]

    pool.acquire(prepare, torch.device("cpu"))
    barrier = threading.Barrier(3)
    outputs: list[torch.Tensor | None] = [None, None]

    def infer(index: int) -> None:
        value = torch.full((2, 1), float(index + 1))
        barrier.wait()
        outputs[index] = pool.infer_ball(value)

    workers = [threading.Thread(target=infer, args=(index,)) for index in range(2)]
    for worker in workers:
        worker.start()
    barrier.wait()
    for worker in workers:
        worker.join(timeout=2)
        assert not worker.is_alive()

    assert loads == 1
    assert calls == [2, 4]
    assert torch.equal(outputs[0], torch.full((2, 1), 2.0))
    assert torch.equal(outputs[1], torch.full((2, 1), 4.0))
    assert pool.predict_people([1, 2]) == [1, 2]
    people_barrier = threading.Barrier(3)
    people_outputs: list[list[int] | None] = [None, None]

    def infer_people(index: int) -> None:
        people_barrier.wait()
        people_outputs[index] = pool.predict_people([index * 10, index * 10 + 1])

    people_workers = [
        threading.Thread(target=infer_people, args=(index,)) for index in range(2)
    ]
    for worker in people_workers:
        worker.start()
    people_barrier.wait()
    for worker in people_workers:
        worker.join(timeout=2)
        assert not worker.is_alive()
    assert people_outputs == [[0, 1], [10, 11]]
    assert people_calls == [2, 4]
    assert pool.statistics() == {
        "ball_calls": 2, "ball_frames": 6,
        "person_calls": 2, "person_frames": 6,
    }
    pool.release()
    pool.release()
    assert pool.active_count == 0


def test_three_stream_batches_can_share_one_model_call() -> None:
    calls: list[int] = []

    class Ball:
        def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
            calls.append(len(inputs))
            return inputs

    class People:
        def predict(self, frames, **_options):
            return frames

    pool = LiveInferencePool(max_batch=48, gather_ms=50)
    for _ in range(3):
        pool.acquire(lambda: (Ball(), People()), torch.device("cpu"))
    barrier = threading.Barrier(4)

    def submit() -> None:
        barrier.wait()
        assert len(pool.infer_ball(torch.ones(16, 1))) == 16

    workers = [threading.Thread(target=submit) for _ in range(3)]
    for worker in workers:
        worker.start()
    barrier.wait()
    for worker in workers:
        worker.join(timeout=2)
        assert not worker.is_alive()
    assert calls == [48]
    for _ in range(3):
        pool.release()
