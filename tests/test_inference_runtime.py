from inference_runtime import iter_batched_detections, iter_sparse_person_detections


class FakeCapture:
    def __init__(self, frames):
        self.frames = iter(frames)

    def read(self):
        try:
            return True, next(self.frames)
        except StopIteration:
            return False, None


class FakeModel:
    def __init__(self, prefix, fail_at=None):
        self.prefix = prefix
        self.fail_at = fail_at
        self.calls = []

    def predict(self, frames, **_kwargs):
        self.calls.append(len(frames))
        if self.fail_at == len(frames):
            self.fail_at = None
            raise RuntimeError("CUDA out of memory")
        return [f"{self.prefix}{frame}" for frame in frames]


def run(frames, ball, person, batch_size=4):
    return list(iter_batched_detections(
        FakeCapture(frames), ball, person, device="cuda", batch_size=batch_size,
        ball_kwargs={}, person_kwargs={}))


def test_batched_inference_preserves_frame_order_and_tail():
    ball, person = FakeModel("b"), FakeModel("p")
    output = run(list(range(10)), ball, person)
    assert output == [(i, f"b{i}", f"p{i}") for i in range(10)]
    assert ball.calls == [4, 4, 2]
    assert person.calls == [4, 4, 2]


def test_cuda_oom_retries_same_frames_in_smaller_batches():
    ball, person = FakeModel("b", fail_at=4), FakeModel("p")
    output = run(list(range(4)), ball, person)
    assert output == [(i, f"b{i}", f"p{i}") for i in range(4)]
    assert ball.calls == [4, 2, 2]
    assert person.calls == [2, 2]


def test_sparse_person_inference_preserves_every_frame_and_reuses_keyframes():
    person = FakeModel("p")
    output = list(iter_sparse_person_detections(
        FakeCapture(list(range(7))), person, device="cuda", batch_size=2,
        person_kwargs={}, stride=2,
    ))
    assert output == [
        (0, "p0", "p0", 0.0, True), (1, "p0", "p2", 0.5, False),
        (2, "p2", "p2", 0.0, True), (3, "p2", "p4", 0.5, False),
        (4, "p4", "p4", 0.0, True), (5, "p4", "p6", 0.5, False),
        (6, "p6", "p6", 0.0, True),
    ]
    assert sum(person.calls) == 4


def test_sparse_stride_one_matches_full_person_inference():
    person = FakeModel("p")
    output = list(iter_sparse_person_detections(
        FakeCapture(list(range(5))), person, device="cuda", batch_size=3,
        person_kwargs={}, stride=1,
    ))
    assert output == [(i, f"p{i}", f"p{i}", 0.0, True) for i in range(5)]
