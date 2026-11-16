import threading
import time
from concurrent.futures import ThreadPoolExecutor

from app.core.detector import YOLODetector


class FakeModel:
    def __init__(self):
        self.active_calls = 0
        self.max_active_calls = 0
        self.lock = threading.Lock()

    def __call__(self, frame, **kwargs):
        with self.lock:
            self.active_calls += 1
            self.max_active_calls = max(
                self.max_active_calls,
                self.active_calls,
            )

        time.sleep(0.03)

        with self.lock:
            self.active_calls -= 1

        return [frame]


def test_concurrent_inference_is_serialized():
    detector = YOLODetector("unused.pt")
    fake_model = FakeModel()
    detector.model = fake_model

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(
            executor.map(detector.detect, range(8))
        )

    assert len(results) == 8
    assert fake_model.max_active_calls == 1
