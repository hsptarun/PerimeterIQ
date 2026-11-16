import time
from unittest.mock import Mock

import numpy as np
import pytest

from app.schemas.detection import Alert
from app.services.camera_detection import (
    CameraDetectionManager,
    CameraDetectionSession,
)


class FakeCamera:
    def __init__(self, connected=True):
        self.connected = connected
        self.sequence = 1
        self.frame = np.zeros((100, 100, 3), dtype=np.uint8)

    def get_status(self):
        return {"status": "connected" if self.connected else "disconnected"}

    def get_frame_snapshot(self):
        return self.frame.copy(), self.sequence


def wait_until(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_detection_processes_only_new_frames(monkeypatch):
    import app.services.camera_detection as service

    calls = []

    def fake_process_frame(frame, detector, tracker, rules, frame_id, threshold, **kwargs):
        calls.append(frame_id)
        return frame, [], []

    monkeypatch.setattr(service, "process_frame", fake_process_frame)

    session = CameraDetectionSession(
        camera_id=1,
        camera_session=FakeCamera(),
        detector=Mock(),
    )

    session.start()
    assert wait_until(lambda: session.processed_frames == 1)
    time.sleep(0.1)
    assert session.stop()

    assert calls == [0]
    assert session.processed_frames == 1


def test_detection_finalizes_active_alerts(monkeypatch):
    import app.services.camera_detection as service

    alert = Alert(
        frame=0,
        type="zone_intrusion",
        message="Zone intrusion ended",
        event="ended",
        duration_seconds=1.0,
    )

    class FakeRuleEngine:
        def __init__(self, fps):
            self.fps = fps

        def finalize(self, last_frame_id):
            return [alert]

    monkeypatch.setattr(service, "RuleEngine", FakeRuleEngine)
    monkeypatch.setattr(
        service,
        "process_frame",
        lambda frame, *args, **kwargs: (frame, [], []),
    )

    session = CameraDetectionSession(
        camera_id=1,
        camera_session=FakeCamera(),
        detector=Mock(),
    )

    session.start()
    assert wait_until(lambda: session.processed_frames == 1)
    assert session.stop()

    assert list(session.alert_history) == [alert]


def test_detection_error_is_reported(monkeypatch):
    import app.services.camera_detection as service

    def fail_inference(*args, **kwargs):
        raise RuntimeError("Inference failed")

    monkeypatch.setattr(service, "process_frame", fail_inference)

    session = CameraDetectionSession(
        camera_id=1,
        camera_session=FakeCamera(),
        detector=Mock(),
    )

    session.start()
    assert wait_until(lambda: session.get_status()["status"] == "error")

    assert session.get_status()["error"] == "Inference failed"
    assert session.stop()


def test_detection_requires_connected_camera():
    manager = CameraDetectionManager()

    with pytest.raises(ValueError, match="Camera is not connected"):
        manager.start(
            camera_id=1,
            camera_session=FakeCamera(connected=False),
            detector=Mock(),
        )


def test_each_camera_gets_separate_detection_session():
    manager = CameraDetectionManager()

    first = manager.start(1, FakeCamera(), Mock())
    second = manager.start(2, FakeCamera(), Mock())

    assert first is not second
    assert first.camera_id == 1
    assert second.camera_id == 2

    assert manager.stop(1)
    assert manager.stop(2)
