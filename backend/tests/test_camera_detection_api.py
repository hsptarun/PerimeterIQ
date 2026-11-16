from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient

from app.core import registry
from app.core.config import settings
from app.main import app
from app.services.camera_detection import camera_detection_manager
from app.services.camera_manager import camera_manager


def register_camera(client):
    response = client.post(
        "/api/cameras",
        json={
            "name": "Detection Test Camera",
            "source_type": "webcam",
            "source": "0",
        },
    )
    assert response.status_code == 201
    return response.json()["id"]


def test_start_detection_successfully(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "camera_db_path", tmp_path / "cameras.db")
    monkeypatch.setattr(registry, "yolo_detector", object())
    monkeypatch.setattr(
        camera_manager,
        "get_session",
        lambda camera_id: SimpleNamespace(
            get_status=lambda: {"status": "connected"}
        ),
    )

    monkeypatch.setattr(
        camera_detection_manager,
        "start",
        lambda *args: SimpleNamespace(
            get_status=lambda: {
                "camera_id": args[0],
                "status": "running",
            }
        ),
    )

    with TestClient(app) as client:
        camera_id = register_camera(client)
        response = client.post(
            f"/api/cameras/{camera_id}/detection/start"
        )

        assert response.status_code == 200
        assert response.json()["status"] == "running"


def test_detection_rejected_when_camera_disconnected(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "camera_db_path", tmp_path / "cameras.db")
    monkeypatch.setattr(
        camera_manager,
        "get_session",
        lambda camera_id: None,
    )

    with TestClient(app) as client:
        camera_id = register_camera(client)
        response = client.post(
            f"/api/cameras/{camera_id}/detection/start"
        )

        assert response.status_code == 409
        assert response.json()["detail"] == "Camera is not connected"


def test_detection_rejected_when_model_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "camera_db_path", tmp_path / "cameras.db")
    monkeypatch.setattr(registry, "yolo_detector", None)
    monkeypatch.setattr(
        camera_manager,
        "get_session",
        lambda camera_id: SimpleNamespace(
            get_status=lambda: {"status": "connected"}
        ),
    )

    with TestClient(app) as client:
        camera_id = register_camera(client)
        response = client.post(
            f"/api/cameras/{camera_id}/detection/start"
        )

        assert response.status_code == 503


def test_detection_status_results_and_frame(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "camera_db_path", tmp_path / "cameras.db")

    detection_session = SimpleNamespace(
        get_status=lambda: {
            "camera_id": 1,
            "status": "running",
            "processed_frames": 10,
        },
        get_results=lambda: {
            "camera_id": 1,
            "detections": [],
            "alerts": [],
            "processed_frames": 10,
        },
        get_frame=lambda: np.zeros((100, 100, 3), dtype=np.uint8),
    )

    monkeypatch.setattr(
        camera_detection_manager,
        "get_session",
        lambda camera_id: detection_session,
    )

    with TestClient(app) as client:
        camera_id = register_camera(client)

        status_response = client.get(
            f"/api/cameras/{camera_id}/detection/status"
        )
        assert status_response.status_code == 200
        assert status_response.json()["status"] == "running"

        results_response = client.get(
            f"/api/cameras/{camera_id}/detection/results"
        )
        assert results_response.status_code == 200
        assert results_response.json()["processed_frames"] == 10

        frame_response = client.get(
            f"/api/cameras/{camera_id}/detection/frame"
        )
        assert frame_response.status_code == 200
        assert frame_response.headers["content-type"] == "image/jpeg"


def test_stop_detection(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "camera_db_path", tmp_path / "cameras.db")

    stopped = []
    monkeypatch.setattr(
        camera_detection_manager,
        "stop",
        lambda camera_id: stopped.append(camera_id) or True,
    )

    with TestClient(app) as client:
        camera_id = register_camera(client)
        response = client.post(
            f"/api/cameras/{camera_id}/detection/stop"
        )

        assert response.status_code == 200
        assert response.json()["status"] == "stopped"
        assert stopped == [camera_id]
