from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.services.camera_manager import camera_manager


def register_camera(client, enabled=True):
    response = client.post(
        "/api/cameras",
        json={
            "name": "Test Camera",
            "source_type": "rtsp",
            "source": "rtsp://user:pass@127.0.0.1:8554/test",
        },
    )
    assert response.status_code == 201

    camera_id = response.json()["id"]

    if not enabled:
        response = client.put(
            f"/api/cameras/{camera_id}",
            json={"name": "Test Camera", "enabled": False},
        )
        assert response.status_code == 200

    return camera_id


def test_connect_registered_camera(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "camera_db_path", tmp_path / "cameras.db")
    monkeypatch.setattr(
        camera_manager,
        "connect",
        lambda camera_id, source, source_type:
            SimpleNamespace(status="connecting"),
    )

    with TestClient(app) as client:
        camera_id = register_camera(client)
        response = client.post(f"/api/cameras/{camera_id}/connect")

        assert response.status_code == 200
        assert response.json()["status"] == "connecting"


def test_disabled_camera_cannot_connect(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "camera_db_path", tmp_path / "cameras.db")

    with TestClient(app) as client:
        camera_id = register_camera(client, enabled=False)
        response = client.post(f"/api/cameras/{camera_id}/connect")

        assert response.status_code == 409


def test_missing_camera_cannot_connect(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "camera_db_path", tmp_path / "cameras.db")

    with TestClient(app) as client:
        response = client.post("/api/cameras/999/connect")

        assert response.status_code == 404


def test_disconnect_camera(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "camera_db_path", tmp_path / "cameras.db")
    disconnected = []

    monkeypatch.setattr(
        camera_manager,
        "disconnect",
        lambda camera_id: disconnected.append(camera_id) or True,
    )

    with TestClient(app) as client:
        camera_id = register_camera(client)
        response = client.post(f"/api/cameras/{camera_id}/disconnect")

        assert response.status_code == 200
        assert response.json()["status"] == "disconnected"
        assert disconnected == [camera_id]


def test_camera_connection_status(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "camera_db_path", tmp_path / "cameras.db")
    monkeypatch.setattr(
        camera_manager,
        "list_status",
        lambda: [{"camera_id": 1, "status": "connected", "error": None}],
    )

    with TestClient(app) as client:
        response = client.get("/api/cameras/connections/status")

        assert response.status_code == 200
        assert response.json()[0]["status"] == "connected"
