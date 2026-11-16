from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app


def test_camera_crud_and_password_masking(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "camera_db_path", tmp_path / "cameras.db")

    with TestClient(app) as client:
        created = client.post(
            "/api/cameras",
            json={
                "name": "Main Entrance",
                "source_type": "rtsp",
                "source": "rtsp://admin:secret123@192.168.1.10:554/stream1",
            },
        )

        assert created.status_code == 201
        camera = created.json()
        camera_id = camera["id"]
        assert camera["name"] == "Main Entrance"
        assert "secret123" not in camera["source"]

        listed = client.get("/api/cameras")
        assert listed.status_code == 200
        assert len(listed.json()) == 1

        fetched = client.get(f"/api/cameras/{camera_id}")
        assert fetched.status_code == 200

        updated = client.put(
            f"/api/cameras/{camera_id}",
            json={"name": "Updated Entrance", "enabled": False},
        )
        assert updated.status_code == 200
        assert updated.json()["name"] == "Updated Entrance"
        assert updated.json()["enabled"] is False

        deleted = client.delete(f"/api/cameras/{camera_id}")
        assert deleted.status_code == 204

        missing = client.get(f"/api/cameras/{camera_id}")
        assert missing.status_code == 404


def test_invalid_camera_sources_are_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "camera_db_path", tmp_path / "cameras.db")

    with TestClient(app) as client:
        webcam = client.post(
            "/api/cameras",
            json={"name": "Webcam", "source_type": "webcam", "source": "invalid"},
        )
        assert webcam.status_code == 400

        rtsp = client.post(
            "/api/cameras",
            json={"name": "IP Camera", "source_type": "rtsp", "source": "https://example.com"},
        )
        assert rtsp.status_code == 400
