"""Camera registration and management API."""

from urllib.parse import urlsplit, urlunsplit

import cv2
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import Response

from app.core import registry
from app.core.config import settings
from app.schemas.camera import CameraCreate, CameraResponse, CameraUpdate
from app.services.camera_detection import camera_detection_manager
from app.services.camera_manager import camera_manager
from app.services.camera_store import CameraStore

router = APIRouter(prefix="/api/cameras", tags=["Cameras"])


def get_store() -> CameraStore:
    return CameraStore(settings.camera_db_path)


def safe_source(source: str, source_type: str) -> str:
    """Hide RTSP usernames and passwords in API responses."""
    if source_type != "rtsp":
        return source

    try:
        parsed = urlsplit(source)
        if parsed.scheme not in {"rtsp", "rtsps"} or not parsed.hostname:
            return "[configured RTSP source]"

        host = parsed.hostname
        if parsed.port:
            host = f"{host}:{parsed.port}"

        return urlunsplit(
            (parsed.scheme, "***:***@" + host, parsed.path, "", "")
        )
    except ValueError:
        return "[configured RTSP source]"


def serialize_camera(camera: dict) -> dict:
    return {
        **camera,
        "source": safe_source(camera["source"], camera["source_type"]),
        "enabled": bool(camera["enabled"]),
    }


@router.post("", response_model=CameraResponse, status_code=status.HTTP_201_CREATED)
def create_camera(payload: CameraCreate):
    if payload.source_type == "webcam":
        if not payload.source.isdigit():
            raise HTTPException(400, "Webcam source must be a numeric device index")
    else:
        parsed = urlsplit(payload.source)
        if parsed.scheme not in {"rtsp", "rtsps"} or not parsed.hostname:
            raise HTTPException(400, "A valid RTSP or RTSPS URL is required")

    camera = get_store().create_camera(
        payload.name, payload.source_type, payload.source
    )
    return serialize_camera(camera)


@router.get("", response_model=list[CameraResponse])
def list_cameras():
    return [serialize_camera(camera) for camera in get_store().list_cameras()]


@router.get("/{camera_id}", response_model=CameraResponse)
def get_camera(camera_id: int):
    camera = get_store().get_camera(camera_id)
    if camera is None:
        raise HTTPException(404, "Camera not found")
    return serialize_camera(camera)


@router.put("/{camera_id}", response_model=CameraResponse)
def update_camera(camera_id: int, payload: CameraUpdate):
    camera = get_store().update_camera(
        camera_id, payload.name, payload.enabled
    )
    if camera is None:
        raise HTTPException(404, "Camera not found")
    if not camera["enabled"]:
        camera_detection_manager.stop(camera_id)
        camera_manager.disconnect(camera_id)
    return serialize_camera(camera)


@router.delete("/{camera_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_camera(camera_id: int):
    camera_detection_manager.stop(camera_id)
    camera_manager.disconnect(camera_id)
    if not get_store().delete_camera(camera_id):
        raise HTTPException(404, "Camera not found")


@router.post("/{camera_id}/connect")
def connect_camera(camera_id: int):
    camera = get_store().get_camera(camera_id)

    if camera is None:
        raise HTTPException(404, "Camera not found")

    if not camera["enabled"]:
        raise HTTPException(409, "Camera is disabled")

    try:
        session = camera_manager.connect(
            camera_id,
            camera["source"],
            camera["source_type"],
        )
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc

    return {
        "camera_id": camera_id,
        "status": session.status,
    }


@router.post("/{camera_id}/disconnect")
def disconnect_camera(camera_id: int):
    camera = get_store().get_camera(camera_id)

    if camera is None:
        raise HTTPException(404, "Camera not found")

    camera_manager.disconnect(camera_id)

    return {
        "camera_id": camera_id,
        "status": "disconnected",
    }

@router.get("/connections/status")
def camera_connection_status():
    return camera_manager.list_status()


@router.get("/{camera_id}/frame")
def get_camera_frame(camera_id: int):
    camera = get_store().get_camera(camera_id)

    if camera is None:
        raise HTTPException(404, "Camera not found")

    session = camera_manager.get_session(camera_id)

    if session is None:
        raise HTTPException(409, "Camera is not connected")

    frame = session.get_frame()

    if frame is None:
        raise HTTPException(503, "No camera frame available yet")

    success, encoded = cv2.imencode(".jpg", frame)

    if not success:
        raise HTTPException(500, "Could not encode camera frame")

    return Response(
        content=encoded.tobytes(),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


@router.post("/{camera_id}/detection/start")
def start_camera_detection(camera_id: int):
    camera = get_store().get_camera(camera_id)
    if camera is None:
        raise HTTPException(404, "Camera not found")
    if not camera["enabled"]:
        raise HTTPException(409, "Camera is disabled")

    camera_session = camera_manager.get_session(camera_id)
    if camera_session is None:
        raise HTTPException(409, "Camera is not connected")

    detector = registry.yolo_detector
    if detector is None:
        raise HTTPException(503, "YOLO detector is not available")

    try:
        session = camera_detection_manager.start(
            camera_id,
            camera_session,
            detector,
        )
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc

    return session.get_status()


@router.post("/{camera_id}/detection/stop")
def stop_camera_detection(camera_id: int):
    if get_store().get_camera(camera_id) is None:
        raise HTTPException(404, "Camera not found")

    stopped = camera_detection_manager.stop(camera_id)
    return {
        "camera_id": camera_id,
        "status": "stopped" if stopped else "not_running",
    }


@router.get("/{camera_id}/detection/status")
def camera_detection_status(camera_id: int):
    if get_store().get_camera(camera_id) is None:
        raise HTTPException(404, "Camera not found")

    session = camera_detection_manager.get_session(camera_id)
    if session is None:
        return {"camera_id": camera_id, "status": "stopped"}

    return session.get_status()


@router.get("/{camera_id}/detection/results")
def camera_detection_results(camera_id: int):
    if get_store().get_camera(camera_id) is None:
        raise HTTPException(404, "Camera not found")

    session = camera_detection_manager.get_session(camera_id)
    if session is None:
        raise HTTPException(409, "Detection is not running")

    return session.get_results()


@router.get("/{camera_id}/detection/frame")
def camera_detection_frame(camera_id: int):
    if get_store().get_camera(camera_id) is None:
        raise HTTPException(404, "Camera not found")

    session = camera_detection_manager.get_session(camera_id)
    if session is None:
        raise HTTPException(409, "Detection is not running")

    frame = session.get_frame()
    if frame is None:
        raise HTTPException(503, "No processed frame available yet")

    success, encoded = cv2.imencode(".jpg", frame)
    if not success:
        raise HTTPException(500, "Could not encode detection frame")

    return Response(
        content=encoded.tobytes(),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )
