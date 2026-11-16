"""Manage camera capture workers and their lifecycle."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class CameraSession:
    camera_id: int
    source: str
    source_type: str
    status: str = "connecting"
    error: str | None = None
    latest_frame: np.ndarray | None = None
    frame_sequence: int = 0

    _stop_event: threading.Event = field(default_factory=threading.Event)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _capture: cv2.VideoCapture | None = None
    _thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._capture_loop,
            name=f"camera-{self.camera_id}",
            daemon=True,
        )
        self._thread.start()

    def _set_status(self, status: str, error: str | None = None) -> None:
        with self._lock:
            self.status = status
            self.error = error

    def _capture_loop(self) -> None:
        capture = None

        try:
            if self.source_type == "webcam":
                capture = cv2.VideoCapture(int(self.source))
            else:
                capture = cv2.VideoCapture(
                    self.source,
                    cv2.CAP_FFMPEG,
                    [
                        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
                        cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000,
                    ],
                )

            self._capture = capture

            if not capture.isOpened():
                self._set_status("error", "Could not open camera source")
                return

            self._set_status("connected")

            while not self._stop_event.is_set():
                ok, frame = capture.read()

                if not ok:
                    if not self._stop_event.is_set():
                        self._set_status(
                            "error",
                            "Camera stream stopped or frame read failed",
                        )
                    break

                with self._lock:
                    self.latest_frame = frame
                    self.frame_sequence += 1

        except Exception as exc:
            self._set_status("error", str(exc))

        finally:
            if capture is not None:
                capture.release()

            self._capture = None

            if self._stop_event.is_set():
                self._set_status("disconnected")

    def get_frame(self) -> np.ndarray | None:
        with self._lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def get_frame_snapshot(self) -> tuple[np.ndarray | None, int]:
        """Return the latest frame and its sequence number."""
        with self._lock:
            if self.latest_frame is None:
                return None, self.frame_sequence
            return self.latest_frame.copy(), self.frame_sequence

    def get_status(self) -> dict:
        with self._lock:
            return {
                "camera_id": self.camera_id,
                "status": self.status,
                "error": self.error,
            }

    def stop(self) -> None:
        self._stop_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=7)

        if self._thread and self._thread.is_alive():
            self._set_status(
                "error",
                "Camera worker did not stop within 7 seconds",
            )
        else:
            self._set_status("disconnected")


class CameraManager:
    def __init__(self) -> None:
        self._sessions: dict[int, CameraSession] = {}
        self._lock = threading.Lock()

    def connect(
        self,
        camera_id: int,
        source: str,
        source_type: str,
    ) -> CameraSession:
        with self._lock:
            existing = self._sessions.get(camera_id)

            if existing and existing.status in {"connecting", "connected"}:
                raise ValueError("Camera is already connected or connecting")

            session = CameraSession(
                camera_id=camera_id,
                source=source,
                source_type=source_type,
            )
            self._sessions[camera_id] = session
            session.start()
            return session

    def disconnect(self, camera_id: int) -> bool:
        with self._lock:
            session = self._sessions.get(camera_id)

        if session is None:
            return False

        session.stop()
        stopped = session.get_status()["status"] == "disconnected"

        if stopped:
            with self._lock:
                if self._sessions.get(camera_id) is session:
                    self._sessions.pop(camera_id)

        return stopped

    def get_session(self, camera_id: int) -> CameraSession | None:
        with self._lock:
            return self._sessions.get(camera_id)

    def list_status(self) -> list[dict]:
        with self._lock:
            sessions = list(self._sessions.values())

        return [session.get_status() for session in sessions]

    def disconnect_all(self) -> None:
        with self._lock:
            camera_ids = list(self._sessions)

        for camera_id in camera_ids:
            self.disconnect(camera_id)


camera_manager = CameraManager()
