"""Manage independent AI detection workers for connected cameras."""

from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np

from app.core.detector import YOLODetector
from app.core.mps_embedder import MPSMobileNetV2Embedder
from app.core.rules import RuleEngine
from app.core.tracker import ObjectTracker
from app.schemas.detection import Alert, Detection
from app.services.camera_manager import CameraSession
from app.services.pipeline import process_frame


class CameraDetectionSession:
    def __init__(
        self,
        camera_id: int,
        camera_session: CameraSession,
        detector: YOLODetector,
        confidence_threshold: float = 0.4,
        target_fps: float = 5.0,
    ) -> None:
        self.camera_id = camera_id
        self.camera_session = camera_session
        self.detector = detector
        self.confidence_threshold = confidence_threshold
        self.target_fps = target_fps

        self.status = "starting"
        self.error: str | None = None
        self.latest_frame: np.ndarray | None = None
        self.latest_detections: list[Detection] = []
        self.latest_alerts: list[Alert] = []
        self.alert_history: deque[Alert] = deque(maxlen=500)
        self.processed_frames = 0

        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run,
            name=f"detection-{self.camera_id}",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        shared_embedder = MPSMobileNetV2Embedder()
        tracker = ObjectTracker(embedder=shared_embedder)
        object_tracker = ObjectTracker(embedder=shared_embedder)
        rule_engine = RuleEngine(fps=self.target_fps)
        last_sequence = 0
        frame_id = 0
        interval = 1.0 / self.target_fps

        with self._lock:
            self.status = "running"

        try:
            while not self._stop_event.is_set():
                started_at = time.monotonic()

                frame, sequence = self.camera_session.get_frame_snapshot()

                if frame is None or sequence <= last_sequence:
                    self._stop_event.wait(0.02)
                    continue

                last_sequence = sequence

                annotated, detections, alerts = process_frame(
                    frame,
                    self.detector,
                    tracker,
                    rule_engine,
                    frame_id,
                    self.confidence_threshold,
                    object_tracker=object_tracker,
                )

                frame_id += 1

                with self._lock:
                    self.latest_frame = annotated
                    self.latest_detections = detections
                    self.latest_alerts = alerts
                    self.alert_history.extend(alerts)
                    self.processed_frames = frame_id

                remaining = interval - (time.monotonic() - started_at)
                if remaining > 0:
                    self._stop_event.wait(remaining)

        except Exception as exc:
            with self._lock:
                self.status = "error"
                self.error = str(exc)

        finally:
            if frame_id > 0:
                final_alerts = rule_engine.finalize(frame_id - 1)
                with self._lock:
                    self.latest_alerts = final_alerts
                    self.alert_history.extend(final_alerts)

            with self._lock:
                if self.status != "error":
                    self.status = "stopped"

    def get_status(self) -> dict:
        with self._lock:
            return {
                "camera_id": self.camera_id,
                "status": self.status,
                "error": self.error,
                "processed_frames": self.processed_frames,
                "detection_fps_target": self.target_fps,
            }

    def get_frame(self) -> np.ndarray | None:
        with self._lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def get_results(self) -> dict:
        with self._lock:
            return {
                "camera_id": self.camera_id,
                "detections": list(self.latest_detections),
                "alerts": list(self.alert_history),
                "processed_frames": self.processed_frames,
            }

    def stop(self) -> bool:
        self._stop_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=7)

        return not (self._thread and self._thread.is_alive())


class CameraDetectionManager:
    def __init__(self) -> None:
        self._sessions: dict[int, CameraDetectionSession] = {}
        self._lock = threading.Lock()

    def start(
        self,
        camera_id: int,
        camera_session: CameraSession,
        detector: YOLODetector,
        confidence_threshold: float = 0.4,
    ) -> CameraDetectionSession:
        if camera_session.get_status()["status"] != "connected":
            raise ValueError("Camera is not connected")

        with self._lock:
            existing = self._sessions.get(camera_id)

            if existing and existing.get_status()["status"] in {
                "starting",
                "running",
            }:
                raise ValueError("Detection is already running")

            session = CameraDetectionSession(
                camera_id=camera_id,
                camera_session=camera_session,
                detector=detector,
                confidence_threshold=confidence_threshold,
            )

            self._sessions[camera_id] = session
            session.start()
            return session

    def stop(self, camera_id: int) -> bool:
        with self._lock:
            session = self._sessions.get(camera_id)

        if session is None:
            return False

        stopped = session.stop()

        if stopped:
            with self._lock:
                if self._sessions.get(camera_id) is session:
                    self._sessions.pop(camera_id)

        return stopped

    def get_session(self, camera_id: int) -> CameraDetectionSession | None:
        with self._lock:
            return self._sessions.get(camera_id)

    def stop_all(self) -> None:
        with self._lock:
            camera_ids = list(self._sessions)

        for camera_id in camera_ids:
            self.stop(camera_id)


camera_detection_manager = CameraDetectionManager()
