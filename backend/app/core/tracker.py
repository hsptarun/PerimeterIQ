"""Thin wrapper around DeepSORT so callers don't depend on its API directly."""

from __future__ import annotations

from deep_sort_realtime.deepsort_tracker import DeepSort

from app.core.mps_embedder import MPSMobileNetV2Embedder


class ObjectTracker:
    def __init__(
        self,
        max_age: int = 30,
        n_init: int = 3,
        max_iou_distance: float = 0.7,
        nms_max_overlap: float = 0.9,
        nn_budget: int = 100,
        embedder=None,
    ):
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_iou_distance=max_iou_distance,
            nms_max_overlap=nms_max_overlap,
            nn_budget=nn_budget,
            embedder=None,
        )

        self.tracker.embedder = (
            embedder if embedder is not None else MPSMobileNetV2Embedder()
        )

    def update(self, detections: list, frame):
        """Update tracks using detections in [x, y, w, h] format."""
        return self.tracker.update_tracks(detections, frame=frame)
