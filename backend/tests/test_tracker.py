from unittest.mock import Mock

from app.core import tracker as tracker_module


def test_tracker_enables_conservative_nms_by_default(monkeypatch):
    mock_deepsort = Mock()
    monkeypatch.setattr(tracker_module, "DeepSort", mock_deepsort)

    tracker_module.ObjectTracker()

    assert mock_deepsort.call_args.kwargs["nms_max_overlap"] == 0.9


def test_tracker_allows_custom_nms_threshold(monkeypatch):
    mock_deepsort = Mock()
    monkeypatch.setattr(tracker_module, "DeepSort", mock_deepsort)

    tracker_module.ObjectTracker(nms_max_overlap=0.8)

    assert mock_deepsort.call_args.kwargs["nms_max_overlap"] == 0.8



def test_tracker_limits_appearance_feature_history(monkeypatch):
    mock_deepsort = Mock()
    monkeypatch.setattr(tracker_module, "DeepSort", mock_deepsort)

    tracker_module.ObjectTracker()

    assert mock_deepsort.call_args.kwargs["nn_budget"] == 100


def test_tracker_uses_custom_embedder(monkeypatch):
    mock_deepsort = Mock()
    mock_embedder = Mock()

    monkeypatch.setattr(tracker_module, "DeepSort", mock_deepsort)
    monkeypatch.setattr(
        tracker_module,
        "MPSMobileNetV2Embedder",
        mock_embedder,
    )

    tracker_module.ObjectTracker()

    assert mock_deepsort.call_args.kwargs["embedder"] is None
    assert (
        mock_deepsort.return_value.embedder
        is mock_embedder.return_value
    )


def test_tracker_preserves_id_during_smooth_movement(monkeypatch):
    import numpy as np

    class FakeEmbedder:
        def predict(self, images):
            return [
                np.ones(1280, dtype=np.float32)
                for _ in images
            ]

    monkeypatch.setattr(
        tracker_module,
        "MPSMobileNetV2Embedder",
        FakeEmbedder,
    )

    tracker = tracker_module.ObjectTracker(n_init=3)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    confirmed_ids = []

    for frame_number in range(20):
        x = 100 + frame_number * 3
        detections = [([x, 100, 80, 200], 0.95, "person")]

        tracks = tracker.update(detections, frame)

        confirmed_ids.extend(
            track.track_id
            for track in tracks
            if track.is_confirmed()
        )

    assert confirmed_ids
    assert len(set(confirmed_ids)) == 1


def test_tracker_preserves_ids_when_two_people_cross(monkeypatch):
    import cv2
    import numpy as np

    class ColorEmbedder:
        def predict(self, images):
            features = []

            for image in images:
                color = image.mean(axis=(0, 1)).astype(np.float32)
                feature = np.zeros(1280, dtype=np.float32)
                feature[:3] = color
                norm = np.linalg.norm(feature)
                features.append(feature / max(norm, 1e-12))

            return features

    monkeypatch.setattr(
        tracker_module,
        "MPSMobileNetV2Embedder",
        ColorEmbedder,
    )

    tracker = tracker_module.ObjectTracker(n_init=3)

    identity_ids = {"red": set(), "green": set()}

    for frame_number in range(31):
        frame = np.zeros((400, 500, 3), dtype=np.uint8)

        red_x = 80 + frame_number * 7
        green_x = 360 - frame_number * 7

        red_box = (red_x, 120, red_x + 50, 190)
        green_box = (green_x, 150, green_x + 50, 220)

        cv2.rectangle(frame, red_box[:2], red_box[2:], (0, 0, 255), -1)
        cv2.rectangle(frame, green_box[:2], green_box[2:], (0, 255, 0), -1)

        detections = [
            ([red_x, 120, 50, 70], 0.95, "person"),
            ([green_x, 150, 50, 70], 0.95, "person"),
        ]

        tracks = tracker.update(detections, frame)
        confirmed = [t for t in tracks if t.is_confirmed()]

        if frame_number < 5:
            continue

        for track in confirmed:
            x1, y1, x2, y2 = track.to_ltrb()
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            red_center = (red_x + 25, 155)
            green_center = (green_x + 25, 185)

            red_distance = (
                (center_x - red_center[0]) ** 2
                + (center_y - red_center[1]) ** 2
            )
            green_distance = (
                (center_x - green_center[0]) ** 2
                + (center_y - green_center[1]) ** 2
            )

            identity = "red" if red_distance < green_distance else "green"
            identity_ids[identity].add(track.track_id)

    print("Red person IDs:", identity_ids["red"])
    print("Green person IDs:", identity_ids["green"])

    assert len(identity_ids["red"]) == 1
    assert len(identity_ids["green"]) == 1
    assert identity_ids["red"] != identity_ids["green"]


def test_tracker_recovers_id_after_short_occlusion(monkeypatch):
    import numpy as np

    class FakeEmbedder:
        def predict(self, images):
            return [
                np.ones(1280, dtype=np.float32)
                for _ in images
            ]

    monkeypatch.setattr(
        tracker_module,
        "MPSMobileNetV2Embedder",
        FakeEmbedder,
    )

    tracker = tracker_module.ObjectTracker(max_age=30, n_init=3)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detection = [([200, 100, 80, 200], 0.95, "person")]

    original_id = None

    for _ in range(6):
        tracks = tracker.update(detection, frame)
        confirmed = [t for t in tracks if t.is_confirmed()]
        if confirmed:
            original_id = confirmed[0].track_id

    assert original_id is not None

    for _ in range(5):
        tracker.update([], frame)

    recovered_ids = []

    for _ in range(5):
        tracks = tracker.update(detection, frame)
        recovered_ids.extend(
            t.track_id for t in tracks if t.is_confirmed()
        )

    assert original_id in recovered_ids
    assert len(set(recovered_ids)) == 1


def test_tracker_recovers_id_after_longer_occlusion(monkeypatch):
    import numpy as np

    class FakeEmbedder:
        def predict(self, images):
            return [
                np.ones(1280, dtype=np.float32)
                for _ in images
            ]

    monkeypatch.setattr(
        tracker_module,
        "MPSMobileNetV2Embedder",
        FakeEmbedder,
    )

    tracker = tracker_module.ObjectTracker(max_age=30, n_init=3)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detection = [([200, 100, 80, 200], 0.95, "person")]

    original_id = None

    for _ in range(6):
        tracks = tracker.update(detection, frame)
        for track in tracks:
            if track.is_confirmed():
                original_id = track.track_id

    assert original_id is not None

    for _ in range(15):
        tracker.update([], frame)

    recovered_ids = []

    for _ in range(5):
        tracks = tracker.update(detection, frame)
        recovered_ids.extend(
            track.track_id
            for track in tracks
            if track.is_confirmed()
        )

    assert original_id in recovered_ids
    assert len(set(recovered_ids)) == 1


def test_tracker_reuses_injected_embedder(monkeypatch):
    from unittest.mock import Mock

    import app.core.tracker as tracker_module

    mock_deepsort = Mock()
    mock_embedder = Mock()
    mock_embedder_factory = Mock()

    monkeypatch.setattr(tracker_module, "DeepSort", mock_deepsort)
    monkeypatch.setattr(
        tracker_module,
        "MPSMobileNetV2Embedder",
        mock_embedder_factory,
    )

    tracker = tracker_module.ObjectTracker(embedder=mock_embedder)

    assert tracker.tracker.embedder is mock_embedder
    mock_embedder_factory.assert_not_called()

    mock_deepsort.assert_called_once()
