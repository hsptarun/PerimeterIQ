import numpy as np

from app.services.pipeline import process_frame


class FakeDetector:
    def detect(self, frame, confidence_threshold):
        return type("Result", (), {"boxes": [], "names": {}})()


class FakeTrack:
    def __init__(self, confidence, class_name="person", track_id=1):
        self.confidence = confidence
        self.class_name = class_name
        self.track_id = track_id

    def is_confirmed(self):
        return True

    def get_det_conf(self):
        return self.confidence

    def to_ltrb(self):
        return [10, 20, 100, 200]

    def get_det_class(self):
        return self.class_name


class FakeTracker:
    def __init__(self, track):
        self.track = track
        self.received = None

    def update(self, detections, frame):
        self.received = detections
        return [self.track]


class FakeRuleEngine:
    def process_frame(self, frame_id, detections, width, height):
        return []


def test_stale_track_without_current_detection_is_excluded():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    annotated, detections, alerts = process_frame(
        frame,
        FakeDetector(),
        FakeTracker(FakeTrack(None)),
        FakeRuleEngine(),
        frame_id=1,
    )

    assert annotated is frame
    assert detections == []
    assert alerts == []


def test_track_with_current_detection_keeps_real_confidence():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    _, detections, _ = process_frame(
        frame,
        FakeDetector(),
        FakeTracker(FakeTrack(0.83)),
        FakeRuleEngine(),
        frame_id=1,
    )

    assert len(detections) == 1
    assert detections[0].confidence == 0.83
    assert detections[0].class_name == "person"


def test_people_and_abandonable_objects_use_separate_trackers():
    from types import SimpleNamespace

    class DetectionDetector:
        def detect(self, frame, confidence_threshold):
            boxes = [
                (0, 0.91, [10, 20, 100, 200]),
                (24, 0.88, [110, 20, 180, 120]),
                (26, 0.87, [190, 20, 250, 120]),
                (28, 0.86, [260, 20, 320, 120]),
                (43, 0.95, [30, 30, 60, 90]),
                (2, 0.90, [40, 40, 120, 100]),
            ]
            return SimpleNamespace(
                boxes=[
                    SimpleNamespace(
                        cls=[class_id],
                        conf=[confidence],
                        xyxy=[coordinates],
                    )
                    for class_id, confidence, coordinates in boxes
                ],
                names={
                    0: "person",
                    24: "backpack",
                    26: "handbag",
                    28: "suitcase",
                    43: "knife",
                    2: "car",
                },
            )

    person_tracker = FakeTracker(FakeTrack(0.91, "person", 1))
    object_tracker = FakeTracker(FakeTrack(0.88, "backpack", 1))

    _, detections, _ = process_frame(
        np.zeros((240, 320, 3), dtype=np.uint8),
        DetectionDetector(),
        person_tracker,
        FakeRuleEngine(),
        frame_id=1,
        draw=False,
        object_tracker=object_tracker,
    )

    assert [item[2] for item in person_tracker.received] == ["person"]
    assert {item[2] for item in object_tracker.received} == {
        "backpack", "handbag", "suitcase"
    }
    assert all(item[2] not in {"knife", "car"} for item in object_tracker.received)

    person_detection = next(d for d in detections if d.class_name == "person")
    object_detection = next(d for d in detections if d.class_name == "backpack")

    assert person_detection.track_id == 1
    assert object_detection.track_id == -1
