from app.services.event_candidate_selector import EventCandidateSelector
from app.services.object_detection.schemas import Detection, FrameDetection
from app.services.object_tracker import ObjectTrackerService


def test_event_candidate_selector_skips_empty_frames():
    tuples = [
        ("f01", "video", 0.0, "a.jpg"),
        ("f02", "video", 1.0, "b.jpg"),
    ]
    frame_detections = [
        FrameDetection(frame_id="f01", video_id="video", timestamp_seconds=0.0, frame_width=100, frame_height=100, detections=[]),
        FrameDetection(frame_id="f02", video_id="video", timestamp_seconds=1.0, frame_width=100, frame_height=100, detections=[]),
    ]
    tracking_map = ObjectTrackerService.track_frames(frame_detections)

    selection = EventCandidateSelector.select(tuples, frame_detections, tracking_map, [])

    assert selection["f01"]["selected"] is False
    assert selection["f02"]["selected"] is False


def test_event_candidate_selector_selects_detection_and_context():
    tuples = [
        ("f01", "video", 0.0, "a.jpg"),
        ("f02", "video", 1.0, "b.jpg"),
    ]
    frame_detections = [
        FrameDetection(frame_id="f01", video_id="video", timestamp_seconds=0.0, frame_width=100, frame_height=100, detections=[]),
        FrameDetection(
            frame_id="f02",
            video_id="video",
            timestamp_seconds=1.0,
            frame_width=100,
            frame_height=100,
            detections=[
                Detection(
                    class_id=0,
                    class_name="person",
                    confidence=0.95,
                    bbox=[0.0, 0.0, 10.0, 10.0],
                    center_x=5.0,
                    center_y=5.0,
                    width=10.0,
                    height=10.0,
                )
            ],
        ),
    ]
    tracking_map = ObjectTrackerService.track_frames(frame_detections)

    selection = EventCandidateSelector.select(tuples, frame_detections, tracking_map, [(1.0, 1.0)])

    assert selection["f02"]["selected"] is True
    assert "dynamic_object_detected" in selection["f02"]["candidate_reasons"]
    assert "motion_window" in selection["f02"]["candidate_reasons"]
    assert selection["f02"]["track_ids"] == [1]


def test_event_candidate_selector_ignores_static_furniture_only_frames():
    tuples = [
        ("f01", "video", 10.0, "a.jpg"),
        ("f02", "video", 11.0, "b.jpg"),
    ]
    frame_detections = [
        FrameDetection(
            frame_id="f01",
            video_id="video",
            timestamp_seconds=10.0,
            frame_width=100,
            frame_height=100,
            detections=[
                Detection(
                    class_id=56,
                    class_name="chair",
                    confidence=0.91,
                    bbox=[0.0, 0.0, 10.0, 10.0],
                    center_x=5.0,
                    center_y=5.0,
                    width=10.0,
                    height=10.0,
                )
            ],
        ),
        FrameDetection(
            frame_id="f02",
            video_id="video",
            timestamp_seconds=11.0,
            frame_width=100,
            frame_height=100,
            detections=[
                Detection(
                    class_id=62,
                    class_name="tv",
                    confidence=0.88,
                    bbox=[0.0, 0.0, 10.0, 10.0],
                    center_x=5.0,
                    center_y=5.0,
                    width=10.0,
                    height=10.0,
                )
            ],
        ),
    ]
    tracking_map = ObjectTrackerService.track_frames(frame_detections)

    selection = EventCandidateSelector.select(tuples, frame_detections, tracking_map, [])

    assert selection["f01"]["selected"] is False
    assert selection["f02"]["selected"] is False
    assert selection["f01"]["detected_objects"] == []
    assert selection["f02"]["detected_objects"] == []


def test_event_candidate_selector_keeps_context_frames_around_window():
    tuples = [
        ("f01", "video", 8.0, "a.jpg"),
        ("f02", "video", 9.0, "b.jpg"),
        ("f03", "video", 10.0, "c.jpg"),
        ("f04", "video", 11.0, "d.jpg"),
    ]
    frame_detections = [
        FrameDetection(frame_id="f01", video_id="video", timestamp_seconds=8.0, frame_width=100, frame_height=100, detections=[]),
        FrameDetection(frame_id="f02", video_id="video", timestamp_seconds=9.0, frame_width=100, frame_height=100, detections=[]),
        FrameDetection(frame_id="f03", video_id="video", timestamp_seconds=10.0, frame_width=100, frame_height=100, detections=[]),
        FrameDetection(frame_id="f04", video_id="video", timestamp_seconds=11.0, frame_width=100, frame_height=100, detections=[]),
    ]
    tracking_map = ObjectTrackerService.track_frames(frame_detections)

    selection = EventCandidateSelector.select(tuples, frame_detections, tracking_map, [(10.0, 10.0)])

    assert selection["f01"]["selected"] is True
    assert "event_context" in selection["f01"]["candidate_reasons"]
    assert selection["f04"]["selected"] is True
    assert "event_context" in selection["f04"]["candidate_reasons"]
