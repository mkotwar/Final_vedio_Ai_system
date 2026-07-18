"""Compatibility adapters, tracker-ID normalization, and packet sinks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .schemas import BoundingBox, DetectionPacket, DetectionRecord, FramePacket, TrackedFramePacket, TrackedObject
from .serialization import dataclass_to_dict, to_json_safe
from .validation import validate_non_empty_string, validate_non_negative_int, validate_positive_int


class TrackIdNormalizer:
    """Normalize tracker IDs while preserving reverse source-ID mapping.

    Native non-negative integer IDs keep their value. String IDs, including
    numeric strings such as "1", are source IDs and receive first-seen allocated
    integers that cannot collide with native integer IDs already observed.
    """

    def __init__(self) -> None:
        self._source_to_normalized: dict[str | int, int] = {}
        self._normalized_to_source: dict[int, str | int] = {}
        self._next_allocated = 1

    def normalize(self, source_track_id: str | int) -> int:
        if isinstance(source_track_id, bool) or not isinstance(source_track_id, (str, int)):
            raise ValueError("source_track_id must be a non-negative int or non-blank string.")
        if isinstance(source_track_id, int):
            validate_non_negative_int(source_track_id, "source_track_id")
            self._remember(source_track_id, source_track_id)
            return source_track_id
        source_id = validate_non_empty_string(source_track_id, "source_track_id")
        if source_id in self._source_to_normalized:
            return self._source_to_normalized[source_id]
        while self._next_allocated in self._normalized_to_source:
            self._next_allocated += 1
        normalized = self._next_allocated
        self._remember(source_id, normalized)
        self._next_allocated += 1
        return normalized

    def get_source_id(self, normalized_track_id: int) -> str | int | None:
        validate_non_negative_int(normalized_track_id, "normalized_track_id")
        return self._normalized_to_source.get(normalized_track_id)

    def reset(self) -> None:
        self._source_to_normalized.clear()
        self._normalized_to_source.clear()
        self._next_allocated = 1

    def _remember(self, source_id: str | int, normalized: int) -> None:
        existing_source = self._normalized_to_source.get(normalized)
        if existing_source is not None and existing_source != source_id:
            raise ValueError(f"Normalized track ID collision for {normalized}: {existing_source!r} vs {source_id!r}.")
        self._source_to_normalized[source_id] = normalized
        self._normalized_to_source[normalized] = source_id


def _require(item: dict[str, Any], field_name: str) -> Any:
    if field_name not in item:
        raise ValueError(f"Missing required field: {field_name}")
    return item[field_name]


def _bbox_from_xyxy(value: Any) -> BoundingBox:
    values = list(value)
    if len(values) != 4:
        raise ValueError("bbox_xyxy must contain exactly four values.")
    return BoundingBox(float(values[0]), float(values[1]), float(values[2]), float(values[3]))


def td_case2_detection_record_to_schema(record: dict[str, Any]) -> DetectionRecord:
    """Convert one Step 03 detection dict into a DetectionRecord."""

    return DetectionRecord(
        bbox=_bbox_from_xyxy(_require(record, "bbox_xyxy")),
        confidence=float(_require(record, "confidence")),
        class_id=int(_require(record, "class_id")),
        class_name=str(_require(record, "class_name")),
    )


def td_case2_frame_group_to_detection_packet(
    frame_group: dict[str, Any],
    *,
    source_id: str,
    frame_width: int,
    frame_height: int,
    frame: Any = None,
) -> DetectionPacket:
    """Convert an existing Step 03 frame group into a DetectionPacket."""

    validate_non_empty_string(source_id, "source_id")
    width = validate_positive_int(frame_width, "frame_width")
    height = validate_positive_int(frame_height, "frame_height")
    detections = [td_case2_detection_record_to_schema(item) for item in list(_require(frame_group, "detections"))]
    return DetectionPacket(
        source_id=source_id,
        frame_index=int(_require(frame_group, "frame_idx")),
        timestamp_sec=float(_require(frame_group, "timestamp_seconds")),
        frame_width=width,
        frame_height=height,
        detections=detections,
        frame=frame,
    )


def td_case2_track_detection_to_tracked_object(
    record: dict[str, Any],
    *,
    normalizer: TrackIdNormalizer,
) -> TrackedObject:
    """Convert a Step 04B tracked detection row into a TrackedObject."""

    source_track_id = _require(record, "track_id")
    normalized_track_id = normalizer.normalize(source_track_id)
    return TrackedObject(
        track_id=normalized_track_id,
        source_track_id=source_track_id,
        bbox=_bbox_from_xyxy(_require(record, "bbox_xyxy")),
        confidence=float(_require(record, "confidence")),
        class_id=int(record.get("class_id", 0)),
        class_name=str(_require(record, "class_name")),
        frame_index=int(_require(record, "frame_idx")),
        timestamp_sec=float(_require(record, "timestamp_seconds")),
    )


def td_case2_track_frame_to_tracked_packet(
    frame_group: dict[str, Any],
    *,
    source_id: str,
    frame_width: int,
    frame_height: int,
    normalizer: TrackIdNormalizer,
    frame: Any = None,
) -> TrackedFramePacket:
    """Convert a frame-level Step 04B-style group with tracked detections."""

    tracks = [
        td_case2_track_detection_to_tracked_object(item, normalizer=normalizer)
        for item in list(_require(frame_group, "tracks"))
    ]
    frame_index = int(_require(frame_group, "frame_idx"))
    timestamp = float(_require(frame_group, "timestamp_seconds"))
    for track in tracks:
        if track.frame_index != frame_index or track.timestamp_sec != timestamp:
            raise ValueError("Tracked frame group metadata does not match nested track.")
    return TrackedFramePacket(
        source_id=validate_non_empty_string(source_id, "source_id"),
        frame_index=frame_index,
        timestamp_sec=timestamp,
        frame_width=validate_positive_int(frame_width, "frame_width"),
        frame_height=validate_positive_int(frame_height, "frame_height"),
        tracks=tracks,
        frame=frame,
    )


def tracked_object_to_step05_detection_dict(track: TrackedObject, *, frame_id: str | None = None) -> dict[str, Any]:
    """Return a Step 05-compatible detection dictionary without calling Step 05."""

    track_source = track.source_track_id if track.source_track_id is not None else track.track_id
    detection_id = f"{track_source}_frame_{track.frame_index}"
    return {
        "track_id": str(track_source),
        "frame_id": frame_id or f"frame_{track.frame_index:06d}",
        "frame_idx": track.frame_index,
        "timestamp_seconds": track.timestamp_sec,
        "detection_id": detection_id,
        "class_name": track.class_name,
        "confidence": round(float(track.confidence), 6),
        "bbox_xyxy": [round(value, 3) for value in track.bbox.to_xyxy()],
        "bbox_area_ratio": 0.0,
        "crop_path": "",
        "crop_exists": False,
    }


class InMemoryPacketSink:
    """Store JSON-safe packet dictionaries in memory."""

    def __init__(self) -> None:
        self.frames: list[dict[str, Any]] = []
        self.detections: list[dict[str, Any]] = []
        self.tracked_frames: list[dict[str, Any]] = []
        self.closed = False

    def write_frame(self, packet: FramePacket) -> None:
        self.frames.append(dataclass_to_dict(packet))

    def write_detection(self, packet: DetectionPacket) -> None:
        self.detections.append(dataclass_to_dict(packet))

    def write_tracked_frame(self, packet: TrackedFramePacket) -> None:
        self.tracked_frames.append(dataclass_to_dict(packet))

    def close(self) -> None:
        self.closed = True


class JsonlPacketSink:
    """Write independent JSONL packet artifacts under an output directory."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._handles = {
            "frames": (self.output_dir / "frame_packets.jsonl").open("w", encoding="utf-8", newline="\n"),
            "detections": (self.output_dir / "detection_packets.jsonl").open("w", encoding="utf-8", newline="\n"),
            "tracked": (self.output_dir / "tracked_frame_packets.jsonl").open("w", encoding="utf-8", newline="\n"),
        }
        self.closed = False

    def write_frame(self, packet: FramePacket) -> None:
        self._write("frames", packet)

    def write_detection(self, packet: DetectionPacket) -> None:
        self._write("detections", packet)

    def write_tracked_frame(self, packet: TrackedFramePacket) -> None:
        self._write("tracked", packet)

    def close(self) -> None:
        if self.closed:
            return
        for handle in self._handles.values():
            handle.flush()
            handle.close()
        self.closed = True

    def __enter__(self) -> "JsonlPacketSink":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _write(self, name: str, packet: Any) -> None:
        if self.closed:
            raise RuntimeError("Cannot write to a closed JsonlPacketSink.")
        import json

        self._handles[name].write(json.dumps(to_json_safe(packet), ensure_ascii=False, sort_keys=True))
        self._handles[name].write("\n")
        self._handles[name].flush()
