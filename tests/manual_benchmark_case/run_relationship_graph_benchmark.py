import json
import math
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT_PATH = SCRIPT_PATH.parents[2]
OUTPUT_ROOT = PROJECT_ROOT_PATH / "tests" / "manual_benchmark_case" / "data" / "output"

ENTITY_TIMELINES_PATH = OUTPUT_ROOT / "entity_timelines.json"
RELATIONSHIP_GRAPH_PATH = OUTPUT_ROOT / "relationship_graph.json"

NEAR_PERSON_DISTANCE = 120.0
MEETING_DISTANCE = 90.0
FOLLOWING_DISTANCE = 180.0
NEAR_OBJECT_DISTANCE = 110.0
TOUCH_OBJECT_DISTANCE = 65.0
NEAR_VEHICLE_DISTANCE = 140.0
TOUCH_VEHICLE_DISTANCE = 80.0
STATIONARY_SPEED = 8.0
MOVING_SPEED = 10.0
MAX_CLUSTER_GAP_SECONDS = 2.0


def _format_timestamp(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


def _load_timelines() -> Dict[str, Any]:
    if not ENTITY_TIMELINES_PATH.exists():
        raise FileNotFoundError(
            f"Missing Phase 1 input: {ENTITY_TIMELINES_PATH}. "
            "Run run_entity_timeline_benchmark.py first."
        )
    payload = json.loads(ENTITY_TIMELINES_PATH.read_text(encoding="utf-8"))
    timelines = payload.get("entity_timelines")
    if not isinstance(timelines, list):
        raise ValueError("entity_timelines.json does not contain an entity_timelines list.")
    return payload


def _center(observation: Dict[str, Any]) -> Tuple[float, float]:
    centroid = observation.get("centroid", {})
    return float(centroid.get("x", 0.0)), float(centroid.get("y", 0.0))


def _bbox(observation: Dict[str, Any]) -> List[float]:
    return [float(value) for value in observation.get("bbox", [0.0, 0.0, 0.0, 0.0])]


def _distance(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    ax, ay = _center(a)
    bx, by = _center(b)
    return math.hypot(ax - bx, ay - by)


def _bbox_overlap(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    ax1, ay1, ax2, ay2 = _bbox(a)
    bx1, by1, bx2, by2 = _bbox(b)
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = min(area_a, area_b)
    return inter_area / denom if denom > 0.0 else 0.0


def _observation_lookup(timeline: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {observation["frame_id"]: observation for observation in timeline.get("observations", [])}


def _track_ref(timeline: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "track_id": timeline.get("track_id"),
        "entity_type": timeline.get("entity_type"),
        "class_name": timeline.get("class_name"),
    }


def _observation_speed(timeline: Dict[str, Any], frame_id: str) -> float:
    positions = timeline.get("centroid_positions", [])
    for index, position in enumerate(positions):
        if position.get("frame_id") != frame_id or index == 0:
            continue
        previous = positions[index - 1]
        elapsed = float(position.get("timestamp_seconds", 0.0)) - float(previous.get("timestamp_seconds", 0.0))
        if elapsed <= 0.0:
            return 0.0
        return math.hypot(
            float(position.get("x", 0.0)) - float(previous.get("x", 0.0)),
            float(position.get("y", 0.0)) - float(previous.get("y", 0.0)),
        ) / elapsed
    return 0.0


def _movement_vector(timeline: Dict[str, Any], frame_id: str) -> Tuple[float, float]:
    positions = timeline.get("centroid_positions", [])
    for index, position in enumerate(positions):
        if position.get("frame_id") != frame_id or index == 0:
            continue
        previous = positions[index - 1]
        return (
            float(position.get("x", 0.0)) - float(previous.get("x", 0.0)),
            float(position.get("y", 0.0)) - float(previous.get("y", 0.0)),
        )
    return 0.0, 0.0


def _same_direction(a_vector: Tuple[float, float], b_vector: Tuple[float, float]) -> bool:
    a_len = math.hypot(*a_vector)
    b_len = math.hypot(*b_vector)
    if a_len <= 0.0 or b_len <= 0.0:
        return False
    dot = a_vector[0] * b_vector[0] + a_vector[1] * b_vector[1]
    return (dot / (a_len * b_len)) >= 0.75


def _is_following(
    leader_observation: Dict[str, Any],
    follower_observation: Dict[str, Any],
    leader_vector: Tuple[float, float],
) -> bool:
    leader_len = math.hypot(*leader_vector)
    if leader_len <= 0.0:
        return False
    lx, ly = _center(leader_observation)
    fx, fy = _center(follower_observation)
    vector_to_follower = (fx - lx, fy - ly)
    return (vector_to_follower[0] * leader_vector[0] + vector_to_follower[1] * leader_vector[1]) < 0.0


def _cluster_signals(
    signals: List[Dict[str, Any]],
    relation_type: str,
    source: Dict[str, Any],
    target: Dict[str, Any],
    base_confidence: float,
) -> List[Dict[str, Any]]:
    if not signals:
        return []

    ordered = sorted(signals, key=lambda item: item["timestamp_seconds"])
    clusters: List[List[Dict[str, Any]]] = []
    current = [ordered[0]]
    for signal in ordered[1:]:
        previous = current[-1]
        if signal["timestamp_seconds"] - previous["timestamp_seconds"] <= MAX_CLUSTER_GAP_SECONDS:
            current.append(signal)
        else:
            clusters.append(current)
            current = [signal]
    clusters.append(current)

    relationships = []
    for cluster in clusters:
        start = cluster[0]
        end = cluster[-1]
        distances = [item["distance_pixels"] for item in cluster if item.get("distance_pixels") is not None]
        confidence = min(0.99, base_confidence + (0.04 * min(len(cluster), 5)))
        if distances:
            confidence = min(0.99, confidence + max(0.0, (NEAR_PERSON_DISTANCE - min(distances)) / NEAR_PERSON_DISTANCE) * 0.08)
        relationships.append(
            {
                "relationship_type": relation_type,
                "source": source,
                "target": target,
                "start": {
                    "frame_id": start["frame_id"],
                    "timestamp_seconds": start["timestamp_seconds"],
                    "timestamp_human": _format_timestamp(start["timestamp_seconds"]),
                },
                "end": {
                    "frame_id": end["frame_id"],
                    "timestamp_seconds": end["timestamp_seconds"],
                    "timestamp_human": _format_timestamp(end["timestamp_seconds"]),
                },
                "confidence": round(confidence, 3),
                "evidence": {
                    "frame_ids": [item["frame_id"] for item in cluster],
                    "sample_count": len(cluster),
                    "distance_pixels": {
                        "min": round(min(distances), 2) if distances else None,
                        "avg": round(mean(distances), 2) if distances else None,
                        "max": round(max(distances), 2) if distances else None,
                    },
                    "observations": cluster[:10],
                },
            }
        )
    return relationships


def _distance_relationship(a: Dict[str, Any], b: Dict[str, Any], common_frames: List[str]) -> Optional[Dict[str, Any]]:
    if not common_frames:
        return None
    a_lookup = _observation_lookup(a)
    b_lookup = _observation_lookup(b)
    samples = []
    for frame_id in common_frames:
        obs_a = a_lookup[frame_id]
        obs_b = b_lookup[frame_id]
        samples.append(
            {
                "frame_id": frame_id,
                "timestamp_seconds": float(obs_a.get("timestamp_seconds", 0.0)),
                "distance_pixels": _distance(obs_a, obs_b),
                "source_zone": obs_a.get("zone"),
                "target_zone": obs_b.get("zone"),
            }
        )
    return _cluster_signals(samples, "distance", _track_ref(a), _track_ref(b), 0.55)[0]


def _person_person_relationships(a: Dict[str, Any], b: Dict[str, Any]) -> List[Dict[str, Any]]:
    a_lookup = _observation_lookup(a)
    b_lookup = _observation_lookup(b)
    common_frames = sorted(set(a_lookup).intersection(b_lookup))
    relationships = []
    distance_rel = _distance_relationship(a, b, common_frames)
    if distance_rel is not None:
        relationships.append(distance_rel)

    meeting = []
    walking = []
    following_ab = []
    following_ba = []
    for frame_id in common_frames:
        obs_a = a_lookup[frame_id]
        obs_b = b_lookup[frame_id]
        distance = _distance(obs_a, obs_b)
        signal = {
            "frame_id": frame_id,
            "timestamp_seconds": float(obs_a.get("timestamp_seconds", 0.0)),
            "distance_pixels": distance,
            "source_zone": obs_a.get("zone"),
            "target_zone": obs_b.get("zone"),
        }
        if distance <= MEETING_DISTANCE:
            meeting.append(signal)

        a_speed = _observation_speed(a, frame_id)
        b_speed = _observation_speed(b, frame_id)
        a_vector = _movement_vector(a, frame_id)
        b_vector = _movement_vector(b, frame_id)
        if distance <= NEAR_PERSON_DISTANCE and a_speed >= MOVING_SPEED and b_speed >= MOVING_SPEED and _same_direction(a_vector, b_vector):
            walking.append({**signal, "source_speed": a_speed, "target_speed": b_speed})
        if distance <= FOLLOWING_DISTANCE and a_speed >= MOVING_SPEED and b_speed >= MOVING_SPEED and _same_direction(a_vector, b_vector):
            if _is_following(obs_a, obs_b, a_vector):
                following_ab.append({**signal, "leader_track_id": a["track_id"], "follower_track_id": b["track_id"]})
            if _is_following(obs_b, obs_a, b_vector):
                following_ba.append({**signal, "leader_track_id": b["track_id"], "follower_track_id": a["track_id"]})

    relationships.extend(_cluster_signals(meeting, "meeting", _track_ref(a), _track_ref(b), 0.68))
    relationships.extend(_cluster_signals(walking, "walking_together", _track_ref(a), _track_ref(b), 0.7))
    relationships.extend(_cluster_signals(following_ab, "following", _track_ref(b), _track_ref(a), 0.66))
    relationships.extend(_cluster_signals(following_ba, "following", _track_ref(a), _track_ref(b), 0.66))
    return relationships


def _person_object_relationships(person: Dict[str, Any], obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    person_lookup = _observation_lookup(person)
    object_lookup = _observation_lookup(obj)
    common_frames = sorted(set(person_lookup).intersection(object_lookup))
    near = []
    touch = []
    leave = []
    pick = []

    previous_near = False
    previous_touch = False
    previous_distance: Optional[float] = None
    for frame_id in common_frames:
        p_obs = person_lookup[frame_id]
        o_obs = object_lookup[frame_id]
        distance = _distance(p_obs, o_obs)
        overlap = _bbox_overlap(p_obs, o_obs)
        object_speed = _observation_speed(obj, frame_id)
        signal = {
            "frame_id": frame_id,
            "timestamp_seconds": float(p_obs.get("timestamp_seconds", 0.0)),
            "distance_pixels": distance,
            "bbox_overlap_ratio": round(overlap, 3),
            "source_zone": p_obs.get("zone"),
            "target_zone": o_obs.get("zone"),
            "object_speed_pixels_per_second": round(object_speed, 2),
        }
        is_near = distance <= NEAR_OBJECT_DISTANCE
        is_touch = distance <= TOUCH_OBJECT_DISTANCE or overlap > 0.05
        if is_near:
            near.append(signal)
        if is_touch:
            touch.append(signal)
        if previous_near and not is_near and previous_distance is not None and distance > previous_distance:
            leave.append(signal)
        if previous_touch and object_speed >= MOVING_SPEED:
            pick.append(signal)
        previous_near = is_near
        previous_touch = is_touch
        previous_distance = distance

    relationships = []
    relationships.extend(_cluster_signals(near, "near_object", _track_ref(person), _track_ref(obj), 0.62))
    relationships.extend(_cluster_signals(touch, "touch_object", _track_ref(person), _track_ref(obj), 0.72))
    relationships.extend(_cluster_signals(leave, "leave_object", _track_ref(person), _track_ref(obj), 0.61))
    relationships.extend(_cluster_signals(pick, "pick_object", _track_ref(person), _track_ref(obj), 0.58))
    return relationships


def _person_vehicle_relationships(person: Dict[str, Any], vehicle: Dict[str, Any], objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    person_lookup = _observation_lookup(person)
    vehicle_lookup = _observation_lookup(vehicle)
    common_frames = sorted(set(person_lookup).intersection(vehicle_lookup))
    near_vehicle_frames = []
    for frame_id in common_frames:
        p_obs = person_lookup[frame_id]
        v_obs = vehicle_lookup[frame_id]
        distance = _distance(p_obs, v_obs)
        if distance <= NEAR_VEHICLE_DISTANCE or _bbox_overlap(p_obs, v_obs) > 0.02:
            near_vehicle_frames.append(
                {
                    "frame_id": frame_id,
                    "timestamp_seconds": float(p_obs.get("timestamp_seconds", 0.0)),
                    "distance_pixels": distance,
                    "bbox_overlap_ratio": round(_bbox_overlap(p_obs, v_obs), 3),
                    "source_zone": p_obs.get("zone"),
                    "target_zone": v_obs.get("zone"),
                }
            )

    relationships = []
    person_last = float(person.get("last_appearance", {}).get("timestamp_seconds", 0.0))
    person_first = float(person.get("first_appearance", {}).get("timestamp_seconds", 0.0))
    vehicle_last = float(vehicle.get("last_appearance", {}).get("timestamp_seconds", 0.0))
    vehicle_first = float(vehicle.get("first_appearance", {}).get("timestamp_seconds", 0.0))
    if near_vehicle_frames and person_last < vehicle_last:
        relationships.extend(_cluster_signals([near_vehicle_frames[-1]], "enter_vehicle", _track_ref(person), _track_ref(vehicle), 0.56))
    if near_vehicle_frames and person_first > vehicle_first:
        relationships.extend(_cluster_signals([near_vehicle_frames[0]], "exit_vehicle", _track_ref(person), _track_ref(vehicle), 0.56))

    load_signals = []
    for obj in objects:
        object_lookup = _observation_lookup(obj)
        for frame_id in sorted(set(common_frames).intersection(object_lookup)):
            p_obs = person_lookup[frame_id]
            v_obs = vehicle_lookup[frame_id]
            o_obs = object_lookup[frame_id]
            person_object_distance = _distance(p_obs, o_obs)
            object_vehicle_distance = _distance(o_obs, v_obs)
            if person_object_distance <= NEAR_OBJECT_DISTANCE and object_vehicle_distance <= TOUCH_VEHICLE_DISTANCE:
                load_signals.append(
                    {
                        "frame_id": frame_id,
                        "timestamp_seconds": float(p_obs.get("timestamp_seconds", 0.0)),
                        "distance_pixels": object_vehicle_distance,
                        "person_object_distance_pixels": round(person_object_distance, 2),
                        "object_track_id": obj.get("track_id"),
                        "source_zone": p_obs.get("zone"),
                        "target_zone": v_obs.get("zone"),
                    }
                )
    relationships.extend(_cluster_signals(load_signals, "load_object", _track_ref(person), _track_ref(vehicle), 0.6))
    return relationships


def _vehicle_zone_relationships(vehicle: Dict[str, Any]) -> List[Dict[str, Any]]:
    relationships = []
    for zone in vehicle.get("zone_history", []):
        relationships.append(_zone_relationship(vehicle, zone, "entered", 0.82))
        if float(zone.get("duration_seconds", 0.0)) > 0.0:
            relationships.append(_zone_relationship(vehicle, zone, "left", 0.76))
        if _zone_stationary_seconds(vehicle, zone.get("zone")) >= 2.0:
            relationships.append(_zone_relationship(vehicle, zone, "parked", 0.68))
    return relationships


def _object_zone_relationships(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    relationships = []
    for zone in obj.get("zone_history", []):
        relation_type = "outside" if zone.get("zone") == "unknown" else "inside"
        relationships.append(_zone_relationship(obj, zone, relation_type, 0.78))
        if _zone_stationary_seconds(obj, zone.get("zone")) >= 2.0:
            relationships.append(_zone_relationship(obj, zone, "stationary", 0.7))
    return relationships


def _zone_relationship(timeline: Dict[str, Any], zone: Dict[str, Any], relation_type: str, confidence: float) -> Dict[str, Any]:
    start_seconds = float(zone.get("entered_at_seconds", 0.0))
    end_seconds = float(zone.get("left_at_seconds", start_seconds))
    return {
        "relationship_type": relation_type,
        "source": _track_ref(timeline),
        "target": {"entity_type": "zone", "zone": zone.get("zone", "unknown")},
        "start": {
            "frame_id": zone.get("entered_frame_id"),
            "timestamp_seconds": start_seconds,
            "timestamp_human": zone.get("entered_at") or _format_timestamp(start_seconds),
        },
        "end": {
            "frame_id": zone.get("left_frame_id"),
            "timestamp_seconds": end_seconds,
            "timestamp_human": zone.get("left_at") or _format_timestamp(end_seconds),
        },
        "confidence": confidence,
        "evidence": {
            "zone": zone.get("zone"),
            "duration_seconds": float(zone.get("duration_seconds", 0.0)),
            "entered_frame_id": zone.get("entered_frame_id"),
            "left_frame_id": zone.get("left_frame_id"),
        },
    }


def _zone_stationary_seconds(timeline: Dict[str, Any], zone_name: Optional[str]) -> float:
    observations = timeline.get("observations", [])
    total = 0.0
    for previous, current in zip(observations, observations[1:]):
        if previous.get("zone") != zone_name or current.get("zone") != zone_name:
            continue
        elapsed = float(current.get("timestamp_seconds", 0.0)) - float(previous.get("timestamp_seconds", 0.0))
        if elapsed <= 0.0:
            continue
        if _observation_speed(timeline, current.get("frame_id")) <= STATIONARY_SPEED:
            total += elapsed
    return total


def _build_relationship_graph(timelines: List[Dict[str, Any]]) -> Dict[str, Any]:
    persons = [timeline for timeline in timelines if timeline.get("entity_type") == "person"]
    vehicles = [timeline for timeline in timelines if timeline.get("entity_type") == "vehicle"]
    objects = [timeline for timeline in timelines if timeline.get("entity_type") == "object"]

    relationships: List[Dict[str, Any]] = []
    for index, person in enumerate(persons):
        for other_person in persons[index + 1:]:
            relationships.extend(_person_person_relationships(person, other_person))
        for obj in objects:
            relationships.extend(_person_object_relationships(person, obj))
        for vehicle in vehicles:
            relationships.extend(_person_vehicle_relationships(person, vehicle, objects))

    for vehicle in vehicles:
        relationships.extend(_vehicle_zone_relationships(vehicle))
    for obj in objects:
        relationships.extend(_object_zone_relationships(obj))

    for index, relationship in enumerate(relationships, start=1):
        relationship["relationship_id"] = f"rel_{index:05d}"

    by_type: Dict[str, int] = {}
    for relationship in relationships:
        rel_type = relationship["relationship_type"]
        by_type[rel_type] = by_type.get(rel_type, 0) + 1

    return {
        "benchmark": "relationship_graph_builder_phase_2",
        "input": str(ENTITY_TIMELINES_PATH),
        "output": str(RELATIONSHIP_GRAPH_PATH),
        "summary": {
            "entity_count": len(timelines),
            "relationship_count": len(relationships),
            "relationship_count_by_type": dict(sorted(by_type.items())),
            "no_vlm": True,
            "method": "deterministic_geometry_and_timeline_rules",
        },
        "parameters": {
            "near_person_distance_pixels": NEAR_PERSON_DISTANCE,
            "meeting_distance_pixels": MEETING_DISTANCE,
            "following_distance_pixels": FOLLOWING_DISTANCE,
            "near_object_distance_pixels": NEAR_OBJECT_DISTANCE,
            "touch_object_distance_pixels": TOUCH_OBJECT_DISTANCE,
            "near_vehicle_distance_pixels": NEAR_VEHICLE_DISTANCE,
            "touch_vehicle_distance_pixels": TOUCH_VEHICLE_DISTANCE,
            "stationary_speed_pixels_per_second": STATIONARY_SPEED,
            "moving_speed_pixels_per_second": MOVING_SPEED,
            "max_cluster_gap_seconds": MAX_CLUSTER_GAP_SECONDS,
        },
        "entities": [_track_ref(timeline) for timeline in timelines],
        "relationships": relationships,
    }


def main() -> None:
    start = time.perf_counter()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    payload = _load_timelines()
    graph = _build_relationship_graph(payload["entity_timelines"])
    graph["summary"]["wall_clock_runtime_seconds"] = time.perf_counter() - start
    RELATIONSHIP_GRAPH_PATH.write_text(json.dumps(graph, indent=4), encoding="utf-8")

    print("RELATIONSHIP_GRAPH_BENCHMARK_START")
    print(json.dumps({"relationship_graph": str(RELATIONSHIP_GRAPH_PATH)}))
    print("RELATIONSHIP_GRAPH_BENCHMARK_END")


if __name__ == "__main__":
    main()
