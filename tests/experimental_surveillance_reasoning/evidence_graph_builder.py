from collections import defaultdict
from typing import Any, Dict, List


def _event_nodes(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    nodes = []
    for event in events:
        nodes.append(
            {
                "node_id": f"event:{event.get('event_id')}",
                "node_type": "event",
                "event_id": event.get("event_id"),
                "event_type": event.get("event_type"),
                "start_seconds": event.get("timestamp_start_seconds", 0.0),
                "end_seconds": event.get("timestamp_end_seconds", 0.0),
                "summary": event.get("summary", ""),
                "activities": event.get("activities", []),
                "behavioral_flags": event.get("behavioral_flags", []),
            }
        )
    return nodes


def _actor_nodes(actor_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    nodes = []
    for actor in actor_payload.get("actors", []):
        nodes.append(
            {
                "node_id": f"actor:{actor['actor_id']}",
                "node_type": "actor",
                "actor_id": actor["actor_id"],
                "entity_type": actor.get("entity_type"),
                "subtype": actor.get("subtype", ""),
                "dominant_color": actor.get("dominant_color", ""),
                "activities": actor.get("activities", []),
                "flags": actor.get("flags", []),
            }
        )
    return nodes


def build_evidence_graph(
    frames: List[Dict[str, Any]],
    events: List[Dict[str, Any]],
    actor_payload: Dict[str, Any],
) -> Dict[str, Any]:
    actor_nodes = _actor_nodes(actor_payload)
    event_nodes = _event_nodes(events)
    nodes = actor_nodes + event_nodes
    edges: List[Dict[str, Any]] = []
    frame_to_event: Dict[str, str] = {}

    for event in events:
        event_id = str(event.get("event_id", ""))
        for frame_id in event.get("source_frames", []) or []:
            frame_to_event[frame_id] = event_id

    actor_event_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for actor in actor_payload.get("actors", []):
        actor_id = actor["actor_id"]
        for observation in actor.get("observations", []):
            event_id = frame_to_event.get(observation.get("frame_id", ""))
            if event_id:
                actor_event_counts[actor_id][event_id] += 1

    for actor_id, event_counts in actor_event_counts.items():
        for event_id, support_count in sorted(event_counts.items()):
            edges.append(
                {
                    "edge_type": "participates_in",
                    "source": f"actor:{actor_id}",
                    "target": f"event:{event_id}",
                    "support_count": support_count,
                }
            )

    event_lookup = {str(event.get("event_id", "")): event for event in events}
    sorted_events = sorted(
        events,
        key=lambda item: float(item.get("timestamp_start_seconds", 0.0)),
    )
    for previous, current in zip(sorted_events, sorted_events[1:]):
        previous_end = float(previous.get("timestamp_end_seconds", 0.0))
        current_start = float(current.get("timestamp_start_seconds", 0.0))
        gap_seconds = round(max(0.0, current_start - previous_end), 2)
        if gap_seconds <= 60.0:
            edges.append(
                {
                    "edge_type": "temporal_next",
                    "source": f"event:{previous.get('event_id')}",
                    "target": f"event:{current.get('event_id')}",
                    "gap_seconds": gap_seconds,
                }
            )

    for event in events:
        event_id = str(event.get("event_id", ""))
        flags = event.get("behavioral_flags", []) or []
        for flag in flags:
            edges.append(
                {
                    "edge_type": "has_behavior",
                    "source": f"event:{event_id}",
                    "target": f"behavior:{flag}",
                }
            )
            nodes.append(
                {
                    "node_id": f"behavior:{flag}",
                    "node_type": "behavior",
                    "name": flag,
                }
            )

    deduped_nodes: Dict[str, Dict[str, Any]] = {}
    for node in nodes:
        deduped_nodes[node["node_id"]] = node

    return {
        "summary": {
            "node_count": len(deduped_nodes),
            "edge_count": len(edges),
            "actor_count": len(actor_nodes),
            "event_count": len(event_nodes),
        },
        "nodes": list(deduped_nodes.values()),
        "edges": edges,
        "event_index": {
            event_id: {
                "summary": event.get("summary", ""),
                "frame_count": len(event.get("source_frames", []) or []),
                "activities": event.get("activities", []),
                "behavioral_flags": event.get("behavioral_flags", []),
            }
            for event_id, event in event_lookup.items()
        },
    }
