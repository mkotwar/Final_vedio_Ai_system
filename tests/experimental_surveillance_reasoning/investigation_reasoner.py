from typing import Any, Dict, List


def _event_text(event: Dict[str, Any]) -> str:
    parts = [event.get("summary", ""), event.get("scene_context", ""), " ".join(event.get("activities", []) or [])]
    for frame_event in event.get("frame_events", []) or []:
        parts.append(frame_event.get("event_type", ""))
        parts.append(frame_event.get("description", ""))
    return " ".join(str(part) for part in parts).lower()


def _possible_theft(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    findings = []
    for event in events:
        text = _event_text(event)
        flags = set(event.get("behavioral_flags", []) or [])
        if "carrying_object" in flags and ("exit" in text or "depart" in text or "leave" in text):
            findings.append(
                {
                    "hypothesis_type": "possible_theft_or_removal",
                    "confidence": 0.58,
                    "event_ids": [event.get("event_id")],
                    "reasoning": "Object-carrying behavior coincides with an exit-like movement pattern.",
                }
            )
    return findings


def _possible_loitering(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    findings = []
    for event in events:
        duration = float(event.get("duration_seconds", 0.0))
        flags = set(event.get("behavioral_flags", []) or [])
        if duration >= 30.0 or "loitering" in flags or "prolonged_activity" in flags:
            findings.append(
                {
                    "hypothesis_type": "possible_loitering",
                    "confidence": 0.62 if duration >= 60.0 else 0.49,
                    "event_ids": [event.get("event_id")],
                    "reasoning": "Extended stationary or prolonged-presence behavior was observed in a single event cluster.",
                }
            )
    return findings


def _possible_crowd(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    findings = []
    for event in events:
        participant_count = int(event.get("participant_count", 0))
        if participant_count >= 3 or "multi_person" in set(event.get("behavioral_flags", []) or []):
            findings.append(
                {
                    "hypothesis_type": "crowd_formation",
                    "confidence": 0.55,
                    "event_ids": [event.get("event_id")],
                    "reasoning": "Multiple participants were present in the same aggregated event.",
                }
            )
    return findings


def _possible_abandoned_object(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    findings = []
    for event in events:
        text = _event_text(event)
        flags = set(event.get("behavioral_flags", []) or [])
        if "placing object" in text or "dropping object" in text or "abandoned_object" in text or "person_leaves_object" in text:
            findings.append(
                {
                    "hypothesis_type": "possible_abandoned_object",
                    "confidence": 0.57,
                    "event_ids": [event.get("event_id")],
                    "reasoning": "Object placement/drop-off language appears in the event evidence.",
                }
            )
        elif "carrying_object" in flags and ("leave object" in text or "left while the object remained" in text):
            findings.append(
                {
                    "hypothesis_type": "possible_abandoned_object",
                    "confidence": 0.52,
                    "event_ids": [event.get("event_id")],
                    "reasoning": "Carry-and-leave behavior suggests a dropped or unattended object.",
                }
            )
    return findings


def _multi_stage_sequences(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    findings = []
    ordered = sorted(events, key=lambda item: float(item.get("timestamp_start_seconds", 0.0)))
    for previous, current in zip(ordered, ordered[1:]):
        gap = float(current.get("timestamp_start_seconds", 0.0)) - float(previous.get("timestamp_end_seconds", 0.0))
        if gap > 60.0:
            continue
        prev_flags = set(previous.get("behavioral_flags", []) or [])
        curr_flags = set(current.get("behavioral_flags", []) or [])
        if ("access_event" in prev_flags and "egress_event" in curr_flags) or (
            "carrying_object" in prev_flags and "egress_event" in curr_flags
        ):
            findings.append(
                {
                    "hypothesis_type": "multi_stage_incident_sequence",
                    "confidence": 0.51,
                    "event_ids": [previous.get("event_id"), current.get("event_id")],
                    "reasoning": "Two adjacent events form a plausible entry/activity/exit sequence.",
                }
            )
    return findings


def reason_over_surveillance(
    frames: List[Dict[str, Any]],
    events: List[Dict[str, Any]],
    actor_states: Dict[str, Any],
    evidence_graph: Dict[str, Any],
) -> Dict[str, Any]:
    hypotheses: List[Dict[str, Any]] = []
    hypotheses.extend(_possible_loitering(events))
    hypotheses.extend(_possible_theft(events))
    hypotheses.extend(_possible_crowd(events))
    hypotheses.extend(_possible_abandoned_object(events))
    hypotheses.extend(_multi_stage_sequences(events))

    hypotheses.sort(key=lambda item: (-float(item.get("confidence", 0.0)), item.get("hypothesis_type", "")))
    top_hypotheses = hypotheses[:12]

    overview = "No higher-risk surveillance hypotheses were triggered."
    if top_hypotheses:
        overview = "; ".join(
            f"{item['hypothesis_type']} ({item['confidence']:.2f})" for item in top_hypotheses[:4]
        )

    return {
        "summary": {
            "frame_count": len(frames),
            "event_count": len(events),
            "actor_count": len(actor_states.get('actors', [])),
            "graph_nodes": evidence_graph.get("summary", {}).get("node_count", 0),
            "graph_edges": evidence_graph.get("summary", {}).get("edge_count", 0),
            "hypothesis_count": len(top_hypotheses),
            "overview": overview,
        },
        "hypotheses": top_hypotheses,
    }
