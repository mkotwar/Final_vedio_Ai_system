"""Event Aggregation Service V2.
Groups consecutive frames into semantic events using structured metadata 
(Actor, Relationship, Location) matching, with a fallback to the V1 heuristic continuity logic.
"""

import json
import re
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from loguru import logger

from app.core.config import settings
from app.core.utils import format_timestamp_human
from app.services.status_service import JobStatusService


class EventAggregationServiceV2:
    """Service to group frames into semantic events based on structured metadata and graph continuity."""

    @staticmethod
    def _extract_actor_ids(frame: Dict[str, Any]) -> Set[str]:
        """Extract all object IDs from the frame that could act as actors/nodes."""
        actors = set()
        for obj in frame.get("objects", []):
            if isinstance(obj, dict):
                obj_id = obj.get("id")
                if obj_id:
                    actors.add(obj_id)
        return actors

    @staticmethod
    def _extract_actor_signatures(frame: Dict[str, Any]) -> Set[str]:
        """Extracts (type, subtype, color) signatures for actor continuity fallback."""
        signatures = set()
        for obj in frame.get("objects", []):
            if isinstance(obj, dict):
                typ = obj.get("type", "").lower()
                sub = obj.get("subtype", "").lower()
                col = obj.get("color", "").lower()
                if typ or sub:
                    signatures.add(f"{typ}_{sub}_{col}")
        return signatures

    @staticmethod
    def _extract_graph_edges(frame: Dict[str, Any]) -> Set[Tuple[str, str, str]]:
        """Extracts relationships as (subject_id, relation, target_id) edges."""
        edges = set()
        for rel in frame.get("relationships", []):
            if isinstance(rel, dict):
                sub = rel.get("subject_id")
                rel_type = rel.get("relation")
                tgt = rel.get("target_id")
                if sub and rel_type and tgt:
                    edges.add((sub, rel_type, tgt))
        return edges

    @staticmethod
    def _extract_location_edges(frame: Dict[str, Any]) -> Set[Tuple[str, str]]:
        """Extracts location context as (object_id, location) spatial edges."""
        locs = set()
        for loc in frame.get("location_context", []):
            if isinstance(loc, dict):
                obj = loc.get("object_id")
                location = loc.get("location")
                if obj and location:
                    locs.add((obj, location))
        return locs

    @classmethod
    def _calculate_structured_continuity(cls, group_frames: List[Dict[str, Any]], new_frame: Dict[str, Any]) -> float:
        """
        Simple weighted matching for V2:
        Same actor (50%) + same relationship (30%) + same location (20%).
        Returns a score between 0.0 and 1.0.
        """
        if not group_frames:
            return 1.0

        last_frame = group_frames[-1]

        old_actor_ids = cls._extract_actor_ids(last_frame)
        new_actor_ids = cls._extract_actor_ids(new_frame)

        old_actor_sigs = cls._extract_actor_signatures(last_frame)
        new_actor_sigs = cls._extract_actor_signatures(new_frame)

        old_edges = cls._extract_graph_edges(last_frame)
        new_edges = cls._extract_graph_edges(new_frame)

        old_locs = cls._extract_location_edges(last_frame)
        new_locs = cls._extract_location_edges(new_frame)

        def jaccard(set1: set, set2: set) -> float:
            if not set1 and not set2:
                return 1.0
            union_len = len(set1.union(set2))
            return len(set1.intersection(set2)) / union_len if union_len else 0.0

        id_overlap = jaccard(old_actor_ids, new_actor_ids)
        sig_overlap = jaccard(old_actor_sigs, new_actor_sigs)
        actor_overlap = (id_overlap * 0.7) + (sig_overlap * 0.3)

        edge_overlap = jaccard(old_edges, new_edges)
        loc_overlap = jaccard(old_locs, new_locs)

        score = (actor_overlap * 0.5) + (edge_overlap * 0.3) + (loc_overlap * 0.2)
        return score

    @staticmethod
    def _calculate_v1_fallback_continuity(group_frames: List[Dict[str, Any]], new_frame: Dict[str, Any]) -> float:
        """Original V1 heuristic Jaccard scoring logic, retained as a fallback layer."""
        if not group_frames:
            return 1.0

        window_size = getattr(settings, "EVENT_CONTEXT_WINDOW", 5)
        recent_frames = group_frames[-window_size:]

        default_weights = [0.50, 0.25, 0.15, 0.07, 0.03]
        weights = default_weights[:len(recent_frames)]
        weight_sum = sum(weights)
        weights = [w / weight_sum for w in weights]

        def get_actors(f: Dict) -> set:
            actors = set()
            for obj in f.get("objects", []):
                if isinstance(obj, dict):
                    typ = str(obj.get("type", "")).lower()
                    sub = str(obj.get("subtype", "")).lower()
                    if "person" in typ or "human" in typ or "vehicle" in typ or "car" in typ or "bike" in typ or "truck" in typ:
                        actors.add(f"{typ}_{sub}")
            return actors

        def get_activities(f: Dict) -> set:
            acts = set()
            stop_words = {"in", "on", "at", "the", "a", "an", "is", "are", "and", "or"}
            for a in f.get("activities", []):
                words = [w.strip() for w in str(a).lower().replace('/', ' ').replace(',', ' ').split()]
                acts.update([w for w in words if w not in stop_words])
            return acts

        def get_scene_context(f: Dict) -> set:
            ctx = set()
            if "scene_type" in f:
                ctx.add(str(f["scene_type"]).lower())
            for kw in f.get("keywords", []):
                ctx.add(str(kw).lower())
            return ctx

        def get_behavioral_flags(f: Dict) -> set:
            flags = set()
            text = str(f.get("activities", [])) + " " + str(f.get("objects", []))
            text = text.lower()
            if any(k in text for k in ["fall", "falling", "collapse", "ground"]): flags.add("fall")
            if any(k in text for k in ["crash", "collision", "accident", "hit", "strike"]): flags.add("crash")
            if any(k in text for k in ["fire", "smoke", "flame", "burn"]): flags.add("fire")
            if any(k in text for k in ["run", "flee", "sprint", "chase"]): flags.add("fleeing")
            if any(k in text for k in ["guard", "security", "police", "officer"]): flags.add("security")
            return flags

        def jaccard(set1: set, set2: set) -> float:
            if not set1 and not set2:
                return 1.0
            union = set1.union(set2)
            return len(set1.intersection(set2)) / len(union) if union else 0.0

        new_actors = get_actors(new_frame)
        new_acts = get_activities(new_frame)
        new_ctx = get_scene_context(new_frame)
        new_flags = get_behavioral_flags(new_frame)

        weighted_actor = 0.0
        weighted_act = 0.0
        weighted_ctx = 0.0
        weighted_flag = 0.0

        for idx, frame in enumerate(reversed(recent_frames)):
            weight = weights[idx]

            f_actors = get_actors(frame)
            f_acts = get_activities(frame)
            f_ctx = get_scene_context(frame)
            f_flags = get_behavioral_flags(frame)

            weighted_actor += jaccard(f_actors, new_actors) * weight
            weighted_act += jaccard(f_acts, new_acts) * weight
            weighted_ctx += jaccard(f_ctx, new_ctx) * weight
            weighted_flag += jaccard(f_flags, new_flags) * weight

        total_score = (weighted_actor * 0.3) + (weighted_act * 0.3) + (weighted_ctx * 0.2) + (weighted_flag * 0.2)
        return total_score

    @classmethod
    def _is_continuous(cls, group_frames: List[Dict[str, Any]], new_frame: Dict[str, Any]) -> Dict[str, Any]:
        """Determines if the new frame is part of the current event graph using hybrid scoring."""
        first_time = float(group_frames[0].get("timestamp_seconds", 0.0))
        current_time = float(new_frame.get("timestamp_seconds", 0.0))
        duration = current_time - first_time

        if duration > getattr(settings, "MAX_EVENT_DURATION_SECONDS", 120.0):
            return {"is_continuous": False, "structural_score": 0.0, "fallback_score": 0.0, "decision_source": "duration_limit"}

        # 1. Structured Scoring
        structured_score = cls._calculate_structured_continuity(group_frames, new_frame)
        
        # 2. Fallback Scoring
        fallback_score = cls._calculate_v1_fallback_continuity(group_frames, new_frame)

        # Thresholds
        structured_threshold = 0.40
        fallback_threshold = getattr(settings, "EVENT_CONTINUITY_THRESHOLD", 0.35)

        is_continuous = False
        decision = "none"
        if structured_score >= structured_threshold:
            is_continuous = True
            decision = "structural"
        elif fallback_score >= fallback_threshold:
            is_continuous = True
            decision = "fallback"
        
        logger.debug(
            f"Frame={new_frame.get('frame_id')} | "
            f"StructuredScore={structured_score:.2f} | FallbackScore={fallback_score:.2f} | "
            f"Decision={'CONTINUE' if is_continuous else 'BREAK'} | Source={decision}"
        )

        return {
            "is_continuous": is_continuous,
            "structural_score": structured_score,
            "fallback_score": fallback_score,
            "decision_source": decision
        }

    @staticmethod
    def _get_primary_actor_details(merged_objects: List[Dict], primary_actor_id: str) -> Dict[str, Any]:
        """Finds and formats the primary actor's type/subtype details for narrative use."""
        for obj in merged_objects:
            if obj.get("id") == primary_actor_id:
                return obj
        return {}

    @classmethod
    def process_events(cls, video_id: str, frames_metadata: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Groups consecutive frames into events, generated aggregated fields using structural metadata."""
        logger.info(f"[V2] Starting structured event aggregation for video: {video_id} with {len(frames_metadata)} frames")
        JobStatusService.update(video_id, current_step="Aggregating semantic events (V2)...", progress_percent=85.0)

        if not frames_metadata:
            return []

        sorted_frames = sorted(frames_metadata, key=lambda x: x.get("timestamp_seconds", 0.0))

        groups: List[List[Dict[str, Any]]] = []
        current_group: List[Dict[str, Any]] = [sorted_frames[0]]

        i = 1
        cluster_idx = 1
        while i < len(sorted_frames):
            frame = sorted_frames[i]

            telemetry = cls._is_continuous(current_group, frame)

            if telemetry["is_continuous"]:
                frame["_v2_telemetry"] = telemetry
                current_group.append(frame)
                i += 1
            else:
                # Tolerance look-ahead
                if i + 1 < len(sorted_frames):
                    next_frame = sorted_frames[i+1]
                    next_telemetry = cls._is_continuous(current_group, next_frame)
                    if next_telemetry["is_continuous"]:
                        frame["_v2_telemetry"] = {"structural_score": 0.0, "fallback_score": 0.0, "decision_source": "tolerance_gap"}
                        next_frame["_v2_telemetry"] = next_telemetry
                        current_group.append(frame)
                        current_group.append(next_frame)
                        i += 2
                        continue

                # Break
                groups.append(current_group)
                current_group = [frame]
                cluster_idx += 1
                i += 1

        if current_group:
            groups.append(current_group)

        # Prepare storage
        video_events_dir = settings.EVENTS_DIR / video_id / "v2"
        video_events_dir.mkdir(parents=True, exist_ok=True)

        events: List[Dict[str, Any]] = []

        for idx, group in enumerate(groups, 1):
            event_id = f"evt_v2_{idx:03d}"
            first_frame = group[0]
            last_frame = group[-1]

            start_time = float(first_frame.get("timestamp_start_seconds", first_frame.get("timestamp_seconds", 0.0)))
            end_time = float(last_frame.get("timestamp_end_seconds", last_frame.get("timestamp_seconds", 0.0) + 1.0))
            duration = round(end_time - start_time, 2)

            # Aggregate activities
            all_acts = []
            for f in group:
                all_acts.extend(f.get("activities", []))
            activities = [act for act, _ in Counter(all_acts).most_common(2)]

            # Aggregate objects using ID mapping to track state correctly
            merged_objects_dict = {}
            for f in group:
                for obj in f.get("objects", []):
                    if not isinstance(obj, dict): continue
                    obj_id = obj.get("id")
                    if not obj_id: continue
                    if obj_id not in merged_objects_dict:
                        merged_objects_dict[obj_id] = {
                            "id": obj_id,
                            "type": obj.get("type", "unknown"),
                            "subtype": obj.get("subtype", ""),
                            "color": obj.get("color", ""),
                            "attributes": set(obj.get("attributes", []))
                        }
                    else:
                        merged_objects_dict[obj_id]["attributes"].update(obj.get("attributes", []))

            merged_objects = []
            for obj in merged_objects_dict.values():
                obj["attributes"] = sorted(list(obj["attributes"]))
                merged_objects.append(obj)

            # Aggregate relationships
            all_rels = []
            for f in group:
                all_rels.extend([r for r in f.get("relationships", []) if isinstance(r, dict)])
            
            # Aggregate locations
            all_locs = []
            for f in group:
                all_locs.extend([l for l in f.get("location_context", []) if isinstance(l, dict)])

            # Core Topology Extraction for Narrative
            # Determine the primary actor by seeing who appears most frequently in relationships
            rel_subjects = [r.get("subject_id") for r in all_rels if r.get("subject_id")]
            if rel_subjects:
                primary_actor_id = Counter(rel_subjects).most_common(1)[0][0]
            else:
                # Fallback to the first available object ID if no relationships
                primary_actor_id = merged_objects[0]["id"] if merged_objects else "an unknown object"

            # Determine primary relationship and target for the primary actor
            actor_rels = [r for r in all_rels if r.get("subject_id") == primary_actor_id]
            primary_relation = None
            primary_target_id = None
            if actor_rels:
                rel_counts = Counter((r.get("relation"), r.get("target_id")) for r in actor_rels if r.get("relation") and r.get("target_id"))
                if rel_counts:
                    (primary_relation, primary_target_id), _ = rel_counts.most_common(1)[0]

            # Determine primary location for the primary actor
            actor_locs = [l.get("location") for l in all_locs if l.get("object_id") == primary_actor_id and l.get("location")]
            primary_location = Counter(actor_locs).most_common(1)[0][0] if actor_locs else "the monitored area"

            # Resolve natural language names
            primary_actor_obj = cls._get_primary_actor_details(merged_objects, primary_actor_id)
            primary_actor_name = primary_actor_obj.get("subtype") or primary_actor_obj.get("type") or primary_actor_id
            primary_actor_name = primary_actor_name.replace("_", " ")

            primary_target_name = primary_target_id.replace("_", " ") if primary_target_id else None
            primary_relation_name = primary_relation.replace("_", " ") if primary_relation else None
            primary_location_name = primary_location.replace("_", " ")

            # Phase 1 Taxonomy Mapping (Basic inference)
            event_type = "presence"
            if primary_relation_name:
                event_type = "interaction"
            elif any(act in activities for act in ["entering", "exiting", "walking", "running"]):
                event_type = "movement"
            elif any(act in activities for act in ["waiting", "standing"]):
                event_type = "waiting"
            elif len(merged_objects) >= 3 and any(act in activities for act in ["gathering", "talking"]):
                event_type = "crowd"

            # Deterministic Narrative Generation
            subject_phrase = f"A {primary_actor_name}"
            loc_phrase = f"at {primary_location_name}"
            dur_phrase = f"for approximately {duration} seconds"

            if event_type == "interaction" and primary_relation_name and primary_target_name:
                rel_lower = primary_relation_name.lower()
                if "approach" in rel_lower:
                    narrative = f"{subject_phrase} approached {primary_target_name} {loc_phrase} {dur_phrase}."
                elif "talk" in rel_lower or "interact" in rel_lower:
                    narrative = f"{subject_phrase} interacted with {primary_target_name} {loc_phrase} {dur_phrase}."
                elif "near" in rel_lower or "stand" in rel_lower:
                    narrative = f"{subject_phrase} was near {primary_target_name} {loc_phrase} {dur_phrase}."
                elif "hold" in rel_lower:
                    narrative = f"{subject_phrase} was holding {primary_target_name} {loc_phrase} {dur_phrase}."
                else:
                    narrative = f"{subject_phrase} was observed {primary_relation_name} {primary_target_name} {loc_phrase} {dur_phrase}."
            elif activities:
                joined_acts = " and ".join(activities)
                narrative = f"{subject_phrase} was observed {joined_acts} {loc_phrase} {dur_phrase}."
            else:
                narrative = f"{subject_phrase} was present {loc_phrase} {dur_phrase}."

            event_severity = 15 # Default for Phase 1 taxonomy

            # Telemetry logic
            telemetries = [f.get("_v2_telemetry") for f in group if f.get("_v2_telemetry")]
            if telemetries:
                avg_struct = sum(t["structural_score"] for t in telemetries) / len(telemetries)
                avg_fall = sum(t["fallback_score"] for t in telemetries) / len(telemetries)
                sources = [t["decision_source"] for t in telemetries if t["decision_source"] in ("structural", "fallback")]
                dominant_source = Counter(sources).most_common(1)[0][0] if sources else "structural"
            else:
                avg_struct = 1.0
                avg_fall = 1.0
                dominant_source = "structural"

            event_data = {
                "event_id": event_id,
                "video_id": video_id,
                "start_time": format_timestamp_human(start_time),
                "end_time": format_timestamp_human(end_time),
                "timestamp_start_seconds": start_time,
                "timestamp_end_seconds": end_time,
                "timestamp_start_human": format_timestamp_human(start_time),
                "timestamp_end_human": format_timestamp_human(end_time),
                "duration_seconds": duration,
                "frame_count": len(group),
                "objects": merged_objects,
                "activities": activities,
                "primary_object": primary_actor_name,
                "color": primary_actor_obj.get("color", ""),
                "primary_activity": activities[0] if activities else "present",
                "source_frames": [f.get("frame_id") for f in group if f.get("frame_id")],
                "event_type": event_type,
                "summary": narrative,
                
                # Narrative / Graph properties
                "scene_context": first_frame.get("scene_description", ""),
                "real_world_time": None, # Removed OCR dependence for V2 core
                "actor_description": f"{primary_actor_obj.get('color', '')} {primary_actor_name}".strip(),
                "participants": [obj.get("id") for obj in merged_objects if obj.get("id") != primary_actor_id],
                "participant_count": len(merged_objects) - 1 if merged_objects else 0,
                "behavioral_flags": [event_type],
                "confidence": 0.85, # Fixed confidence for V2 Phase 1
                "narrative_sentence": narrative,
                "location_text": primary_location_name,
                "event_severity": event_severity,
                "unified_text": narrative, # We map the clean narrative to unified_text to avoid regex spam downstream
                "frame_events": [],
                
                # Telemetry
                "structural_score": round(avg_struct, 3),
                "fallback_score": round(avg_fall, 3),
                "decision_source": dominant_source
            }

            event_file_path = video_events_dir / f"{event_id}.json"
            try:
                with open(event_file_path, "w", encoding="utf-8") as ef:
                    json.dump(event_data, ef, indent=4)
            except Exception as exc:
                logger.error(f"Failed to write event file {event_file_path}: {exc}")

            events.append(event_data)

        # Consolidate
        consolidated_events = []
        for e in events:
            source_frames = e.get("source_frames", [])
            first_frame_id = source_frames[0] if source_frames else None
            thumbnail_path = f"/api/v1/events/{video_id}/thumbnail/{first_frame_id}" if first_frame_id else None
            
            # Map fields strictly to V1 expected output schemas while using V2 deterministic data
            consolidated_events.append({
                **e,
                "description": e["summary"],
                "thumbnail_path": thumbnail_path,
            })

        consolidated_path = settings.METADATA_DIR / f"{video_id}_events_v2.json"
        try:
            with open(consolidated_path, "w", encoding="utf-8") as cf:
                json.dump(consolidated_events, cf, indent=4)
            logger.info(f"[V2] Saved consolidated events array to {consolidated_path} with {len(consolidated_events)} events.")
        except Exception as exc:
            logger.error(f"[V2] Failed to write consolidated events file {consolidated_path}: {exc}")

        JobStatusService.update(video_id, events_generated=len(consolidated_events))
        return events
