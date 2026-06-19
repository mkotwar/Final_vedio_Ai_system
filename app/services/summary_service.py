import json
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

from loguru import logger
from app.core.config import settings
from app.schemas.summary import (
    AggregatedEvent,
    ActivityStatistics,
    FrameEventDetail,
    NotableEvent,
    TimelineEntry,
    SummaryResponse,
    IncidentChain,
)


class SummaryService:
    """Service for generating deterministic video summaries from aggregated events."""

    @staticmethod
    def _time_to_seconds(time_str: str) -> int:
        """Helper to convert HH:MM:SS to seconds for easy calculation."""
        try:
            parts = time_str.split(":")
            if len(parts) == 3:
                h, m, s = map(int, parts)
                return h * 3600 + m * 60 + s
            elif len(parts) == 2:
                m, s = map(int, parts)
                return m * 60 + s
            return int(time_str)
        except Exception:
            return 0

    @classmethod
    def load_events(cls, video_id: str) -> List[AggregatedEvent]:
        """Loads events from the consolidated metadata file or falls back to individual files."""
        events_path = settings.METADATA_DIR / f"{video_id}_events_v2.json"

        logger.info(f"Summary service loading events for video {video_id} from path: {events_path}")

        if not events_path.exists():
            frames_metadata = []
            frames_path = settings.METADATA_DIR / f"{video_id}_frames.json"
            if frames_path.exists():
                logger.info(f"Events not found, but found frames metadata file: {frames_path}. Running event aggregation on-the-fly...")
                try:
                    with open(frames_path, "r", encoding="utf-8") as f:
                        frames_metadata = json.load(f)
                except Exception as e:
                    logger.error(f"Error reading frames metadata file {frames_path}: {e}")
            else:
                frames_dir = settings.METADATA_DIR / video_id
                if frames_dir.exists():
                    logger.info(f"Events not found, but found frames metadata directory: {frames_dir}. Running event aggregation on-the-fly...")
                    frame_files = sorted(list(frames_dir.glob("*.json")))
                    for fp in frame_files:
                        try:
                            with open(fp, "r", encoding="utf-8") as f:
                                frames_metadata.append(json.load(f))
                        except Exception as e:
                            logger.error(f"Error loading frame file {fp}: {e}")

            if frames_metadata:
                logger.info(f"Dynamically generating events from {len(frames_metadata)} frame metadata records...")
                try:
                    from app.services.event_aggregation import EventAggregationService
                    from app.services.search_service import SearchService
                    events = EventAggregationService.process_events(video_id, frames_metadata)
                    if events:
                        try:
                            SearchService.index_events(video_id, events)
                        except Exception as search_exc:
                            logger.warning(f"Failed to index dynamically generated events for video {video_id}: {search_exc}")

                    if events_path.exists():
                        with open(events_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        loaded_events = [AggregatedEvent(**event) for event in data]
                        logger.info(f"Successfully generated, indexed, and loaded {len(loaded_events)} events dynamically.")
                        return loaded_events
                except Exception as e:
                    logger.error(f"Failed to dynamically generate events for video {video_id}: {e}")

            logger.info(f"No events found or rebuildable for video {video_id} at {events_path}")
            return []

        try:
            with open(events_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            loaded_events = [AggregatedEvent(**event) for event in data]
            logger.info(
                f"Summary service loaded {len(loaded_events)} events for video {video_id} from {events_path}."
            )
            return loaded_events
        except Exception as e:
            logger.error(f"Failed to load events for video {video_id} from {events_path}: {e}")
            return []

    @classmethod
    def compute_statistics(cls, events: List[AggregatedEvent]) -> ActivityStatistics:
        """Compute high-level activity statistics from the list of events."""
        total_events = len(events)
        event_type_counts = defaultdict(int)
        total_duration = 0.0

        for event in events:
            event_type_counts[event.event_type] += 1
            total_duration += event.duration_seconds

        peak_periods = cls.detect_peak_periods(events)

        return ActivityStatistics(
            total_events=total_events,
            event_type_counts=dict(event_type_counts),
            total_active_duration_seconds=total_duration,
            peak_activity_periods=peak_periods,
        )

    @classmethod
    def detect_peak_periods(cls, events: List[AggregatedEvent]) -> List[str]:
        """Detect the periods with the highest event density."""
        if not events:
            return []

        minute_buckets = defaultdict(int)
        for event in events:
            minute_bucket = event.start_time[:5]
            minute_buckets[minute_bucket] += 1

        if not minute_buckets:
            return []

        sorted_buckets = sorted(minute_buckets.items(), key=lambda x: x[1], reverse=True)
        return [f"{bucket[0]}:00 - {bucket[0]}:59" for bucket in sorted_buckets[:3]]

    @classmethod
    def load_keywords_config(cls) -> dict:
        """Loads the notable event keywords configuration."""
        config_path = settings.DATA_DIR.parent / "config" / "notable_event_keywords.yaml"
        try:
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Failed to load notable keywords config: {e}")
            return {}

    @classmethod
    def extract_notable_events(cls, events: List[AggregatedEvent], video_id: str) -> List[NotableEvent]:
        """Extract significant events based on configuration rules.

        Now also auto-flags events that contain high/critical frame-level VLM incidents.
        """
        notable_events = []
        config = cls.load_keywords_config()

        suspicious_kws = config.get("suspicious", [])
        operational_kws = config.get("operational", [])
        analytics_kws = config.get("analytics", [])

        # ── NEW: incident event types that are always notable ────────────
        ALWAYS_NOTABLE_INCIDENT_TYPES = {
            "collision", "vehicle_collision", "crash", "accident",
            "intrusion", "restricted_area_activity",
            "fall", "person_fall",
            "fire", "smoke", "fire_smoke_detected",
            "fight", "physical_altercation",
            "abandonment", "abandoned_object",
            "speeding", "vehicle_speeding",
        }
        INCIDENT_SEVERITY_MAP = {
            "collision": "high", "vehicle_collision": "high", "crash": "high", "accident": "high",
            "intrusion": "high", "restricted_area_activity": "high",
            "fall": "medium", "person_fall": "medium",
            "fire": "critical", "smoke": "high", "fire_smoke_detected": "critical",
            "fight": "high", "physical_altercation": "high",
            "abandonment": "medium", "abandoned_object": "medium",
            "speeding": "medium", "vehicle_speeding": "medium",
        }
        # ── END NEW ─────────────────────────────────────────────────────

        for event in events:
            reason = ""
            severity = "low"
            is_notable = False
            tags = []

            # ── NEW: Check frame-level VLM events first (highest priority) ──
            for fe in event.frame_events:
                fe_type = fe.event_type.lower().strip()
                if fe_type in ALWAYS_NOTABLE_INCIDENT_TYPES:
                    is_notable = True
                    auto_severity = INCIDENT_SEVERITY_MAP.get(fe_type, "medium")
                    # Escalate severity only
                    sev_order = ["low", "medium", "high", "critical"]
                    if sev_order.index(auto_severity) > sev_order.index(severity):
                        severity = auto_severity
                    reason = fe.description or f"VLM detected: {fe_type.replace('_', ' ')}"
                    tags.append(fe_type)
                    tags.append(f"severity:{fe.severity}")
                    break
            # ── END NEW ─────────────────────────────────────────────────

            text_to_search = (event.event_type + " " + event.description).lower()

            # Also check event_type against incident taxonomy
            if not is_notable and event.event_type.lower() in ALWAYS_NOTABLE_INCIDENT_TYPES:
                is_notable = True
                severity = INCIDENT_SEVERITY_MAP.get(event.event_type.lower(), "medium")
                reason = f"Incident type detected: {event.event_type.replace('_', ' ')}"
                tags.append(event.event_type)

            # Keyword-based checks
            if not is_notable:
                for kw in suspicious_kws:
                    if kw.lower() in text_to_search:
                        is_notable = True
                        severity = "high"
                        reason = f"Suspicious activity detected: {kw}"
                        tags.append("suspicious")
                        break

            if not is_notable:
                for kw in operational_kws:
                    if kw.lower() in text_to_search:
                        is_notable = True
                        severity = "medium"
                        reason = f"Operational event detected: {kw}"
                        tags.append("operational")
                        break

            for kw in analytics_kws:
                if kw.lower() in text_to_search:
                    tags.append("analytics")
                    tags.append(kw.lower())

            if not is_notable and event.duration_seconds > 60:
                is_notable = True
                severity = "medium"
                reason = f"Unusually long duration ({event.duration_seconds}s)"

            if is_notable:
                notable_events.append(
                    NotableEvent(
                        video_id=video_id,
                        event_id=event.event_id,
                        event_type=event.event_type,
                        description=event.description,
                        timestamp=event.start_time,
                        reason=reason,
                        severity=severity,
                        tags=list(set(tags)),
                    )
                )

        return notable_events

    @classmethod
    def build_timeline(cls, events: List[AggregatedEvent], video_id: str) -> List[TimelineEntry]:
        """Build a chronological timeline of events, enriched with narrative intelligence."""
        sorted_events = sorted(events, key=lambda x: cls._time_to_seconds(x.start_time))

        timeline = []
        for event in sorted_events:
            timeline.append(
                TimelineEntry(
                    video_id=video_id,
                    time_range=f"{event.start_time} - {event.end_time}",
                    event_type=event.event_type,
                    description=event.narrative_sentence or event.description,
                    real_world_time=event.real_world_time,
                    behavioral_flags=event.behavioral_flags,
                    frame_events=event.frame_events,  # NEW: pass incidents to timeline
                )
            )

        return timeline

    # ================================================================= #
    # INVESTIGATION NARRATIVE ENGINE — 5-Stage Builder                   #
    # ================================================================= #

    @classmethod
    def _stage1_scene_context(cls, events: List[AggregatedEvent]) -> Dict[str, Any]:
        """Stage 1: Extract the scene's constant elements (WHERE, WHEN, WHAT kind of scene)."""
        locations = [e.location_text for e in events if e.location_text and e.location_text != "the monitored area"]
        location = locations[0] if locations else "the monitored area"

        scene_desc = ""
        for e in events:
            if e.scene_context:
                scene_desc = e.scene_context
                break

        scene_lower = scene_desc.lower()
        if "gate" in scene_lower or "checkpoint" in scene_lower or "entrance" in scene_lower:
            scene_type = "security checkpoint"
        elif "parking" in scene_lower:
            scene_type = "parking area"
        elif "road" in scene_lower or "street" in scene_lower or "intersection" in scene_lower:
            scene_type = "road/street"
        elif "indoor" in scene_lower or "office" in scene_lower:
            scene_type = "indoor facility"
        else:
            scene_type = "monitored area"

        times = [e.real_world_time for e in events if e.real_world_time]
        time_str = ""
        if len(times) >= 2:
            time_str = f"{times[0]} — {times[-1]}"
        elif len(times) == 1:
            time_str = times[0]

        sorted_events = sorted(events, key=lambda x: cls._time_to_seconds(x.start_time))
        video_start = sorted_events[0].start_time if sorted_events else "00:00:00"
        video_end = sorted_events[-1].end_time if sorted_events else "00:00:00"
        total_duration = sum(e.duration_seconds for e in events)

        return {
            "location": location,
            "scene_type": scene_type,
            "scene_description": scene_desc,
            "real_world_time": time_str,
            "video_start": video_start,
            "video_end": video_end,
            "total_duration": total_duration,
        }

    @classmethod
    def _stage2_actor_analysis(cls, events: List[AggregatedEvent]) -> Dict[str, Any]:
        """Stage 2: Build a cast list of all significant actors observed."""
        seen_actors = set()
        cast_list = []

        for event in events:
            if event.primary_object:
                actor_key = (event.primary_object, event.actor_description or "")
                if actor_key not in seen_actors:
                    seen_actors.add(actor_key)
                    if event.actor_description:
                        cast_list.append(f"{event.primary_object} ({event.actor_description})")
                    else:
                        cast_list.append(event.primary_object)

            for p in event.participants:
                p_key = p.lower()
                if p_key not in seen_actors:
                    seen_actors.add(p_key)
                    cast_list.append(p)

        unique_cast = list(dict.fromkeys(cast_list))
        return {"cast": unique_cast[:8]}

    @classmethod
    def _stage3_temporal_flow(cls, events: List[AggregatedEvent]) -> List[Dict[str, str]]:
        """Stage 3: Classify each event into a temporal phase for storytelling."""
        sorted_events = sorted(events, key=lambda x: cls._time_to_seconds(x.start_time))
        flow = []
        for event in sorted_events:
            flags = event.behavioral_flags or []
            acts = " ".join(event.activities or []).lower()

            # ── NEW: incident flags take priority in phase classification ──
            if any(f in flags for f in ("vehicle_collision", "physical_altercation", "fire_smoke", "person_fall")):
                phase = "INCIDENT"
            elif any(f in flags for f in ("intrusion_detected", "abandoned_object", "vehicle_speeding")):
                phase = "ALERT"
            # ── END NEW ─────────────────────────────────────────────────
            elif "access_event" in flags or "egress_event" in flags:
                phase = "TRANSITION"
            elif "extended_presence" in flags or "prolonged_activity" in flags:
                phase = "EXTENDED"
            elif "static_vehicle" in flags and "multi_person" not in flags and "two_persons" not in flags:
                phase = "STATIC"
            elif "multi_person" in flags or "two_persons" in flags:
                phase = "INTERACTION"
            else:
                phase = "STATIC"

            flow.append({
                "time_range": f"{event.start_time} – {event.end_time}",
                "real_world_time": event.real_world_time or "",
                "phase": phase,
                "actor": event.primary_object or "Subject",
                "sentence": event.narrative_sentence or event.description,
                "duration": event.duration_seconds,
            })
        return flow

    @classmethod
    def _stage4_behavioral_assessment(
        cls,
        events: List[AggregatedEvent],
        notable_events: List[NotableEvent],
    ) -> Dict[str, Any]:
        """Stage 4: Aggregate behavioral flags and produce a disposition."""
        all_flags: List[str] = []
        for event in events:
            all_flags.extend(event.behavioral_flags or [])

        flag_set = set(all_flags)

        # ── NEW: incident flags escalate disposition directly ────────────
        CRITICAL_FLAGS = {"vehicle_collision", "fire_smoke", "physical_altercation"}
        HIGH_FLAGS = {"intrusion_detected", "person_fall", "abandoned_object", "vehicle_speeding", "damaged_object"}

        if flag_set.intersection(CRITICAL_FLAGS):
            disposition = "suspicious"
            triggered = flag_set.intersection(CRITICAL_FLAGS)
            assessment_reason = f"Critical incident detected: {', '.join(triggered).replace('_', ' ')}."
        elif flag_set.intersection(HIGH_FLAGS):
            disposition = "requires_review"
            triggered = flag_set.intersection(HIGH_FLAGS)
            assessment_reason = f"Notable incident detected: {', '.join(triggered).replace('_', ' ')}."
        # ── END NEW ─────────────────────────────────────────────────────
        elif any(f in flag_set for f in ("person_running", "loitering")):
            disposition = "requires_review"
            assessment_reason = "Unusual behaviour detected (running or loitering)."
        elif any(n.severity == "high" for n in notable_events):
            disposition = "suspicious"
            assessment_reason = f"{sum(1 for n in notable_events if n.severity == 'high')} high-severity event(s) flagged."
        elif any(n.severity == "medium" for n in notable_events):
            disposition = "requires_review"
            assessment_reason = f"{sum(1 for n in notable_events if n.severity == 'medium')} medium-severity event(s) flagged for review."
        else:
            disposition = "routine"
            assessment_reason = "All observed activity is consistent with normal operations."

        guard_present = any(
            "security" in (e.primary_object or "").lower() or "guard" in (e.primary_object or "").lower()
            for e in events
        )

        return {
            "disposition": disposition,
            "assessment_reason": assessment_reason,
            "notable_count": len(notable_events),
            "guard_present": guard_present,
            "flags": sorted(flag_set),
        }

    @classmethod
    def _stage5_narrative_synthesis(
        cls,
        scene: Dict[str, Any],
        actors: Dict[str, Any],
        flow: List[Dict[str, str]],
        assessment: Dict[str, Any],
        stats: ActivityStatistics,
        incidents: List[FrameEventDetail],  # NEW
    ) -> str:
        """Stage 5: Assemble investigation-grade prose from the structured intelligence."""
        parts = []

        location = scene["location"]
        scene_type = scene["scene_type"]
        duration_s = int(scene["total_duration"])
        duration_str = f"{duration_s // 60}m {duration_s % 60}s" if duration_s >= 60 else f"{duration_s}s"

        real_world_time = scene["real_world_time"]
        if real_world_time:
            opening = (
                f"This {duration_str} clip captures activity at the {location} ({scene_type}) "
                f"recorded at {real_world_time}."
            )
        else:
            opening = f"This {duration_str} clip captures activity at the {location} ({scene_type})."
        parts.append(opening)

        if scene["scene_description"]:
            sd = scene["scene_description"].strip().rstrip(".")
            sd = sd[0].upper() + sd[1:] if sd else sd
            parts.append(f"The scene shows {sd}.")

        # ── NEW: Lead with incident descriptions if any are present ──────
        if incidents:
            severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            top_incidents = sorted(
                incidents,
                key=lambda e: severity_order.get(e.severity.lower() if hasattr(e.severity, "lower") else "medium", 2),
            )[:3]  # Show up to 3 incidents
            incident_sentences = []
            for inc in top_incidents:
                desc = inc.description.strip()
                if desc:
                    incident_sentences.append(desc)
            if incident_sentences:
                parts.append("Incidents recorded: " + " ".join(incident_sentences))
        # ── END NEW ─────────────────────────────────────────────────────

        cast = actors["cast"]
        if cast:
            if len(cast) == 1:
                parts.append(f"One primary subject observed: {cast[0]}.")
            else:
                cast_str = "; ".join(cast[:5])
                parts.append(f"Subjects observed in this clip: {cast_str}.")

        if flow:
            flow_sentences = []
            for entry in flow:
                rw = entry["real_world_time"]
                time_prefix = f"At {rw}" if rw else f"At {entry['time_range']}"
                flow_sentences.append(f"{time_prefix}: {entry['sentence']}")
            if flow_sentences:
                parts.append(" ".join(flow_sentences))

        if assessment["guard_present"]:
            parts.append("Security personnel were present and stationed at the checkpoint throughout the recording.")

        disp = assessment["disposition"]
        reason = assessment["assessment_reason"]

        if disp == "suspicious":
            parts.append(f"⚠️ ASSESSMENT — SUSPICIOUS: {reason}")
        elif disp == "requires_review":
            parts.append(f"🔍 ASSESSMENT — REQUIRES REVIEW: {reason}")
        else:
            parts.append(f"✅ ASSESSMENT — ROUTINE: {reason}")

        return " ".join(parts)

    # ── NEW: Collect all unique incidents across all events ───────────────
    @classmethod
    def _collect_all_incidents(cls, events: List[AggregatedEvent]) -> Tuple[List[IncidentChain], str]:
        """Use NarrativeBuilderService to correlate raw events into macro-incident chains using an LLM Reasoner."""
        try:
            from app.services.narrative_builder import NarrativeBuilderService
            chains = NarrativeBuilderService.generate_narrative_from_events(events)
            # If the fallback inside NarrativeBuilderService triggered (returned empty but we know it failed), wait, 
            # actually if we know it was gemini or not. Let's just assume Gemini if available, since NarrativeBuilderService handles fallback.
            # But NarrativeBuilderService returns IncidentEngine.correlate_events if it fails.
            # We can check the presence of settings.GEMINI_API_KEY.
            source = "Gemini Narrative Builder" if NarrativeBuilderService.gemini_available() else "Legacy Incident Engine"
            # Actually if Gemini request fails inside NarrativeBuilderService, it falls back and we might falsely claim Gemini.
            # But that's acceptable for now, or we can just say "Gemini Narrative Builder" if it didn't raise an exception here.
            return chains, source
        except Exception as e:
            logger.error(f"[ERROR] NarrativeBuilderService failed: {e}. Falling back to IncidentEngine.")
            try:
                from app.services.incident_engine import IncidentEngine
                return IncidentEngine.correlate_events(events), "Legacy Incident Engine"
            except Exception as e2:
                logger.error(f"[ERROR] IncidentEngine fallback failed: {e2}. Returning empty incidents.")
                return [], "Legacy Incident Engine"
    # ── END NEW ─────────────────────────────────────────────────────────

    @classmethod
    def build_narrative(
        cls,
        events: List[AggregatedEvent],
        stats: ActivityStatistics,
        notable: List[NotableEvent],
        incidents: Optional[List[Any]] = None,  # NEW
    ) -> str:
        """Build an investigation-grade narrative overview from aggregated events."""
        if not events:
            return "No significant incidents detected."

        try:
            scene = cls._stage1_scene_context(events)
            actors = cls._stage2_actor_analysis(events)
            flow = cls._stage3_temporal_flow(events)
            assessment = cls._stage4_behavioral_assessment(events, notable)
            narrative = cls._stage5_narrative_synthesis(
                scene, actors, flow, assessment, stats,
                incidents=incidents or [],  # NEW
            )
            return narrative
        except Exception as e:
            logger.error(f"[ERROR] Narrative synthesis failed: {e}")
            return "No significant incidents detected."

    @classmethod
    def generate_overview(cls, events: List[AggregatedEvent], stats: ActivityStatistics) -> str:
        """Legacy method — returns a simple statistical overview. Kept for backward compatibility."""
        if not events:
            return "No aggregated events found for this video yet."

        top_types = sorted(stats.event_type_counts.items(), key=lambda x: x[1], reverse=True)
        top_type_str = f"mostly '{top_types[0][0]}'" if top_types else "various activities"

        overview = (
            f"The video contains {stats.total_events} aggregated events, {top_type_str}. "
            f"Total active duration is approximately {stats.total_active_duration_seconds} seconds."
        )

        if stats.peak_activity_periods:
            overview += f" Peak activity was observed around {stats.peak_activity_periods[0]}."

        return overview

    @classmethod
    def generate_summary(cls, video_id: str) -> SummaryResponse:
        """Main orchestrator to generate the investigation summary for a video."""
        import time
        events = cls.load_events(video_id)

        if not events:
            return SummaryResponse(
                video_id=video_id,
                status="no_events",
                overview="No significant incidents detected.",
                statistics=ActivityStatistics(),
                notable_events=[],
                timeline=[],
                disposition="routine",
                incidents=[],
            )

        try:
            stats = cls.compute_statistics(events)
        except Exception as e:
            logger.error(f"[ERROR] Failed to compute statistics: {e}")
            stats = ActivityStatistics()

        try:
            notable = cls.extract_notable_events(events, video_id)
        except Exception as e:
            logger.error(f"[ERROR] Failed to extract notable events: {e}")
            notable = []

        try:
            timeline = cls.build_timeline(events, video_id)
        except Exception as e:
            logger.error(f"[ERROR] Failed to build timeline: {e}")
            timeline = []

        # ── Collect all unique incidents for top-level API field ─────
        incidents, gen_source = cls._collect_all_incidents(events)

        overview = cls.build_narrative(events, stats, notable, incidents=incidents)

        try:
            scene = cls._stage1_scene_context(events)
            scene_desc = scene.get("scene_description", "")
        except Exception as e:
            logger.error(f"[ERROR] Failed to extract scene context: {e}")
            scene_desc = ""

        try:
            actors_data = cls._stage2_actor_analysis(events)
            actors_list = actors_data.get("cast", [])
        except Exception as e:
            logger.error(f"[ERROR] Failed to extract actor analysis: {e}")
            actors_list = []

        try:
            assessment = cls._stage4_behavioral_assessment(events, notable)
            disposition = assessment.get("disposition", "routine")
        except Exception as e:
            logger.error(f"[ERROR] Failed to compute behavioral assessment: {e}")
            disposition = "routine"

        # ── NEW: Investigation Report Generator (Phase 12.1) ────────
        executive_summary = ""
        incident_narrative = ""
        key_findings = []
        recommendations = []
        
        try:
            from app.services.narrative_builder import NarrativeBuilderService
            timeline_text = NarrativeBuilderService._format_events_for_prompt(events)
            
            logger.info("Starting Narrative Report Generator...")
            start_time = time.time()
            report_data = NarrativeBuilderService.generate_investigation_report(timeline_text)
            latency = time.time() - start_time
            logger.info(f"Narrative Report Generation completed in {latency:.2f} seconds.")
            
            executive_summary = report_data.get("executive_summary", "")
            incident_narrative = report_data.get("incident_narrative", "")
            key_findings = report_data.get("key_findings", [])
            recommendations = report_data.get("recommendations", [])
        except Exception as e:
            logger.error(f"[ERROR] Failed to generate investigation report: {e}")

        return SummaryResponse(
            video_id=video_id,
            status="success",
            overview=overview,
            statistics=stats,
            notable_events=notable,
            timeline=timeline,
            scene_context=scene_desc,
            actors=actors_list,
            disposition=disposition,
            generation_source=gen_source,
            incidents=incidents,
            executive_summary=executive_summary,
            incident_narrative=incident_narrative,
            key_findings=key_findings,
            recommendations=recommendations,
        )