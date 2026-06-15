"""Service for correlating isolated events into candidate macro-incident chains using semantic continuity."""

import uuid
import math
from typing import List, Dict, Any, Optional
from loguru import logger

from app.schemas.summary import AggregatedEvent
from app.schemas.incident import IncidentChain
from app.core.config import settings
from app.services.embedding_service import EmbeddingService

class IncidentMemory:
    """Maintains the active state of an ongoing incident chain."""
    
    def __init__(self, initial_event: AggregatedEvent, initial_embedding: List[float]):
        self.actors: set = set(initial_event.participants)
        if initial_event.actor_description:
            self.actors.add(initial_event.actor_description)
            
        self.objects: set = set([str(obj) for obj in initial_event.objects])
        self.activities: set = set(initial_event.activities)
        self.scene_contexts: set = {initial_event.scene_context}
        
        self.first_seen_seconds: int = self._time_to_seconds(initial_event.start_time)
        self.last_seen_seconds: int = self._time_to_seconds(initial_event.end_time)
        self.events: List[AggregatedEvent] = [initial_event]
        self.event_embeddings: List[List[float]] = [initial_embedding]
        
    @staticmethod
    def _time_to_seconds(time_str: str) -> int:
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
            
    def update(self, event: AggregatedEvent, embedding: List[float]):
        """Incorporate a new event into the incident memory."""
        self.actors.update(event.participants)
        if event.actor_description:
            self.actors.add(event.actor_description)
            
        self.objects.update([str(obj) for obj in event.objects])
        self.activities.update(event.activities)
        self.scene_contexts.add(event.scene_context)
        
        # Update temporal bounds
        start_sec = self._time_to_seconds(event.start_time)
        end_sec = self._time_to_seconds(event.end_time)
        self.first_seen_seconds = min(self.first_seen_seconds, start_sec)
        self.last_seen_seconds = max(self.last_seen_seconds, end_sec)
        
        self.events.append(event)
        self.event_embeddings.append(embedding)

    def get_aggregated_embedding(self) -> List[float]:
        """Compute the mean embedding vector for the incident so far."""
        if not self.event_embeddings:
            return []
        
        dims = len(self.event_embeddings[0])
        mean_emb = [0.0] * dims
        for emb in self.event_embeddings:
            for i in range(dims):
                mean_emb[i] += emb[i]
                
        # Normalize
        norm = math.sqrt(sum(x * x for x in mean_emb))
        if norm > 0:
            mean_emb = [x / norm for x in mean_emb]
        return mean_emb


class IncidentEngine:
    """Detects temporal and semantic relationships between events to build candidate chains."""

    @staticmethod
    def _calculate_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        dot = sum(x * y for x, y in zip(vec1, vec2))
        norm1 = math.sqrt(sum(x * x for x in vec1))
        norm2 = math.sqrt(sum(y * y for y in vec2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)
        
    @staticmethod
    def _calculate_jaccard(set1: set, set2: set) -> float:
        if not set1 and not set2:
            return 0.0 # Both empty means no explicit overlap to boost score
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0

    @classmethod
    def _evaluate_continuity(cls, event: AggregatedEvent, event_embedding: List[float], memory: IncidentMemory) -> float:
        """Evaluate multi-dimensional similarity between a new event and an active memory chain."""
        
        # 1. Semantic Continuity (Compare new event to the incident's aggregated story)
        mem_embedding = memory.get_aggregated_embedding()
        semantic_sim = cls._calculate_cosine_similarity(event_embedding, mem_embedding)
        
        # 2. Scene Continuity
        scene_sim = 1.0 if event.scene_context in memory.scene_contexts else 0.0
        
        # 3. Participant/Object Continuity
        event_actors = set(event.participants)
        if event.actor_description:
            event_actors.add(event.actor_description)
        actor_sim = cls._calculate_jaccard(event_actors, memory.actors)
        
        event_objects = set([str(obj) for obj in event.objects])
        object_sim = cls._calculate_jaccard(event_objects, memory.objects)
        
        participant_sim = max(actor_sim, object_sim)
        
        # 4. Activity Continuity
        activity_sim = cls._calculate_jaccard(set(event.activities), memory.activities)
        
        # 5. Temporal Continuity (Decay based on gap)
        event_start = IncidentMemory._time_to_seconds(event.start_time)
        gap = max(0, event_start - memory.last_seen_seconds)
        # Decay function: 1.0 at gap=0, 0.5 at gap=300s (5m), approaching 0.
        temporal_sim = math.exp(-gap / 300.0) 
        
        # Calculate dynamic confidence score
        # Using base weights, but the engine relies on the aggregate score
        weights = {
            "scene": 0.30,
            "participant": 0.30,
            "semantic": 0.25,
            "temporal": 0.15
        }
        
        score = (
            (scene_sim * weights["scene"]) +
            (participant_sim * weights["participant"]) +
            (semantic_sim * weights["semantic"]) +
            (temporal_sim * weights["temporal"])
        )
        
        # Boost if activities strictly overlap (e.g. continuing to loiter)
        if activity_sim > 0.5:
            score += 0.1
            
        logger.debug(f"Continuity Eval: Scene={scene_sim:.2f}, Part={participant_sim:.2f}, Sem={semantic_sim:.2f}, Temp={temporal_sim:.2f} -> Score={score:.2f}")
        return min(1.0, score)

    @classmethod
    def build_candidate_chains(cls, events: List[AggregatedEvent]) -> List[List[AggregatedEvent]]:
        """Process a list of events and group them into candidate semantic chains."""
        if not events:
            return []

        # Sort events chronologically
        sorted_events = sorted(events, key=lambda x: IncidentMemory._time_to_seconds(x.start_time))
        
        # Prepare embeddings for all events to evaluate semantic continuity
        descriptions = [e.narrative_sentence or e.description for e in sorted_events]
        embeddings = EmbeddingService.generate_embeddings(descriptions)
        
        active_memories: List[IncidentMemory] = []
        finalized_chains: List[List[AggregatedEvent]] = []
        
        window_seconds = settings.INCIDENT_CORRELATION_SLIDING_WINDOW_MINUTES * 60
        threshold = settings.INCIDENT_CORRELATION_THRESHOLD

        for idx, event in enumerate(sorted_events):
            event_emb = embeddings[idx]
            event_start = IncidentMemory._time_to_seconds(event.start_time)
            
            # Flush memories outside the sliding window
            retained_memories = []
            for mem in active_memories:
                if (event_start - mem.last_seen_seconds) > window_seconds:
                    finalized_chains.append(mem.events)
                else:
                    retained_memories.append(mem)
            active_memories = retained_memories
            
            # Find best matching active memory
            best_mem = None
            best_score = -1.0
            
            for mem in active_memories:
                score = cls._evaluate_continuity(event, event_emb, mem)
                if score > best_score:
                    best_score = score
                    best_mem = mem
            
            if best_mem and best_score >= threshold:
                best_mem.update(event, event_emb)
            else:
                # Start a new candidate chain
                active_memories.append(IncidentMemory(event, event_emb))
                
        # Finalize remaining active memories
        for mem in active_memories:
            finalized_chains.append(mem.events)
            
        return finalized_chains

    @classmethod
    def correlate_events(cls, events: List[AggregatedEvent]) -> List[IncidentChain]:
        """Legacy wrapper - groups events and formats them as incidents.
           Ideally, NarrativeBuilderService should be used to validate these candidates.
        """
        candidate_chains = cls.build_candidate_chains(events)
        
        # If Gemini is unavailable, we still need to output formatted IncidentChains based on rules.
        # We will use the most severe event in each candidate chain to title it.
        incident_chains = []
        severity_map = {"critical": 100, "high": 80, "medium": 50, "low": 20, "none": 0}
        
        for chain in candidate_chains:
            max_severity_val = 0
            incident_type = None
            description = ""
            
            for e in chain:
                e_sev = 15 if not hasattr(e, "event_severity") else e.event_severity
                max_severity_val = max(max_severity_val, e_sev)
                
                for fe in e.frame_events:
                    fe_sev = severity_map.get(fe.severity.lower(), 20)
                    if fe_sev > max_severity_val:
                        max_severity_val = fe_sev
                        incident_type = fe.event_type
                        description = fe.description
                        
            if not incident_type:
                incident_type = chain[0].event_type
                description = chain[0].narrative_sentence or chain[0].description
                
            if max_severity_val < 50:
                continue # Skip routine activity
                
            severity_str = "LOW"
            if max_severity_val >= 100: severity_str = "CRITICAL"
            elif max_severity_val >= 80: severity_str = "HIGH"
            elif max_severity_val >= 50: severity_str = "MEDIUM"
            
            incident_chains.append(IncidentChain(
                incident_type=incident_type,
                severity=severity_str,
                description=description,
                start_time=chain[0].start_time,
                end_time=chain[-1].end_time,
                chain_events=[e.model_dump() for e in chain],
                timeline=[f"{e.start_time}: {e.narrative_sentence or e.description}" for e in chain],
                recommendations=["Review flagged events."]
            ))
            
        return incident_chains
