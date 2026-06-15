import json
import time
import psutil
import os
import math
from typing import List
from loguru import logger

from app.schemas.summary import AggregatedEvent
from app.services.incident_engine import IncidentEngine, IncidentMemory
from app.services.narrative_builder import NarrativeBuilderService
from app.services.embedding_service import EmbeddingService

# Video IDs
VIDEOS = {
    "Theft Video": "03301eee-50a4-4a3a-b5db-11f29b339233",
    "Crash Compilation": "09c162d9-b006-444b-8783-89b3ed025420",
    "Elderly Fall Video": "a629ff6d-ae6a-4fbc-ac3d-bffe9952c8bd"
}

def load_events(vid: str) -> List[AggregatedEvent]:
    try:
        with open(f"data/metadata/{vid}_events_v2.json", 'r') as f:
            data = json.load(f)
        return [AggregatedEvent(**e) for e in data]
    except Exception as e:
        print(f"Failed to load {vid}: {e}")
        return []

def synthesize_cctv_events() -> List[AggregatedEvent]:
    return [
        AggregatedEvent(start_time="08:00:00", end_time="08:00:15", scene_context="Front door lobby", participants=["employee in uniform"], activities=["entering building"], description="An employee arrives at work."),
        AggregatedEvent(start_time="08:15:00", end_time="08:15:30", scene_context="Front door lobby", participants=["employee in uniform"], activities=["exiting building"], description="The employee leaves the building."),
        AggregatedEvent(start_time="10:00:00", end_time="10:00:20", scene_context="Front door lobby", participants=["unknown visitor"], activities=["entering building"], description="A visitor enters the building."),
        AggregatedEvent(start_time="10:05:00", end_time="10:06:00", scene_context="Front door lobby", participants=["unknown visitor"], activities=["loitering", "looking around"], description="The visitor loiters near the reception desk."),
        AggregatedEvent(start_time="10:15:00", end_time="10:15:10", scene_context="Front door lobby", participants=["unknown visitor"], activities=["exiting building"], description="The visitor exits."),
        AggregatedEvent(start_time="13:00:00", end_time="13:01:00", scene_context="Loading dock", participants=["delivery driver", "truck"], activities=["arriving", "parking"], description="A delivery truck arrives."),
        AggregatedEvent(start_time="13:05:00", end_time="13:06:00", scene_context="Loading dock", participants=["delivery driver"], activities=["unloading packages"], description="Driver drops off packages."),
        AggregatedEvent(start_time="13:20:00", end_time="13:20:15", scene_context="Loading dock", participants=["truck"], activities=["driving away"], description="Truck leaves the loading dock.")
    ]

def capture_decisions(events: List[AggregatedEvent], window_minutes: int=15, threshold: float=0.65):
    """Run an instrumented version of build_candidate_chains to capture scores."""
    decisions = []
    memories = []
    
    sorted_events = sorted(events, key=lambda x: IncidentMemory._time_to_seconds(x.start_time))
    descriptions = [e.narrative_sentence or e.description for e in sorted_events]
    embeddings = EmbeddingService.generate_embeddings(descriptions)
    
    active_memories: List[IncidentMemory] = []
    window_seconds = window_minutes * 60

    for idx, event in enumerate(sorted_events):
        event_emb = embeddings[idx]
        event_start = IncidentMemory._time_to_seconds(event.start_time)
        
        retained_memories = []
        for mem in active_memories:
            if (event_start - mem.last_seen_seconds) <= window_seconds:
                retained_memories.append(mem)
        active_memories = retained_memories
        
        best_mem = None
        best_score = -1.0
        best_features = {}
        
        for mem in active_memories:
            # Re-implement eval to capture features
            mem_embedding = mem.get_aggregated_embedding()
            semantic_sim = IncidentEngine._calculate_cosine_similarity(event_emb, mem_embedding)
            scene_sim = 1.0 if event.scene_context in mem.scene_contexts else 0.0
            
            event_actors = set(event.participants)
            if event.actor_description: event_actors.add(event.actor_description)
            actor_sim = IncidentEngine._calculate_jaccard(event_actors, mem.actors)
            event_objects = set([str(obj) for obj in event.objects])
            object_sim = IncidentEngine._calculate_jaccard(event_objects, mem.objects)
            participant_sim = max(actor_sim, object_sim)
            
            activity_sim = IncidentEngine._calculate_jaccard(set(event.activities), mem.activities)
            
            gap = max(0, event_start - mem.last_seen_seconds)
            temporal_sim = math.exp(-gap / 300.0)
            
            score = (scene_sim * 0.30) + (participant_sim * 0.30) + (semantic_sim * 0.25) + (temporal_sim * 0.15)
            if activity_sim > 0.5:
                score += 0.1
            score = min(1.0, score)
            
            if score > best_score:
                best_score = score
                best_mem = mem
                best_features = {
                    "scene_similarity": round(scene_sim, 2),
                    "participant_similarity": round(participant_sim, 2),
                    "activity_similarity": round(activity_sim, 2),
                    "narrative_similarity": round(semantic_sim, 2),
                    "temporal_similarity": round(temporal_sim, 2),
                    "final_score": round(score, 2),
                    "decision": "MERGE" if score >= threshold else "SPLIT",
                    "event_desc": event.description,
                    "mem_first_desc": mem.events[0].description
                }
                
        if best_mem and best_score >= threshold:
            best_mem.update(event, event_emb)
            decisions.append(best_features)
        else:
            if best_features:
                decisions.append(best_features)
            new_mem = IncidentMemory(event, event_emb)
            active_memories.append(new_mem)
            memories.append(new_mem)
            
    return memories, decisions

def run_audit():
    out = {}
    out['tests'] = {}
    out['decisions'] = []
    out['memory_snapshots'] = []
    
    process = psutil.Process(os.getpid())
    start_mem = process.memory_info().rss
    
    total_corr_time = 0
    total_emb_calls = 0
    total_gemini_calls = 0
    
    for name, vid in VIDEOS.items():
        print(f"Processing {name}...")
        events = load_events(vid)
        if not events: continue
        
        t0 = time.time()
        mems, decs = capture_decisions(events)
        total_corr_time += (time.time() - t0)
        total_emb_calls += len(events)
        out['decisions'].extend(decs)
        
        for m in mems:
            out['memory_snapshots'].append({
                "test": name,
                "first_seen": m.events[0].start_time,
                "last_seen": m.events[-1].end_time,
                "actors": list(m.actors),
                "objects": list(m.objects),
                "activities": list(m.activities),
                "event_count": len(m.events)
            })
            
        chains = [m.events for m in mems]
        
        t1 = time.time()
        final_incidents = NarrativeBuilderService.generate_narrative_from_events(events)
        total_gemini_calls += 1
        
        out['tests'][name] = {
            "frames": "Unknown (Pre-extracted)",
            "events_generated": len(events),
            "candidate_chains": len(chains),
            "final_incidents": len(final_incidents),
            "final_incidents_data": [inc.model_dump() for inc in final_incidents]
        }
        
    print("Processing CCTV Simulation...")
    cctv_events = synthesize_cctv_events()
    cctv_chains = IncidentEngine.build_candidate_chains(cctv_events)
    total_emb_calls += len(cctv_events)
    cctv_incidents = NarrativeBuilderService.generate_narrative_from_events(cctv_events)
    total_gemini_calls += 1
    
    out['tests']["CCTV Simulation"] = {
        "frames": 0,
        "events_generated": len(cctv_events),
        "candidate_chains": len(cctv_chains),
        "final_incidents": len(cctv_incidents),
        "final_incidents_data": [inc.model_dump() for inc in cctv_incidents]
    }
    
    end_mem = process.memory_info().rss
    
    out['performance'] = {
        "correlation_time_seconds": round(total_corr_time, 3),
        "embedding_calls": total_emb_calls,
        "gemini_calls": total_gemini_calls,
        "peak_memory_mb": round((end_mem - start_mem) / (1024*1024), 2)
    }
    
    with open("audit_12_6b_results.json", "w") as f:
        json.dump(out, f, indent=2)
        
    print("Audit data saved.")

if __name__ == "__main__":
    run_audit()
