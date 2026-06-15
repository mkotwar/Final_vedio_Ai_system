import json
from app.services.incident_engine import IncidentEngine
from app.schemas.summary import AggregatedEvent

def main():
    e1 = AggregatedEvent(start_time="00:00:00", end_time="00:00:10", scene_context="parking lot", participants=["man"], activities=["walking"], description="A man is walking in the parking lot.")
    e2 = AggregatedEvent(start_time="00:00:15", end_time="00:00:20", scene_context="parking lot", participants=["man", "car"], activities=["breaking in", "theft"], description="A man breaks into a car.")
    e3 = AggregatedEvent(start_time="00:05:00", end_time="00:05:10", scene_context="store interior", participants=["woman"], activities=["shopping"], description="A woman is shopping.")
    
    chains = IncidentEngine.build_candidate_chains([e1, e2, e3])
    
    with open("test_engine_out.txt", "w") as f:
        f.write(f"Found {len(chains)} chains.\n")
        for i, c in enumerate(chains):
            f.write(f"Chain {i+1}:\n")
            for e in c:
                f.write(f"  {e.start_time} - {e.description}\n")

if __name__ == "__main__":
    main()
