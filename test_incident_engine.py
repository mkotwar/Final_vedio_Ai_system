import sys
import json
from pathlib import Path

ROOT = Path(r"c:\Mukul K\vinfo1\video-search-engine")
sys.path.insert(0, str(ROOT))

from app.schemas.summary import AggregatedEvent
from app.services.incident_engine import IncidentEngine
from app.services.summary_service import SummaryService

def main():
    test_videos = {
        "Fall Video": "284e527c-888c-4c80-96c8-3cd7d50731b3",
        "Office Video": "09c162d9-b006-444b-8783-89b3ed025420",
        "Accident Video": "a48b4d08-7e3c-4aa3-a801-0756875508b8"
    }

    out = []
    
    for name, vid in test_videos.items():
        events = SummaryService.load_events(vid)
        if not events:
            continue
            
        chains = IncidentEngine.correlate_events(events)
        
        out.append(f"====== {name} ======")
        out.append(f"Found {len(chains)} chains.")
        
        for chain in chains:
            out.append(f"-> Incident Type: {chain.incident_type}")
            out.append(f"   Severity: {chain.severity}")
            out.append(f"   Description: {chain.description}")
            out.append(f"   Recommendations: {', '.join(chain.recommendations)}")
            out.append(f"   Chain Events length: {len(chain.chain_events)}")
            out.append(f"   Timeline:\n      " + "\n      ".join(chain.timeline[:3]))
        
        out.append("\n")

    with open(ROOT / "incident_validation.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(out))

    print("Success")

if __name__ == "__main__":
    main()
