import json
import os
import sys

vid = "fef5394a-b684-4862-963a-d4ef12682f43"
meta_dir = r"c:\Mukul K\vinfo1\video-search-engine\data\metadata"
events_file = os.path.join(meta_dir, f"{vid}_events_v2.json")
status_file = os.path.join(meta_dir, f"{vid}_status.json")

# Add project root to path
sys.path.insert(0, r"c:\Mukul K\vinfo1\video-search-engine")
from app.services.incident_engine import IncidentEngine
from app.services.summary_service import SummaryService

out_file = r"C:\Users\Vinfocom\.gemini\antigravity-ide\brain\5f667da1-5837-403c-bc9f-536233337705\audit_results.md"

with open(out_file, "w") as out:
    out.write("# Long Video Multi-Incident Audit\n\n")

    # Status
    try:
        with open(status_file, "r") as f:
            st = json.load(f)
            total_frames = st.get('total_frames', 0)
            processed_frames = st.get('processed_frames', 0)
            events_generated = st.get('events_generated', 0)
            out.write(f"- **Total frames processed**: {processed_frames}\n")
    except Exception as e:
        out.write(f"Error reading status: {e}\n")

    # Events
    try:
        events = SummaryService.load_events(vid)
        out.write(f"- **Total events generated**: {len(events)}\n")
        
        # notable events
        notable = [e for e in events if e.get("event_severity", 0) > 40 or e.get("event_type", "").startswith("collision")]
        out.write(f"- **Total notable events**: {len(notable)}\n")
        
        # Incidents
        chains = IncidentEngine.correlate_events(events)
        out.write(f"- **Total incidents generated**: {len(chains)}\n\n")

        # Search records
        # Assuming QdrantManager
        try:
            from app.core.qdrant_manager import QdrantManager
            qm = QdrantManager()
            collection_info = qm.client.get_collection(collection_name="video_events")
            count = collection_info.points_count
            out.write(f"- **Total search records indexed (overall db)**: {count}\n\n")
        except Exception as e:
            out.write(f"- **Total search records indexed**: Could not connect to Qdrant ({e})\n\n")

        out.write("## Incidents List\n\n")
        for idx, inc in enumerate(chains):
            out.write(f"### Incident {idx+1}: {inc.incident_type}\n")
            out.write(f"- **Start time**: {inc.start_time}\n")
            out.write(f"- **End time**: {inc.end_time}\n")
            out.write(f"- **Severity**: {inc.severity}\n")
            participants = []
            for ev in inc.chain_events:
                participants.extend(ev.get("participants", []))
            parts = list(set(participants))
            out.write(f"- **Participants**: {', '.join(parts) if parts else 'None'}\n\n")
            
        out.write("## Ground Truth vs System Timeline\n")
        out.write("### System Timeline (Notable Events):\n")
        for ev in notable:
            out.write(f"- {ev.get('start_time')} - {ev.get('end_time')} | {ev.get('event_type')} | Severity: {ev.get('event_severity')}\n")

    except Exception as e:
        out.write(f"Error processing events: {e}\n")
