import json
import os
import sys

vid = "fef5394a-b684-4862-963a-d4ef12682f43"
meta_dir = r"c:\Mukul K\vinfo1\video-search-engine\data\metadata"
events_file = os.path.join(meta_dir, f"{vid}_events_v2.json")
status_file = os.path.join(meta_dir, f"{vid}_status.json")

out_file = r"c:\Mukul K\vinfo1\video-search-engine\output.txt"

sys.path.insert(0, r"c:\Mukul K\vinfo1\video-search-engine")
try:
    from app.services.incident_engine import IncidentEngine
    from app.services.summary_service import SummaryService
except Exception as e:
    with open(out_file, "w") as f:
        f.write(f"Import error: {e}")
    sys.exit(1)

with open(out_file, "w") as out:
    try:
        with open(status_file, "r") as f:
            st = json.load(f)
            processed_frames = st.get('processed_frames', 0)
            out.write(f"Processed frames: {processed_frames}\n")
    except Exception as e:
        out.write(f"Error reading status: {e}\n")

    try:
        events = SummaryService.load_events(vid)
        out.write(f"Events generated: {len(events)}\n")
        
        notable = [e for e in events if e.get("event_severity", 0) > 40 or e.get("event_type", "").startswith("collision")]
        out.write(f"Notable events: {len(notable)}\n")
        
        chains = IncidentEngine.correlate_events(events)
        out.write(f"Incidents generated: {len(chains)}\n\n")

        out.write("INCIDENTS LIST:\n")
        for idx, inc in enumerate(chains):
            out.write(f"--- Incident {idx+1} ---\n")
            out.write(f"Type: {inc.incident_type}\n")
            out.write(f"Start: {inc.start_time}\n")
            out.write(f"End: {inc.end_time}\n")
            out.write(f"Severity: {inc.severity}\n")
            out.write(f"Desc: {inc.description}\n")
            participants = []
            for ev in inc.chain_events:
                participants.extend(ev.get("participants", []))
            parts = list(set(participants))
            out.write(f"Participants: {', '.join(parts) if parts else 'None'}\n\n")

    except Exception as e:
        out.write(f"Error processing events: {e}\n")
