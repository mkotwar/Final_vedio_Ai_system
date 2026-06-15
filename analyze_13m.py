import json
import os
from collections import defaultdict

vid = "fef5394a-b684-4862-963a-d4ef12682f43"
meta_dir = r"C:\Mukul K\vinfo1\video-search-engine\data\metadata"
events_file = os.path.join(meta_dir, f"{vid}_events_v2.json")
status_file = os.path.join(meta_dir, f"{vid}_status.json")
frames_file = os.path.join(meta_dir, f"{vid}_frames.json")

out_file = r"C:\Mukul K\vinfo1\video-search-engine\13m_audit_output.txt"

with open(out_file, "w") as out:
    # 1. Total frames processed
    if os.path.exists(status_file):
        with open(status_file, "r") as f:
            st = json.load(f)
            out.write(f"Total frames processed: {st.get('processed_frames')}\n")
    
    # Events and Incidents
    if os.path.exists(events_file):
        with open(events_file, "r") as f:
            events = json.load(f)
            out.write(f"Total events generated (raw): {len(events)}\n\n")
            
            # Print details of high severity / collision events to see what we have
            incidents = [e for e in events if e.get("event_type") in ("collision_or_accident", "collision", "fire_incident", "injury_or_fall") or e.get("event_severity", 0) > 50]
            out.write(f"Total notable events/incidents found: {len(incidents)}\n\n")
            
            for inc in incidents:
                out.write(f"Incident: {inc.get('event_id')} | Type: {inc.get('event_type')}\n")
                out.write(f"Start: {inc.get('start_time')} | End: {inc.get('end_time')}\n")
                out.write(f"Severity: {inc.get('event_severity')}\n")
                out.write(f"Participants: {', '.join(inc.get('participants', []))}\n")
                out.write(f"Desc: {inc.get('description')}\n")
                out.write("-" * 40 + "\n")
