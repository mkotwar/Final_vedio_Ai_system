import json
import os

vid = "fef5394a-b684-4862-963a-d4ef12682f43"
meta_dir = r"c:\Mukul K\vinfo1\video-search-engine\data\metadata"
events_file = os.path.join(meta_dir, f"{vid}_events_v2.json")
status_file = os.path.join(meta_dir, f"{vid}_status.json")

out_file = r"c:\Mukul K\vinfo1\video-search-engine\fast_output.txt"

with open(out_file, "w") as out:
    try:
        with open(status_file, "r") as f:
            st = json.load(f)
            processed_frames = st.get('processed_frames', 0)
            events_generated = st.get('events_generated', 0)
            out.write(f"Processed frames: {processed_frames}\n")
            out.write(f"Events generated status: {events_generated}\n")
    except Exception as e:
        out.write(f"Error reading status: {e}\n")

    try:
        with open(events_file, "r") as f:
            events = json.load(f)
            out.write(f"Events generated array: {len(events)}\n")
        
        notable = [e for e in events if e.get("event_severity", 0) > 40 or e.get("event_type", "").startswith("collision")]
        out.write(f"Notable events: {len(notable)}\n")
        
        # Correlate events manually (since I know the logic is 60 seconds)
        def time_to_seconds(time_str):
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

        sorted_events = sorted(events, key=lambda x: time_to_seconds(x.get("start_time", "00:00:00")))
        chains = []
        if sorted_events:
            current_chain = [sorted_events[0]]
            for event in sorted_events[1:]:
                prev_event = current_chain[-1]
                prev_end = time_to_seconds(prev_event.get("end_time", "00:00:00"))
                curr_start = time_to_seconds(event.get("start_time", "00:00:00"))
                if curr_start - prev_end <= 60:
                    current_chain.append(event)
                else:
                    chains.append(current_chain)
                    current_chain = [event]
            if current_chain:
                chains.append(current_chain)
                
        out.write(f"Incidents generated (chains): {len(chains)}\n\n")

        out.write("INCIDENTS LIST:\n")
        for idx, chain in enumerate(chains):
            max_sev = max([e.get("event_severity", 0) for e in chain])
            if max_sev < 50:
                continue # Skip non-notable chains
            start = chain[0].get("start_time")
            end = chain[-1].get("end_time")
            out.write(f"--- Incident {idx+1} ---\n")
            out.write(f"Start: {start} | End: {end}\n")
            out.write(f"Max Severity: {max_sev}\n")
            parts = []
            for e in chain:
                parts.extend(e.get("participants", []))
            out.write(f"Participants: {len(set(parts))}\n\n")

    except Exception as e:
        out.write(f"Error processing events: {e}\n")
