# Experimental Actor State

This folder contains a benchmark-only global actor continuity layer.

Input:

- `data/metadata/<video_id>_events.json`

Output:

- `data/output/actor_state_timeline.json`
- `data/output/actor_state_summary.json`
- `data/output/ACTOR_STATE_REPORT.md`

The benchmark does not modify production services. It consumes already
aggregated events and builds persistent actor/object timelines across the
entire video.
