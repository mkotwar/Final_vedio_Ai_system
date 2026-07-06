# Experimental Evidence Graph

This folder contains a benchmark-only evidence graph layer.

Inputs:

- aggregated events from `data/metadata/<video_id>_events.json`
- actor timelines from `tests/experimental_actor_state/data/output/actor_state_timeline.json`

Outputs:

- `data/output/evidence_graph.json`
- `data/output/evidence_graph_summary.json`
- `data/output/EVIDENCE_GRAPH_REPORT.md`

The goal is generic evidence representation only. No incident reasoning is
performed in this layer.
