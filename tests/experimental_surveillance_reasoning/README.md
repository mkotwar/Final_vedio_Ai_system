# Experimental Surveillance Reasoning

This folder contains a benchmark-only reasoning stack that runs after the
existing production pipeline has already produced:

- frame metadata
- aggregated events
- existing VLM outputs embedded in frame metadata

Flow:

1. `run_event_candidate_layer()` wraps the existing production pipeline through
   `FrameExtractionService.extract_frames(video_id)`.
2. The benchmark then loads:
   - `data/metadata/<video_id>_frames.json`
   - `data/metadata/<video_id>_events.json`
3. Experimental layers are applied:
   - Actor State Builder
   - Evidence Graph Builder
   - Investigation Reasoner

No production code is modified by this benchmark.

Primary script:

- `run_experimental_surveillance_reasoning_benchmark.py`

Outputs:

- `data/output/experimental_surveillance_summary.json`
- `data/output/experimental_surveillance_summary.md`
- `data/output/actor_states.json`
- `data/output/evidence_graph.json`
- `data/output/investigation_reasoning.json`
