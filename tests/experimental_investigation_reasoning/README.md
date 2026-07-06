# Experimental Investigation Reasoning

This folder contains a benchmark-only evidence-based investigation reasoning
layer.

Inputs:

- aggregated events from `data/metadata/<video_id>_events.json`
- actor timelines from `tests/experimental_actor_state/data/output/actor_state_timeline.json`
- evidence graph from `tests/experimental_evidence_graph/data/output/evidence_graph.json`

Outputs:

- `data/output/investigation_reasoning.json`
- `data/output/investigation_reasoning_summary.json`
- `data/output/INVESTIGATION_REASONING_REPORT.md`

The benchmark prompts an LLM with structured evidence only and asks for:

- global narrative
- important activities
- suspicious or unusual observations
- timeline summary
- risk assessment
- supporting evidence

No production code is modified.
