# Experimental Retrieval Investigation Reasoning

This folder contains a benchmark-only retrieval-augmented investigation
reasoning layer.

Inputs:

- aggregated events from `data/metadata/<video_id>_events.json`
- actor timelines from `tests/experimental_actor_state/data/output/actor_state_timeline.json`
- evidence graph from `tests/experimental_evidence_graph/data/output/evidence_graph.json`
- historical aggregated event catalogs from `data/metadata/*_events.json`

Outputs:

- `data/output/retrieval_investigation_baseline.json`
- `data/output/retrieval_investigation_augmented.json`
- `data/output/retrieval_candidates.json`
- `data/output/retrieval_benchmark_metrics.json`
- `data/output/RETRIEVAL_INVESTIGATION_REPORT.md`

The benchmark runs:

1. baseline reasoning with current-video evidence only
2. retrieval-augmented reasoning with top-k similar historical evidence
3. comparison metrics across both outputs

No production code is modified.
