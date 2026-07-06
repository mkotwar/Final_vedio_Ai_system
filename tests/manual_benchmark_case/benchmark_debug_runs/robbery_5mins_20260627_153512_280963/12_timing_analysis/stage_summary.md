# Stage 12 - Timing Analysis

- Execution time: `0.001s`

## Summary

- stages: `{'frame_extraction': 15.359716400038451, 'event_candidate_layer': 9.740724600036629, 'candidate_event_clustering': 7.203551499987952, 'keyframe_selection': 0.17519049998372793, 'temporal_strip_builder': 0.43013940000673756, 'vlm_inputs': 0.11246789997676387, 'vlm_inference': 239.15629830001853, 'metadata_cleanup': 0.05679900001268834, 'event_aggregation_inputs': 0.004021400003693998, 'benchmark_outputs': 0.020515099982731044}`
- total_pipeline_time: `272.2594241000479`
- bottleneck_stage: `vlm_inference`
- stage_percentages: `{'frame_extraction': 5.64, 'event_candidate_layer': 3.58, 'candidate_event_clustering': 2.65, 'keyframe_selection': 0.06, 'temporal_strip_builder': 0.16, 'vlm_inputs': 0.04, 'vlm_inference': 87.84, 'metadata_cleanup': 0.02, 'event_aggregation_inputs': 0.0, 'benchmark_outputs': 0.01}`