# Stage 12 - Timing Analysis

- Execution time: `0.001s`

## Summary

- stages: `{'frame_extraction': 19.516933399951085, 'event_candidate_layer': 20.753031100030057, 'candidate_event_clustering': 9.627632799965795, 'keyframe_selection': 0.37292799999704584, 'temporal_strip_builder': 0.6632400000235066, 'vlm_inputs': 0.1252041999832727, 'vlm_inference': 330.08908740000334, 'metadata_cleanup': 0.08980800001882017, 'event_aggregation_inputs': 0.006659500009845942, 'benchmark_outputs': 0.03587209997931495}`
- total_pipeline_time: `381.2803964999621`
- bottleneck_stage: `vlm_inference`
- stage_percentages: `{'frame_extraction': 5.12, 'event_candidate_layer': 5.44, 'candidate_event_clustering': 2.53, 'keyframe_selection': 0.1, 'temporal_strip_builder': 0.17, 'vlm_inputs': 0.03, 'vlm_inference': 86.57, 'metadata_cleanup': 0.02, 'event_aggregation_inputs': 0.0, 'benchmark_outputs': 0.01}`