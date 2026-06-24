# Manual Benchmark Case

This folder contains a reusable benchmark harness that:

- imports and uses the production pipeline
- forces `BATCH_SIZE=4`
- runs one real input video through the pipeline
- saves a copy of the input video
- saves an output video made from the exact frames sent to the VLM
- writes summary JSON/Markdown with latency and frame-selection counts

Primary script:

- `run_vlm_candidate_benchmark.py`

Output folder:

- `data/`
